#!/usr/bin/env python3
import os
from typing import Tuple, Union

import deepspeed
import copy
import numpy as np
import torch
from deepspeed.constants import TORCH_DISTRIBUTED_DEFAULT_PORT
from fire import Fire
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import GpuInfo
from ignite.utils import setup_logger
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.engines.utils import CommonKeys
from monai.handlers import (
    CheckpointSaver,
    CheckpointLoader,
    LrScheduleHandler,
    SegmentationSaver,
)
from torch.utils.tensorboard import SummaryWriter

from src.engines.trainer import AdversarialTrainer
from src.handlers.general import (
    TensorBoardHandler,
    LossSummaryHandler,
    LoggingPreparationHandler,
    EvaluationHandler,
    NpySaver,
    MaxEpochsHandler,
)
from src.handlers.vqvae import (
    VQVAELoggingHandler,
    AdversarialFinetuneHandler,
    TransformTraceLoggerHandler,
)
from src.inferer.vqvae import VQVAEExtractionInferer, VQVAEDecodingInferer
from src.losses.adversarial.configure import (
    get_discriminator_loss,
    get_generator_loss,
    get_criterion,
)
from src.losses.vqvae.configure import get_vqvae_loss, add_vqvae_loss_handlers
from src.metrics.vqvae import MAE, MSE, MultiScaleSSIM
from src.networks.discriminator.configure import get_discriminator_network
from src.networks.vqvae.configure import get_vqvae_network, add_vqvae_network_handlers
from src.utils.general import get_gamma, basic_initialization, log_network_size
from src.utils.vqvae import (
    get_data_flow,
    get_ms_ssim_window,
    prepare_batch,
    prepare_decoding_batch,
    VQVAEModes,
    get_batch_transform,
)


def training(config: dict) -> None:
    logger_base_name = "VQVAE-Training"

    logger, device = basic_initialization(config=config, logger_name=logger_base_name)

    training_loader, evaluation_loader, training_evaluation_loader = get_data_flow(
        config=config, logger=logger
    )

    network = get_vqvae_network(config=config).to(device)
    log_network_size(network=network, logger=logger)

    if config["device"] == "ddp":
        network = torch.nn.parallel.DistributedDataParallel(
            network,
            device_ids=[config["local_rank"]],
            broadcast_buffers=False,
            bucket_cap_mb=12.5,
        )

    loss_function = get_vqvae_loss(config=config)
    loss_function = loss_function.to(device)

    optimizer = torch.optim.Adam(network.parameters(), config["learning_rate"])

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=(
            config["gamma"]
            if isinstance(config["gamma"], float)
            else get_gamma(config=config, logger=logger)
        ),
    )

    if config["adversarial_component"]:
        d_logger = setup_logger(name=logger_base_name + "-Discriminator")
        d_network = get_discriminator_network(config=config).to(device)

        if config["device"] == "ddp":
            d_network = torch.nn.parallel.DistributedDataParallel(
                d_network,
                device_ids=[config["local_rank"]],
                broadcast_buffers=False,
                bucket_cap_mb=12.5,
            )

        log_network_size(network=d_network, logger=d_logger)

        d_optimizer = torch.optim.Adam(
            d_network.parameters(), config["discriminator_learning_rate"]
        )
        d_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            d_optimizer,
            gamma=(
                config["gamma"]
                if isinstance(config["gamma"], float)
                else get_gamma(config=config, logger=d_logger)
            ),
        )

        discriminator_loss = get_discriminator_loss(config=config).to(device)
        generator_loss = get_generator_loss(config=config).to(device)

    ms_ssim_kwargs = {"win_size": get_ms_ssim_window(config=config, logger=logger)}

    key_metric = {
        f"Metric-MS-SSIM_{ms_ssim_kwargs['win_size']}-Reconstruction": MultiScaleSSIM(
            output_transform=lambda network_output: (
                network_output[CommonKeys.PRED]["reconstruction"][0],
                network_output[CommonKeys.IMAGE],
            ),
            ms_ssim_kwargs=ms_ssim_kwargs,
        )
    }

    additional_metrics = {
        "Metric-MAE-Reconstruction": MAE(
            output_transform=lambda network_output: (
                network_output[CommonKeys.PRED]["reconstruction"][0],
                network_output[CommonKeys.IMAGE],
            )
        ),
        "Metric-MSE-Reconstruction": MSE(
            output_transform=lambda network_output: (
                network_output[CommonKeys.PRED]["reconstruction"][0],
                network_output[CommonKeys.IMAGE],
            )
        ),
    }

    # Preparing the trainer handlers
    train_handlers = (
        [
            # The LoggingPreparationHandler should be the firs in the list due to the fact it
            # prepares the engine.state.output to store summaries since our TensorboardLogger
            # is a more feature rich and deviated from the MONAi one.
            LoggingPreparationHandler()
        ]
        if config["rank"] == 0
        else []
    )

    train_handlers += [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=False, epoch_level=False)
    ]

    if config["adversarial_component"]:
        train_handlers += [
            LrScheduleHandler(
                lr_scheduler=d_lr_scheduler, print_lr=False, epoch_level=False
            ),
            LossSummaryHandler(loss=generator_loss),
            LossSummaryHandler(loss=discriminator_loss),
        ]

        if config["finetune_adversarial_component"]:
            train_handlers += [
                AdversarialFinetuneHandler(
                    network=network,
                    adversarial_loss_range=config["finetune_adversarial_component"],
                    adversarial_loss_patience=config["finetune_patience"],
                )
            ]

    train_handlers = add_vqvae_network_handlers(
        train_handlers=train_handlers,
        vqvae=network.module if config["device"] == "ddp" else network,
        config=config,
        logger=logger,
    )
    train_handlers = add_vqvae_loss_handlers(
        train_handlers=train_handlers, loss_function=loss_function, config=config
    )

    if config["rank"] == 0:
        train_handlers += [
            VQVAELoggingHandler(
                network=network.module if config["device"] == "ddp" else network,
                log_3d=config["save_3d_images"],
                log_2d=config["save_2d_images"],
            ),
            LossSummaryHandler(loss=loss_function),
            # Ideally this should be the last handler so you make sure that every other handler
            # has already left the summaries in the engine.state.output["summaries"] dictionary
            TensorBoardHandler(
                summary_writer=SummaryWriter(config["logs_directory"] + "train/"),
                interval=config["log_every"],
                epoch_level=True,
                clamp_images=True,
                clamp_range=(0.0, 1.0),
            ),
        ]

    if config["adversarial_component"]:
        trainer = AdversarialTrainer(
            device=device,
            max_epochs=config["epochs"] if not config["training_epoch_length"] else 1,
            train_data_loader=training_loader,
            g_network=network,
            g_optimizer=optimizer,
            g_loss_function=generator_loss,
            recon_loss_function=loss_function,
            d_network=d_network,
            d_optimizer=d_optimizer,
            d_loss_function=discriminator_loss,
            epoch_length=config["training_epoch_length"],
            non_blocking=True,
            prepare_batch=prepare_batch,
            train_handlers=train_handlers,
            amp=config["amp"],
            use_adversarial_adaptive_weight=config["use_adversarial_adaptive_weight"],
            adaptive_adversarial_weight_threshold=config[
                "adaptive_adversarial_weight_threshold"
            ],
            adaptive_adversarial_weight_value=config[
                "adaptive_adversarial_weight_value"
            ],
        )
    else:
        trainer = SupervisedTrainer(
            device=device,
            max_epochs=config["epochs"] if not config["training_epoch_length"] else 1,
            non_blocking=True,
            train_data_loader=training_loader,
            epoch_length=config["training_epoch_length"],
            network=network,
            optimizer=optimizer,
            loss_function=loss_function,
            prepare_batch=prepare_batch,
            train_handlers=train_handlers,
            amp=config["amp"],
        )

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=evaluation_loader,
        non_blocking=True,
        network=network,
        prepare_batch=prepare_batch,
        key_val_metric=key_metric,
        additional_metrics=additional_metrics,
        amp=config["amp"],
        val_handlers=[
            LoggingPreparationHandler(),
            VQVAELoggingHandler(
                network=network.module if config["device"] == "ddp" else network,
                log_3d=config["save_3d_images"],
                log_2d=config["save_2d_images"],
                is_eval=True,
            ),
            TensorBoardHandler(
                summary_writer=SummaryWriter(config["logs_directory"] + "val_eval/"),
                interval=1,
                epoch_level=True,
                clamp_images=True,
                clamp_range=(0.0, 1.0),
                global_step_transform=lambda x: trainer.state.epoch,
            ),
        ]
        if config["rank"] == 0
        else None,
    )

    training_evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=training_evaluation_loader,
        non_blocking=True,
        network=network,
        prepare_batch=prepare_batch,
        key_val_metric=key_metric,
        additional_metrics=additional_metrics,
        amp=config["amp"],
        val_handlers=[
            LoggingPreparationHandler(),
            VQVAELoggingHandler(
                network=network.module if config["device"] == "ddp" else network,
                log_3d=config["save_3d_images"],
                log_2d=config["save_2d_images"],
                is_eval=True,
            ),
            TensorBoardHandler(
                summary_writer=SummaryWriter(config["logs_directory"] + "train_eval/"),
                interval=1,
                epoch_level=True,
                clamp_images=True,
                clamp_range=(0.0, 1.0),
                global_step_transform=lambda x: trainer.state.epoch,
            ),
        ]
        if config["rank"] == 0
        else None,
    )

    to_save = {
        "network": network,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "trainer": trainer,
    }

    if config["adversarial_component"]:
        to_save.update(
            {
                "d_network": d_network,
                "d_optimizer": d_optimizer,
                "d_lr_scheduler": d_lr_scheduler,
            }
        )

    # The adversarial components are only loaded if we have adversarial component set to true and if finetuning is not
    #   used
    if config["finetune_adversarial_component"]:
        to_load = copy.copy(to_save)
        to_load.pop("d_network")
        to_load.pop("d_optimizer")
        to_load.pop("d_lr_scheduler")
    else:
        to_load = to_save

    if config.get("checkpoint_fp", None):
        # The warning is due to bad type hinting from MONAI. Internally the map_location is passed as follows
        #   checkpoint = torch.load(self.load_path, map_location=self.map_location)
        #   torch.load allows map_location to be a torch.device, so the following code is valid.
        CheckpointLoader(
            load_path=config["checkpoint_fp"], load_dict=to_load, map_location=device
        ).attach(trainer)

    CheckpointSaver(
        save_dir=config["checkpoint_directory"],
        save_dict=to_save,
        epoch_level=True,
        save_interval=config["checkpoint_every"],
        n_saved=1,
    ).attach(trainer)

    CheckpointSaver(
        save_dir=config["checkpoint_directory"],
        save_dict=to_save,
        epoch_level=True,
        save_key_metric=True,
        key_metric_name=evaluator.state.key_metric_name,
        key_metric_n_saved=1,
    ).attach(evaluator)

    MaxEpochsHandler(max_epochs=config["epochs"]).attach(trainer)

    # We manually attach those handlers since we need the trainer to be already defined for the global_step_transform
    # to give us the epoch of the main trainer and not the evaluators. Otherwise it will be registered continously as
    # the epoch 0.
    EvaluationHandler(
        evaluation_engine=evaluator, evaluate_every=config["eval_every"]
    ).attach(trainer)
    EvaluationHandler(
        evaluation_engine=training_evaluator, evaluate_every=config["eval_every"]
    ).attach(trainer)

    if config["rank"] == 0:
        GpuInfo().attach(trainer, name="gpu")

    ProgressBar(
        persist=True,
        bar_format="[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{postfix} [{elapsed}<{remaining}]",
    ).attach(
        trainer,
        metric_names=["gpu:0 mem(%)", "gpu:0 util(%)"],
        output_transform=lambda output: {"Loss": output[CommonKeys.LOSS]},
    )

    trainer.run()

    torch.save(
        network.state_dict(),
        f"{config['checkpoint_directory']}model_state_dict_epoch={trainer.state.epoch}.pt",
    )


def inference(config: dict) -> None:
    logger_base_name = f"VQVAE-Inference-{config['mode']}"

    is_extracting = config["mode"] == VQVAEModes.EXTRACTING.value

    logger, device = basic_initialization(config=config, logger_name=logger_base_name)

    _, evaluation_loader, _ = get_data_flow(config=config, logger=logger)

    network = get_vqvae_network(config=config).to(device)
    log_network_size(network=network, logger=logger)

    if config["device"] == "ddp":
        network = torch.nn.parallel.DistributedDataParallel(
            network,
            device_ids=[config["local_rank"]],
            broadcast_buffers=False,
            bucket_cap_mb=12.5,
        )

    d_network = None
    if config["adversarial_component"]:
        d_logger = setup_logger(name=logger_base_name + "-Discriminator")
        d_network = get_discriminator_network(config=config).to(device)

        if config["device"] == "ddp":
            d_network = torch.nn.parallel.DistributedDataParallel(
                d_network,
                device_ids=[config["local_rank"]],
                broadcast_buffers=False,
                bucket_cap_mb=12.5,
            )

        log_network_size(network=d_network, logger=d_logger)

    engine = SupervisedEvaluator(
        device=device,
        val_data_loader=evaluation_loader,
        inferer=VQVAEExtractionInferer(d_network=d_network)
        if is_extracting
        else VQVAEDecodingInferer(
            num_quantization_levels=len(config["embedding_dim"]), d_network=d_network
        ),
        non_blocking=True,
        network=network,
        prepare_batch=prepare_batch
        if is_extracting
        else lambda batch, device, non_blocking: prepare_decoding_batch(
            batch, len(config["num_embeddings"]), device, non_blocking
        ),
        amp=False,
        val_handlers=[
            TransformTraceLoggerHandler(
                output_dir=config["outputs_directory"],
                metadata_key="MRI_meta_dict",
                trace_key="MRI_trace_dict",
            )
        ] if config["mode"] == VQVAEModes.EXTRACTING.value else [],
    )

    if config.get("checkpoint_fp", None):
        # The warning is due to bad type hinting from MONAI. Internally the map_location is passed as follows
        #   checkpoint = torch.load(self.load_path, map_location=self.map_location)
        #   torch.load allows map_location to be a torch.device, so the following code is valid.
        load_dict = {"network": network}
        if config["adversarial_component"]:
            load_dict.update({"d_network": d_network})

        CheckpointLoader(
            load_path=config["checkpoint_fp"], load_dict=load_dict, map_location=device
        ).attach(engine)

    if config["mode"] == VQVAEModes.EXTRACTING.value:
        SegmentationSaver(
            output_dir=config["outputs_directory"],
            output_postfix="reconstruction",
            output_ext=".nii.gz",
            resample=False,
            scale=None,
            dtype=np.dtype(np.float32),
            batch_transform=get_batch_transform(
                no_augmented_extractions=config["no_augmented_extractions"],
                is_nii_based=True,
                filename_or_objs_only=False,
                mode=config["mode"],
            ),
            output_transform=lambda output: output[CommonKeys.PRED]["reconstruction"],
        ).attach(engine)

        for i in range(len(config["embedding_dim"])):
            NpySaver(
                output_dir=config["outputs_directory"],
                output_postfix=f"quantization_{i}",
                dtype=np.dtype(np.uint16),
                batch_transform=get_batch_transform(
                    no_augmented_extractions=config["no_augmented_extractions"],
                    is_nii_based=True,
                    filename_or_objs_only=True,
                    mode=config["mode"],
                ),
                output_transform=lambda output: output[CommonKeys.PRED][
                    f"quantization_{i}"
                ],
            ).attach(engine)
    else:
        SegmentationSaver(
            output_dir=config["outputs_directory"],
            output_postfix="sample",
            output_ext=".nii.gz",
            resample=False,
            scale=None,
            dtype=np.dtype(np.float32),
            batch_transform=get_batch_transform(
                no_augmented_extractions=0,
                is_nii_based=False,
                filename_or_objs_only=True,
                mode=config["mode"],
            ),
            output_transform=lambda output: output[CommonKeys.PRED]["sample"],
        ).attach(engine)

    if config["adversarial_component"]:
        adversarial_criterion = get_criterion(config["generator_loss"])
        NpySaver(
            output_dir=config["outputs_directory"],
            output_postfix=f"adversarial_loss",
            dtype=np.dtype(np.float32),
            batch_transform=get_batch_transform(
                no_augmented_extractions=config["no_augmented_extractions"],
                is_nii_based=config["mode"] == VQVAEModes.EXTRACTING.value,
                filename_or_objs_only=True,
                mode=config["mode"],
            ),
            output_transform=lambda output: adversarial_criterion(
                logits=output[CommonKeys.PRED][f"adversarial_logits"], is_real=True
            ),
        ).attach(engine)

    ProgressBar().attach(engine, output_transform=lambda output: {"Loss": 0})

    engine.run()


def run(
    # File system parameters
    training_subjects: Union[str, Tuple[str, ...]],
    validation_subjects: Union[str, Tuple[str, ...]],
    project_directory: str,
    experiment_name: str,
    mode: str = "training",
    no_augmented_extractions: int = 0,
    # Hardware parameters
    device: int = 0,
    distributed_port: int = TORCH_DISTRIBUTED_DEFAULT_PORT,
    amp: bool = True,
    deterministic: bool = False,
    cuda_benchmark: bool = True,
    seed: int = 4,
    # Training parameters
    epochs: int = 100,
    learning_rate: float = 0.0003,
    gamma: Union[str, float] = 0.99999,
    log_every: int = 1,
    checkpoint_every: int = 1,
    eval_every: int = 5,
    augmentation_probability: float = 0.2,
    augmentation_strength: float = 0,
    # Loss parameters
    loss: str = "jukebox_perceptual",
    adversarial_component: bool = True,
    # Adversarial loss parameters
    finetune_adversarial_component: Tuple[float, float] = None,
    finetune_patience: int = 100,
    discriminator_network: str = "baseline_discriminator",
    discriminator_learning_rate: float = 0.0005,
    discriminator_loss: str = "least_square",
    generator_loss: str = "least_square",
    use_adversarial_adaptive_weight: bool = False,
    adaptive_adversarial_weight_threshold: int = 0,
    adaptive_adversarial_weight_value: float = 1,
    # Baur factor
    initial_factor_value: int = 0,
    initial_factor_steps: int = 25,
    max_factor_steps: int = 50,
    max_factor_value: int = 5,
    # Dataset parameters
    normalize: bool = True,
    roi: Union[
        Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    ] = None,
    batch_size: int = 3,
    patch_size: Tuple[int, int, int] = None,
    eval_batch_size: int = 3,
    eval_patch_size: Tuple[int, int, int] = None,
    training_epoch_length: int = None,
    num_workers: int = 8,
    prefetch_factor: int = 8,
    starting_epoch: int = 0,
    # Network parameters
    network: str = "baseline_vqvae",
    use_subpixel_conv: bool = False,
    use_slim_residual: bool = True,
    no_levels: int = 3,
    downsample_parameters: Tuple[Tuple[int, int, int, int], ...] = (
        (4, 2, 1, 1),
        (4, 2, 1, 1),
        (4, 2, 1, 1),
    ),
    upsample_parameters: Tuple[Tuple[int, int, int, int, int], ...] = (
        (4, 2, 1, 0, 1),
        (4, 2, 1, 0, 1),
        (4, 2, 1, 0, 1),
    ),
    no_res_layers: int = 3,
    no_channels: int = 256,
    codebook_type: str = "ema",
    num_embeddings: Tuple[int, ...] = (256,),
    embedding_dim: Tuple[int, ...] = (256,),
    embedding_init: Tuple[str, ...] = ("normal",),
    commitment_cost: Tuple[float, ...] = (0.25,),
    decay: Tuple[float, ...] = (0.99,),
    decay_warmup: str = None,
    max_decay_epochs: Union[str, int] = 50,
    norm: str = None,
    dropout: float = 0.0,
    act: str = "RELU",
    output_act: str = None,
    evaluation_checkpoint: str = "recent",
    load_nii_canonical: bool = True,
    save_2d_images: tuple = ("axial", "saggital", "coronal"),
    save_3d_images: tuple = None,
):
    """
    Entry point for the vqvae handling. It follows this structure since it is the same one found in the
    Distributed Data Parallelism Ignite tutorial found at :

    https://github.com/pytorch/ignite/tree/master/examples/contrib/cifar10

    Args:
        training_subjects (Union[str, Tuple[str, ...]]): Path(s) towards either a folder with .nii.gz files or towards
            a csv/tsv which has a 'path' column that stores full paths towards .nii.gz or .npy files. The files must be
            .nii.gz for training and extracting mode, and for decoding they must be .npy. A tuple of paths must be
            passed when the selected model is a hierarchical VQVAE. Those will be used for training.

        validation_subjects (Union[str, Tuple[str, ...]]): Path(s) towards either a folder with .nii.gz files or towards
            a csv/tsv which has a 'path' column that stores full paths towards .nii.gz or .npy files. The files must be
            .nii.gz for training and extracting mode, and for decoding they must be .npy. A tuple of paths must be
            passed when the selected model is a hierarchical VQVAE. Those will be used for validation.

        project_directory (str): Path towards folder where the experiment folder will be created.

        experiment_name (str): Name of the experiment which will be used to name a folder in the project_directory.
            Defaults to 'nvidia'.

        mode (str) : It can be one of the following: ['training', 'extracting', 'decoding'].
            'training': Given the location of the .nii.gz images the configured vqvae model will be trained.
            'extracting': Given the location of the .nii.gz images their quantization representation and reconstruction
                will be saved in the output folder. In this mode only the images found from validation_subjects will be
                used.
            'decoding': Given the location of quantization samples in .npy format they will be decoded in the image
                space and if adversarial training was used it will also save the loss output. All will be saved in the
                output folder. In this mode only the images found from validation_subjects will be used.

        no_augmented_extractions (int): The number of augmentations per sample for extracting codes. This is useful
            when the dataset is small and the transformer is overfitting. When it is 0 no augmentations are used during
            extraction. Defaults to 0.

        device (int): The index of the GPU in the PCI_BUS_ID order or 'ddp' for Distributed Data Parallelism.
            Defaults to 0.

        distributed_port (int): Torch distributed backend port. Defaults to 29500.

        amp (bool): Boolean that sets whether we use AMP or not. Defaults to True.

        deterministic (bool): Boolean that sets monai.utils.set_determinism. Defaults to True.

        cuda_benchmark (bool): Boolean that sets whether cuda_benchmark will be used. It is not exclusive with
            deterministic, but it supersedes it. Defaults to False.

        seed (int): The seed to be used for the experiment. Defaults to 2.

        epochs (int): Number of epochs that the network will be trained. Defaults to 1,000,000.

        learning_rate (float): Learning rate of the optimizer. Defaults to 1e-4.

        gamma (Union[str,float]): Gamma that will be used for learning rate decay. Defaults to 'auto' which calculates
            the decay so that at the end of the training the learning rate is equal to 1e-5.

        log_every (int): After how many epochs we save the logs. Defaults to 25.

        checkpoint_every (int): After how many epochs we save a checkpoint. Defaults to 50.

        evaluation_checkpoint (str): Choose which checkpoint to use when extracting/decoding. "recent" uses the most
        recent available, and "best" uses the checkpoint that achieved the best performance according to the key metric.
        Defaults to "recent".

        eval_every (int): After how many epochs do we run evaluation. Defaults to 50.

        augmentation_probability (float): The probabilities of every augmentation. Defaults to 0.2

        augmentation_strength (float): The multiplier of the ADDED augmentation strength. It defaults to 0 for no added
            strength. Augmentations' strength increments are defined in utils.vqvae.AugmentationStrengthScalers.
            Defaults to 0.

        loss (str): What loss to use 'mse' or 'baur'. Defaults to 'baur'.

        adversarial_component (bool): Whether or not we add an adversarial loss component to the 'loss'.
            Defaults to False.

        finetune_adversarial_component (Tuple[float,float]): Discard saved weights and first train the
            discrimator until the value is withing the given range. If it is None the Discriminator weights will be
            loaded, otherwise not. Defaults to None.

        finetune_patience (int): The number of iterations within the range after which the finetuning will be turned
            off. Defaults to 100.

        discriminator_network (str): What discriminator network to use. Defaults to "baseline_discriminator".

        discriminator_learning_rate (float): The learning rate of the discriminator. Defaults to 0.0005.

        discriminator_loss (str): What loss to be used for the discriminator. Defaults to "least_square".

        generator_loss (str): What loss to be used in the adversarial setup for the generator (VQVAE).
            Defaults to "least_square".

        use_adversarial_adaptive_weight (bool): Whether or not to use adaptive weigthing of the adversarial losses based
            on the last layer of the decoder. Defaults to False.

        adaptive_adversarial_weight_threshold (int): The number of epochs after which the adaptive weighting will be
            turned on. Defaults to 0.

        adaptive_adversarial_weight_value (float): The value the adversarial weight before adaptive weighting will be
            turned on. Defaults 1.0.

        initial_factor_value (float): The initial value of loss's factor. Defaults to 0.

        initial_factor_steps (int): For how many epochs we keep the loss's factor to initial_factor_value.
            Defaults to 25.

        max_factor_steps (int): After how many epochs the loss's factor reaches max_factor_value. Defaults to 50

        max_factor_value (float): The maximum value of the loss's factor. Defaults to 5.

        normalize (bool): Whether to normalize the input data in the 0-1 range. Defaults to True.

        roi (Union[Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]): What is the region
            of interest in the image. Will be given to monai.transforms.CenterSpatialCropd if tree ints are passed,
            otherwise it will be given to monai.transforms.SpatialCropd if three tuples (start, end) are given.
            Defaults to None.

        batch_size (int): The batch size that will be used during training. Defaults to 2.

        patch_size (Tuple[int,int,int]): If set the training will be based on specified patch size. Defaults to None.

        eval_batch_size (int): The batch size that will be used during evaluation. Defaults to 1.

        eval_patch_size (Tuple[int,int,int]): If set the evaluation will be based on specified patch size.
            Defaults to None.

        training_epoch_length (int): Number of iterations for one epoch, if set the epochs is set to 1. Its main purpose
            is for debugging and profiling. Defaults to None.

        num_workers (int): The number of threads that will be used to load batches. Defaults to 8.

        prefetch_factor (int): How may batches each thread will try and buffer. Defaults to 6.

        starting_epoch (int): At which epoch we start the training. Defaults to 0.

        network (str): What network to use. Defaults to 'single_vqvae'.

        use_subpixel_conv (bool): Whether or not to use SubPixelConvolution as the last transpose convolution in the
            network. Defaults to True.

        use_slim_residual (bool): Whether or not to have the kernel of the last convolution in each residual unit
            be equal to 1. Default to True

        no_levels (int): How many levels the VQVAE has. Defaults to 3.

        downsample_parameters (Tuple[Tuple[int,int,int,int],...]): A Tuple of Tuples for defining the downsampling
            convolutions. Each Tuple should hold the following information kernel_size (int), stride (int),
            padding (int), dilation(int). Defaults to ((4,2,1,1),(4,2,1,1),(4,2,1,1)).

        upsample_parameters (Tuple[Tuple[int,int,int,int,int],...]): A Tuple of Tuples for defining the upsampling
            convolutions. Each Tuple should hold the following information kernel_size (int), stride (int),
            padding (int), output_padding (int), dilation(int). Defaults to ((4,2,1,0,1),(4,2,1,0,1),(4,2,1,0,1)).

        no_res_layers (int): How many residual layers we use per level. Defaults to 1.

        no_channels (int): How many channels the deepest level has. Defaults to 128.

        codebook_type (ema): What codebook type to use. Defaults to 'ema'.

        num_embeddings (Tuple[int,...]): How many codebook elements (atomic elements/tokens) each of the embedding
            spaces has. Defaults to (32, ).

        embedding_dim (Tuple[int,...]): The channel dimension size of the elements of each codebook.
            Defaults to (64, ).

        embedding_init (Tuple[str,...]): The initialization used for the codebook elements. Can be either 'normal' or
            'kaiming_uniform'. Defaults to ('normal',).

        commitment_cost (Tuple[float,...]): The commitment cost factor that will be used to scale the EMA latent loss.
            Defaults to (0.25, ).

        decay (Tuple[float, ...]): The decay factor that will be used to update the embedding space when EMA
            Quantization update is being used. Defaults to (0.99, ).

        decay_warmup (str): Dictates the behaviour of the deacy warmup. "step" will increase the decay in 4 steps from
            decay to 0.99 in increments equal to 0.25 of the difference. "linear" will linearly increase the decay from
            decay to 0.99 over max_decay_epochs. None will not use any warmup.

        max_decay_epochs (int): After how many epochs the Exponential Moving Average decay goes from decay to 0.99.
            Defaults to 100.

        norm (str): Which normalization technique the network will use. Defaults to None.

        dropout (float): The amount of dropout use in the network. Defaults to 0.1

        act (str): Which activation function the network uses. Defaults to 'RELU'.

        output_act (str): Which activation function should the output be passed through. Defaults to None.

        load_nii_canonical (bool): If true will reorder image array data when loading .nii images to be as closest to
         canonical. Defaults to True.

        save_2d_images (Tuple[str,...]). Log 2D images during training. Options are 'axial', 'sagittal', 'coronal'.
            Defaults to ('axial','sagittal','coronal').

        save_3d_images (Tuple[str,...]). Log 3D images during training. Options are 'axial', 'sagittal', 'coronal'.
            Defaults to None.

    """
    config = locals()

    modes = [m.value for m in VQVAEModes]

    if config["device"] == "ddp":
        deepspeed.init_distributed(
            dist_backend="nccl",
            auto_mpi_discovery=True,
            verbose=False,
            init_method=None,
            distributed_port=config["distributed_port"],
        )

        config["rank"] = int(os.environ["RANK"])
        config["local_rank"] = int(os.environ["LOCAL_RANK"])
        config["world_size"] = int(os.environ["WORLD_SIZE"])
    else:
        config["rank"] = 0
        config["local_rank"] = 0
        config["world_size"] = 1

    if config["mode"] == VQVAEModes.TRAINING.value:
        training(config)
    elif config["mode"] in modes:
        inference(config)
    else:
        raise ValueError(
            f"VQVAE mode unknown. Was give {config['mode']} but choices are {modes}."
        )


if __name__ == "__main__":
    Fire({"run": run})
