#!/usr/bin/env python3
import os
from typing import Union, Tuple

import numpy as np
import torch
import torch.distributed.distributed_c10d as dist
import deepspeed
from fire import Fire
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import GpuInfo
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.engines.utils import CommonKeys
from monai.handlers import CheckpointSaver, LrScheduleHandler, CheckpointLoader
from torch.utils.tensorboard import SummaryWriter

from src.handlers.general import (
    EvaluationHandler,
    MaxEpochsHandler,
    LoggingPreparationHandler,
    TensorBoardHandler,
    LossSummaryHandler,
    NpySaver,
)
from src.inferer.transformer import (
    TransformerTrainingInferer,
    TransformerInferenceInferer,
)
from src.losses.transformer.transformer import CELoss
from src.metrics.transformer import CE
from src.networks.transformers.img2seq_ordering import (
    Ordering,
    OrderingType,
    OrderingTransformations,
)
from src.networks.transformers.performer import Performer
from src.utils.general import get_gamma, basic_initialization, log_network_size
from src.utils.transformer import (
    get_data_flow,
    TransformerModes,
    TransformerConditioningType,
    prepare_batch,
    prepare_inference_batch,
)


def training(config: dict):
    logger, device = basic_initialization(
        config=config, logger_name="Transformer-Training"
    )

    training_loader, evaluation_loader = get_data_flow(config=config, logger=logger)

    _, input_height, input_width, input_depth = next(iter(training_loader))[
        "quantization"
    ].shape

    ordering = Ordering(
        ordering_type=config["ordering_type"],
        spatial_dims=3,
        dimensions=(1, input_height, input_width, input_depth),
        reflected_spatial_dims=config["reflected_spatial_dims"],
        transpositions_axes=config["transpositions_axes"],
        rot90_axes=config["rot90_axes"],
        transformation_order=config["transformation_order"],
    )

    network = Performer(
        causal=True,
        ordering=ordering,
        num_tokens=config["vocab_size"] + 1,
        # This will be internally modified to account for the added conditioning if conditioning type is
        # prepending, otherwise it will remain unmodified. The +1 is to account for the begging of sentence token
        max_seq_len=input_height * input_width * input_depth + 1,
        dim=config["n_embd"],
        depth=config["n_layers"],
        heads=config["n_head"],
        local_attn_heads=config["local_attn_heads"],
        local_window_size=config["local_window_size"],
        feature_redraw_interval=config["feature_redraw_interval"],
        generalized_attention=config["generalized_attention"],
        emb_dropout=config["emb_dropout"],
        ff_dropout=config["ff_dropout"],
        attn_dropout=config["attn_dropout"],
        use_rezero=config["use_rezero"],
        rotary_position_emb=False,
        fixed_position_emb=config["position_emb"] == "fixed",
        axial_position_emb=False,
        spatial_position_emb=config["spatial_position_emb"],
        spatial_shape=(input_height, input_width, input_depth),
        conditioning_num_tokens=config["conditioning_num_tokens"]
        if config["conditionings"]
        else 0,
        conditioning_type=config["conditioning_type"]
    ).to(device)
    log_network_size(network=network, logger=logger)

    if config["device"] == "ddp":
        network = torch.nn.parallel.DistributedDataParallel(
            network,
            device_ids=[config["local_rank"]],
            broadcast_buffers=True,
            find_unused_parameters=True,
            bucket_cap_mb=12.5,
        )

    loss_function = CELoss().to(device)

    optimizer = torch.optim.Adam(network.parameters(), config["learning_rate"])

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=(
            config["gamma"]
            if isinstance(config["gamma"], float)
            else get_gamma(config=config, logger=logger)
        ),
    )

    train_handlers = [LoggingPreparationHandler()] if config["rank"] == 0 else []

    train_handlers += [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=False, epoch_level=False)
    ]

    if config["rank"] == 0:
        train_handlers += [
            LossSummaryHandler(loss=loss_function),
            TensorBoardHandler(
                summary_writer=SummaryWriter(config["logs_directory"] + "train/"),
                interval=config["log_every"],
                epoch_level=True,
            ),
        ]

    key_metric = {
        f"Metric-CE-Prediction": CE(
            output_transform=lambda network_output: (
                network_output[CommonKeys.PRED],
                network_output[CommonKeys.LABEL],
            )
        )
    }

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=config["epochs"],
        non_blocking=True,
        train_data_loader=training_loader,
        network=network,
        optimizer=optimizer,
        inferer=TransformerTrainingInferer(),
        loss_function=loss_function,
        prepare_batch=lambda batch, pb_device, non_blocking: prepare_batch(
            batch,
            network.module.ordering.get_sequence_ordering()
            if config["device"] == "ddp"
            else network.ordering.get_sequence_ordering(),
            config["vocab_size"],
            config["conditionings"],
            pb_device,
            non_blocking,
        ),
        train_handlers=train_handlers,
        amp=False,
    )

    validation_evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=evaluation_loader,
        non_blocking=True,
        network=network,
        prepare_batch=lambda batch, pb_device, non_blocking: prepare_batch(
            batch,
            network.module.ordering.get_sequence_ordering()
            if config["device"] == "ddp"
            else network.ordering.get_sequence_ordering(),
            config["vocab_size"],
            config["conditionings"],
            pb_device,
            non_blocking,
        ),
        key_val_metric=key_metric,
        inferer=TransformerTrainingInferer(),
        amp=False,
        val_handlers=[
            LoggingPreparationHandler(),
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
        else [],
    )

    training_evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=training_loader,
        non_blocking=True,
        network=network,
        prepare_batch=lambda batch, pb_device, non_blocking: prepare_batch(
            batch,
            network.module.ordering.get_sequence_ordering()
            if config["device"] == "ddp"
            else network.ordering.get_sequence_ordering(),
            config["vocab_size"],
            config["conditionings"],
            pb_device,
            non_blocking,
        ),
        key_val_metric=key_metric,
        inferer=TransformerTrainingInferer(),
        amp=False,
        val_handlers=[
            LoggingPreparationHandler(),
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
        else [],
    )

    to_save = {
        "network": network,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "trainer": trainer,
    }

    if config.get("checkpoint_fp", None):
        # The warning is due to bad type hinting from MONAI. Internally the map_location is passed as follows
        #   checkpoint = torch.load(self.load_path, map_location=self.map_location)
        #   torch.load allows map_location to be a torch.device, so the following code is valid.
        CheckpointLoader(
            load_path=config["checkpoint_fp"], load_dict=to_save, map_location=device
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
        key_metric_name=validation_evaluator.state.key_metric_name,
        key_metric_n_saved=1,
    ).attach(validation_evaluator)

    MaxEpochsHandler(max_epochs=config["epochs"]).attach(trainer)

    EvaluationHandler(
        evaluation_engine=validation_evaluator, evaluate_every=config["eval_every"]
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


def inference(config: dict):
    logger, device = basic_initialization(
        config=config, logger_name="Transformer-Training"
    )

    training_loader, evaluation_loader = get_data_flow(config=config, logger=logger)

    _, input_height, input_width, input_depth = next(iter(training_loader))[
        "quantization"
    ].shape

    ordering = Ordering(
        ordering_type=config["ordering_type"],
        spatial_dims=3,
        dimensions=(1, input_height, input_width, input_depth),
        reflected_spatial_dims=config["reflected_spatial_dims"],
        transpositions_axes=config["transpositions_axes"],
        rot90_axes=config["rot90_axes"],
        transformation_order=config["transformation_order"],
    )

    network = Performer(
        causal=True,
        ordering=ordering,
        num_tokens=config["vocab_size"] + 1,
        # This will be internally modified to account for the added conditioning if conditioning type is
        # prepending, otherwise it will remain unmodified. The +1 is to account for the begging of sentence token
        max_seq_len=input_height * input_width * input_depth + 1,
        dim=config["n_embd"],
        depth=config["n_layers"],
        heads=config["n_head"],
        local_attn_heads=config["local_attn_heads"],
        local_window_size=config["local_window_size"],
        feature_redraw_interval=config["feature_redraw_interval"],
        generalized_attention=config["generalized_attention"],
        emb_dropout=config["emb_dropout"],
        ff_dropout=config["ff_dropout"],
        attn_dropout=config["attn_dropout"],
        use_rezero=config["use_rezero"],
        rotary_position_emb=False,
        fixed_position_emb=config["position_emb"] == "fixed",
        axial_position_emb=False,
        spatial_position_emb=config["spatial_position_emb"],
        spatial_shape=(input_height, input_width, input_depth),
        conditioning_num_tokens=config["conditioning_num_tokens"]
        if config["conditionings"]
        else 0,
        conditioning_type=config["conditioning_type"]
    ).to(device)

    log_network_size(network=network, logger=logger)

    if config["device"] == "ddp":
        network = torch.nn.parallel.DistributedDataParallel(
            network,
            device_ids=[config["local_rank"]],
            broadcast_buffers=True,
            find_unused_parameters=True,
            bucket_cap_mb=12.5,
        )

    engine = SupervisedEvaluator(
        device=device,
        val_data_loader=evaluation_loader,
        inferer=TransformerInferenceInferer(
            sample=config["sample"],
            temperature=config["temperature"],
            top_k=config["top_k"],
        ),
        non_blocking=True,
        network=network,
        prepare_batch=lambda batch, pb_device, non_blocking: prepare_inference_batch(
            batch,
            config["vocab_size"],
            config["conditionings"],
            pb_device,
            non_blocking,
        ),
        amp=False,
        val_handlers=[],
    )

    if config.get("checkpoint_fp", None):
        # The warning is due to bad type hinting from MONAI. Internally the map_location is passed as follows
        #   checkpoint = torch.load(self.load_path, map_location=self.map_location)
        #   torch.load allows map_location to be a torch.device, so the following code is valid.
        CheckpointLoader(
            load_path=config["checkpoint_fp"],
            load_dict={"network": network},
            map_location=device,
        ).attach(engine)

    # TODO: Improve functionality by decorelating the inference from pre-existing encodings, needs data loading to use
    #   a CSV reader to load the conditionings and then use the subject column as file name.
    NpySaver(
        output_dir=config["outputs_directory"],
        output_postfix="sample",
        dtype=np.dtype(np.uint16),
        batch_transform=lambda batch: {
            "filename_or_obj": batch["quantization_meta_dict"]["filename_or_obj"]
        },
        output_transform=lambda output: output[CommonKeys.PRED],
    ).attach(engine)

    ProgressBar().attach(engine, output_transform=lambda output: {"Loss": 0})

    engine.run()


def run(
    # File system parameters
    training_subjects: str,
    validation_subjects: str,
    project_directory: str,
    experiment_name: str,
    mode: str = TransformerModes.TRAINING.value,
    conditioning_path: str = None,
    conditionings: Tuple[str, ...] = None,
    conditioning_type: str = TransformerConditioningType.BOSREPLACEMENT.value,
    # Hardware parameters
    device: int = 0,
    deterministic: bool = False,
    cuda_benchmark: bool = True,
    seed: int = 2,
    # Training parameters
    epochs: int = 1000000,
    learning_rate: float = 1e-4,
    gamma: Union[str, float] = "auto",
    log_every: int = 25,
    checkpoint_every: int = 50,
    eval_every: int = 50,
    # Inference parameters
    sample: bool = True,
    temperature: float = 1.0,
    top_k: int = None,
    # Dataset parameters
    batch_size: int = 2,
    eval_batch_size: int = 2,
    num_workers: int = 8,
    prefetch_factor: int = 6,
    starting_epoch: int = 0,
    # Sequence Ordering parameters:
    ordering_type: str = OrderingType.RASTER_SCAN.value,
    reflected_spatial_dims: Union[Tuple[bool, bool], Tuple[bool, bool, bool]] = (
        False,
        False,
        False,
    ),
    transpositions_axes: Union[
        Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int], ...]
    ] = tuple(),
    rot90_axes: Union[
        Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int], ...]
    ] = tuple(),
    transformation_order: Tuple[
        OrderingTransformations, OrderingTransformations, OrderingTransformations
    ] = (
        OrderingTransformations.TRANSPOSE.value,
        OrderingTransformations.ROTATE_90.value,
        OrderingTransformations.REFLECT.value,
    ),
    # Network parameters
    network: str = "performer",
    vocab_size: int = 32,
    n_embd: int = 256,
    n_layers: int = 10,
    n_head: int = 8,
    local_attn_heads: int = 0,
    local_window_size: int = 256,
    feature_redraw_interval: int = 1000,
    generalized_attention: bool = False,
    emb_dropout: float = 0.0,
    ff_dropout: float = 0.0,
    attn_dropout: float = 0.0,
    use_rezero: bool = False,
    position_emb: str = "absolute",
    spatial_position_emb: str = None,
    evaluation_checkpoint: str = "recent",
):
    f"""
    Entry point for the transformer handling. It follows this structure since it is the same one found in the
    Distributed Data Parallelism Ignite tutorial found at :

    https://github.com/pytorch/ignite/tree/master/examples/contrib/cifar10

    Args:
        training_subjects (str): Path towards either a folder with .npy files or towards a csv/tsv which has a 'path'
            column that stores full paths towards .npy files. Those will be used for training.

        validation_subjects (str): Path towards either a folder with .npy files or towards a csv/tsv which has a 'path'
            column that stores full paths towards .npy files. Those will be used for validation.

        conditioning_path (str): Path towards a csv/tsv file that has a 'subject' column in which the file names from
            both training and validation subjects are and the other columns hold conditioning information

        conditionings (Tuple[str,...]): The conditionings from the conditioning_path files that will be prepended to the
            transformer input. The elements of the Tuple must be column names from the file.
    
        conditioning_type (str): The style of conditioning that will be used in the transformer. It can be 'none' for 
            no conditioning, 'bos_replacement' where the beginning of sentence token is replaced by the summation of the 
            conditionings' embeddings and 'prepending' where the conditionings' embeddings are prepended before the 
            beginning of sentence token.
    
        project_directory (str): Path towards folder where the experiment folder will be created.

        experiment_name (str): Name of the experiment which will be used to name a folder in the project_directory.
            Defaults to 'nvidia'.

        mode (str) : It can be one of the following: ['training', 'inference'].
            'training': Given the location of the .npy quantization representations the configured transformer will be
                trained.
            'inference': In this mode random samples will be generated. The number of samples will be equal to the size
                of the validation dataset.

        device (int): The index of the GPU in the PCI_BUS_ID order. Defaults to 0.

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
        
        evaluation_checkpoint (str): Choose which checkpoint to use when performing inference. "recent" uses the most 
        recent available, and "best" uses the checkpoint that achieved the best performance according to the key metric.
        Defaults to "recent".
        
        eval_every (int): After how many epochs do we run evaluation. Defaults to 50.

        sample (bool): Whether the values are sampled from the distribution or take the most likely. Defaults to True.

        temperature (float): Temperature value to scale the logits by 1/temperature. Defaults to 1.0.

        top_k (int): Crop probabilities to only the top k options. If None, no cropping happens. Defaults to None.

        batch_size (int): The batch size that will be used during training. Defaults to 2.

        eval_batch_size (int): The batch size that will be used during evaluation. Defaults to 2.

        num_workers (int): The number of threads that will be used to load batches. Defaults to 8.

        prefetch_factor (int): How may batches each thread will try and buffer. Defaults to 6.

        starting_epoch (int): At which epoch we start the training. Defaults to 0.

        network (str): What network to use. Defaults to 'performer'.
        
        vocab_size (int): The size of the vocabulary. It must be the same values as the "num_embeddings" argument used
            during the vqvae training. Defaults to 32.

        ordering_type (str): The ordering logic that will be applied to project from 2D/3D tensor to 1D tensor. It can 
            be one of the following: {[e.value for e in OrderingType]}. Defaults to 'raster_scan'.
        
        reflected_spatial_dims (Union[Tuple[bool, bool], Tuple[bool, bool, bool]]): Weather or not to flip axes of the 
            2D/3D tensor before being projected to a 1D tensor. Defaults to (False, False, False).
        
        transpositions_axes (Union[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int], ...]]): Around which axes to 
            apply np.transpose. Defaults to (). 
        
        rot90_axes (Union[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int], ...]]): Around which axes to apply 
            np.rot90. Defaults to ().
        
        transformation_order (Tuple[OrderingTransformations, OrderingTransformations, OrderingTransformations]): In
            which order the np.transpose, np.rot90, np.reflect are being called. It should contain at least one and at 
            most all the following: {[e.value for e in OrderingTransformations]}. Defaults to 
            ("transpose", "rotate90", "reflect").
        
        n_embd (int): The size of the latent representation that the transformer will use. Defaults to 256.

        n_layers (int): The number of layers the transformer will have. Defaults to 10.

        n_head (int): The number of heads that the self attention mechanism will use.
        
        local_attn_heads (int): How many of the n_head will be local attention heads instead of global attention ones.
            Defaults to 0.
            
        local_window_size (int): The number of tokens the local attention heads will look at. Defaults to 256.
        
        feature_redraw_interval (int): How frequently to redraw the projection matrix, the more frequent, the slower 
            the training. Defaults to 1000.
        
        generalized_attention (bool): Whether or not to use generalized attention or the softmax approximation.
            Defaults to False.
        
        emb_dropout (float): Drop probability for the Dropout layer just after the embedding layer.

        ff_dropout (float): Drop probability for the Dropout layer just after the linear layers.

        attn_dropout (float): Drop probability for the Dropout layer just after the attention mechanism.
        
        use_rezero (bool): Whether or not to use Rezero logic for improved convergence. Defaults to False.
        
        position_emb (str): It can be either None for no spatial positioning or 'fixed' or 'absolute'.
                
        spatial_position_emb (str): It can be either None for no spatial positioning or 'fixed' or 'absolute'.
            Defaults to None.
    """
    config = locals()

    modes = [m.value for m in TransformerModes]

    if config["device"] == "ddp":
        deepspeed.init_distributed(
            dist_backend="nccl",
            auto_mpi_discovery=True,
            verbose=False,
            init_method=None,
        )

        config["rank"] = int(os.environ["RANK"])
        config["local_rank"] = int(os.environ["LOCAL_RANK"])
        config["world_size"] = int(os.environ["WORLD_SIZE"])
    else:
        config["rank"] = 0
        config["local_rank"] = 0
        config["world_size"] = 1

    if config["mode"] == TransformerModes.TRAINING.value:
        training(config=config)
    elif config["mode"] == TransformerModes.INFERENCE.value:
        inference(config=config)
    else:
        raise ValueError(
            f"Transformer mode unknown. Was given {config['mode']} but choices are {modes}."
        )


if __name__ == "__main__":
    Fire({"run": run})
