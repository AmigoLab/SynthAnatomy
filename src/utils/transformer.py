import os
from enum import Enum
from logging import Logger
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ignite.utils import convert_tensor
from monai.data import Dataset, DataLoader, DistributedSampler
from monai.transforms import ToTensord, Compose
from monai.transforms.io.dictionary import LoadImaged


class TransformerModes(Enum):
    TRAINING = "training"
    INFERENCE = "inference"


class TransformerConditioningType(Enum):
    NONE = "none"
    BOSREPLACEMENT = "bos_replacement"
    PREPENDING = "prepending"


def get_data_flow(config: dict, logger: Logger = None) -> Tuple[DataLoader, DataLoader]:
    """
        Constructs the data ingestion logic. The quantization element will be loaded at the key "quantization".

        The following fields are needed in config:
            training_subjects (str): Absolute path to either a folder, csv or tsv. If it is a folder all .nii.gz files
            will be ingested. If it is a csv or tsv, it is expected that a "path" column is present and holds
            absolute paths to individual .nii.gz files. Those subjects will be used for the training dataset.

            validation_subjects (str): Absolute path to either a folder, csv or tsv. If it is a folder all .nii.gz files
            will be ingested. If it is a csv or tsv, it is expected that a "path" column is present and holds
            absolute paths to individual .nii.gz files. Those subjects will be used for the validation dataset.

            batch_size (int): The batch size that will be used to train the network. Defaults to 2.

            eval_batch_size (int): The batch size that will be used to evaluate the network. Defaults to 2.

            num_workers (int): How many threads will be spawn to load batches. Defaults to 8.

            prefetch_factor (int): How many batches each thread will try to keep as a buffer. Defaults to 6.

            conditioning_path (str): Path towards a csv/tsv file that has a 'subject' column in which the file names
            from both training and validation subjects are and the other columns hold conditioning information

            conditionings (Tuple[str,...]): The conditionings from the conditioning_path files that will be prepended to
            the transformer input. The elements of the Tuple must be column names from the file.

            vocab_size (int): The size of the vocabulary. It must be the same values as the "num_embeddings" argument
            used during the vqvae training.

        Args:
            config (dict): Configuration dictionary that holds all the required parameters.

            logger (Logger): Logger that will be used to report DataLoaders parameters.

        Returns:
            DataLoader: Training DataLoader for the training data

            DataLoader: Evaluation DataLoader for the validation data.
    """

    def get_subjects(
        subjects_file_path: str, conditioning_path: str, conditionings: Tuple[str]
    ) -> Dict[str, str]:
        if os.path.isdir(subjects_file_path):
            subjects_files = [
                os.path.join(subjects_file_path, os.fsdecode(f))
                for f in os.listdir(subjects_file_path)
            ]
        elif os.path.isfile(subjects_file_path):
            if subjects_file_path.endswith(".csv"):
                subjects_files = pd.read_csv(
                    filepath_or_buffer=subjects_file_path, sep=","
                )["path"].to_list()
            elif subjects_file_path.endswith(".tsv"):
                subjects_files = pd.read_csv(
                    filepath_or_buffer=subjects_file_path, sep="\t"
                )["path"].to_list()
        else:
            raise ValueError(
                "Path is neither a folder (to source all the files inside) or a csv/tsv with file paths inside."
            )

        offsets = None
        if conditioning_path:
            if os.path.isfile(conditioning_path):
                if conditioning_path.endswith(".csv"):
                    conditioning_file = pd.read_csv(
                        filepath_or_buffer=conditioning_path, sep=","
                    )
                elif conditioning_path.endswith(".tsv"):
                    conditioning_file = pd.read_csv(
                        filepath_or_buffer=conditioning_path, sep="\t"
                    )
            else:
                raise ValueError("Path is not a csv/tsv with file paths inside.")

            offsets = conditioning_file[[c for c in conditionings]].nunique().to_dict()

        subjects = []
        nan_subjects = 0
        mia_subjects = 0

        for file in subjects_files:
            valid_subject = True
            subject_name = os.path.basename(file)
            subject = {"quantization": file}
            if conditioning_path:
                for conditioning in conditionings:
                    try:
                        conditioning_value = conditioning_file.loc[
                            conditioning_file["subject"] == subject_name, conditioning
                        ].values[0]
                    except IndexError:
                        mia_subjects += 1
                        valid_subject = False
                        break

                    if np.isnan(conditioning_value):
                        nan_subjects += 1
                        valid_subject = False
                        break

                    subject[conditioning] = conditioning_value

            if valid_subject:
                subjects.append(subject)

        if mia_subjects > 0 or nan_subjects > 0:
            logger.warning(
                f"{mia_subjects + nan_subjects} were discarded during data loading. "
                f"{mia_subjects} did not have matching conditioning and {nan_subjects} had conditioning that was NaN. "
                f"Make sure your conditioning data covers all of your subjects."
            )
        return subjects, offsets

    training_data, ofs = get_subjects(
        subjects_file_path=config["training_subjects"],
        conditioning_path=config["conditioning_path"],
        conditionings=config["conditionings"],
    )

    training_dataset = Dataset(
        data=training_data,
        transform=Compose(
            [LoadImaged(keys=["quantization"]), ToTensord(keys=["quantization"])]
        ),
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=config.get("batch_size", 2),
        num_workers=config.get("num_workers", 8),
        shuffle=config["device"] != "ddp",
        drop_last=True,
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 6),
        # Forcefully setting it to false due to this pull request
        # not being in PyTorch 1.7.1
        # https://github.com/pytorch/pytorch/pull/48543
        persistent_workers=False,
        sampler=DistributedSampler(
            dataset=training_dataset,
            shuffle=True,
            even_divisible=True,
        )
        if config["device"] == "ddp"
        else None,
    )

    evaluation_data, _ = get_subjects(
        subjects_file_path=config["validation_subjects"],
        conditioning_path=config["conditioning_path"],
        conditionings=config["conditionings"],
    )

    evaluation_dataset = Dataset(
        data=evaluation_data,
        transform=Compose(
            [LoadImaged(keys=["quantization"]), ToTensord(keys=["quantization"])]
        ),
    )

    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=config.get("eval_batch_size", 2),
        num_workers=config.get("num_workers", 8),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 6),
        persistent_workers=False,
        sampler=DistributedSampler(
            dataset=evaluation_dataset,
            shuffle=False,
            even_divisible=False,
        )
        if config["device"] == "ddp"
        else None,
    )

    if logger:
        logger.info("Dataflow setting:")
        logger.info("\tTraining:")
        logger.info(f"\t\tLength: {len(training_loader)}")
        logger.info(f"\t\tBatch Size: {training_loader.batch_size}")
        logger.info(f"\t\tPin Memory: {training_loader.pin_memory}")
        logger.info(f"\t\tNumber of Workers: {training_loader.num_workers}")
        logger.info(f"\t\tPrefetch Factor: {training_loader.prefetch_factor}")
        logger.info("\tValidation:")
        logger.info(f"\t\tLength: {len(evaluation_loader)}")
        logger.info(f"\t\tBatch Size: {evaluation_loader.batch_size}")
        logger.info(f"\t\tPin Memory: {evaluation_loader.pin_memory}")
        logger.info(f"\t\tNumber of Workers: {evaluation_loader.num_workers}")
        logger.info(f"\t\tPrefetch Factor: {evaluation_loader.prefetch_factor}")

    config["epoch_length"] = len(training_loader)

    if ofs:
        if logger:
            logger.info("The conditioning vocab size is modified as follows:")
        conditioning_num_tokens = []

        for c, o in ofs.items():
            conditioning_num_tokens.append(o)
            if logger:
                logger.info(f"\tTo {conditioning_num_tokens} due to {c}.")

        config["conditioning_num_tokens"] = conditioning_num_tokens
    return training_loader, evaluation_loader


def prepare_batch(
    batch,
    index_sequence,
    vocab_size,
    conditionings=None,
    device=None,
    non_blocking=False,
):
    """
    Batch preparation logic of the quantization elements for training.

    We first rasterize it through the reshape method followed by a reordering based on the index_sequence which is
    can be arbitrarily generated.

    After rasterizing a padding to the left with the vocab_size int is done since the quantization elements are
    actually in [0, vocab_size) natural numbers range.

    Then the processed encoding is split into input, which is everything but the last element and target which is
    everything but the first element which is added by padding.
    """
    encoded = batch["quantization"]
    encoded = encoded.reshape(encoded.shape[0], -1)
    encoded = encoded[:, index_sequence]
    encoded = F.pad(encoded, (1, 0), "constant", vocab_size)
    encoded = encoded.long()

    conditioned = None
    if conditionings:
        conditioned = []

        for conditioning_label in conditionings:
            conditioning = batch[conditioning_label]

            if len(conditioning.shape) == 1:
                conditioning = conditioning[..., None]

            conditioning = conditioning.long()
            conditioning = convert_tensor(conditioning, device, non_blocking)
            conditioned.append(conditioning)

    x_input = convert_tensor(encoded[:, :-1], device, non_blocking)
    x_target = convert_tensor(encoded[:, 1:], device, non_blocking)

    return (x_input, conditioned), x_target


def prepare_inference_batch(
    batch, num_embeddings, conditionings=None, device=None, non_blocking=False
):
    """
    Batch preparation logic of the quantization elements for inference.

    Given loaded quantization the batch size is determined and no_samples of single value tensor are being generated
    where the value is the num_embedding since this was used as start of sentence token during the training.
    """
    no_samples = batch["quantization"].shape[0]
    start_pixel = np.array([[num_embeddings]])
    start_pixel = np.repeat(start_pixel, no_samples, axis=0)
    initial = torch.from_numpy(start_pixel)
    initial = initial.long()

    conditioned = None
    if conditionings:
        conditioned = []

        for conditioning_label in conditionings:
            conditioning = batch[conditioning_label]

            if len(conditioning.shape) == 1:
                conditioning = conditioning[..., None]

            conditioning = conditioning.long()
            conditioning = convert_tensor(conditioning, device, non_blocking)
            conditioned.append(conditioning)

    x_input = convert_tensor(initial, device, non_blocking)
    x_target = convert_tensor(initial, device, non_blocking)

    return (x_input, conditioned), x_target
