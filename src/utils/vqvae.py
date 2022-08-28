import os
from enum import Enum
from copy import deepcopy
from logging import Logger
from math import floor
from typing import Tuple, Union, List, Dict, Callable

import pandas as pd
from ignite.engine import _prepare_batch
from ignite.utils import convert_tensor
from monai.data import Dataset, DataLoader, DistributedSampler
from monai.transforms import (
    Compose,
    AddChanneld,
    ScaleIntensityd,
    RandGaussianNoised,
    RandShiftIntensityd,
    RandAdjustContrastd,
    CenterSpatialCropd,
    SpatialCropd,
    SpatialPadd,
    RandAffined,
    ThresholdIntensityd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    ToTensord,
)
from monai.transforms.io.dictionary import LoadImaged
from monai.utils.enums import NumpyPadMode
from src.transforms.general.dictonary import TraceTransformsd


class VQVAEModes(Enum):
    TRAINING = "training"
    EXTRACTING = "extracting"
    DECODING = "decoding"


class DecayWarmups(Enum):
    STEP = "step"
    LINEAR = "linear"


# This is needed since some augmentation might need sub/super-linear scaling
class AugmentationStrengthScalers(Enum):
    AFFINEROTATE = 0.2
    AFFINETRANSLATE = 1
    AFFINESCALE = 0.01
    ADJUSTCONTRASTGAMMA = 0.01
    SHIFTINTENSITYOFFSET = 0.025
    GAUSSIANNOISESTD = 0.01


def get_data_flow(
    config: dict, logger: Logger = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs the data ingestion logic. There are different approaches for full-image and patch-based training due to
    gpu usage efficiency.

    The following fields are needed in config (in order of appearance):

        training_subjects (Union[str,Tuple[str, ...]]): Either one or multiple absolute paths to either a folder, csv or
            tsv. If it is a csv or tsv, it is expected that a "path" column is present and holds absolute paths to
            individual files. Those subjects will be used for the training dataset. For 'decoding' a Tuple of paths it
            is expected matching that the number of elements in num_embeddings each element should point to either
            folder or csv/tsv are pointing towards .npy files otherwise a single element is expected pointing towards
            .nii.gz files.

        mode (str): For which mode the data flow is being created. It should be one of the following: 'training',
            'extracting',  'decoding'.

        load_nii_canonical (bool): If true will reorder image array data when loading .nii images to be as closest to
         canonical.

        augmentation_probability (float): The probabilities of every augmentation.

        no_augmented_extractions (int):  The number of augmentations per sample for extracting codes. This is useful
            when the dataset is small and the transformer is overfitting. When it is 0 no augmentations are used during
            extraction.

        num_embeddings (Tuple[int,...]): How many atomic elements each quantization elements has. This is used to
            determine the number of quantizations to be loaded.

        normalize (bool): Whether or not the training and validation datasets are 0-1 normalised. Defaults to True.

        roi (Tuple[int,int,int]): The region of interest in the image that will be cropped out and forward modified. If
            None then no cropping will happen. Defaults to None.

        patch_size (Tuple[int,int,int]): How big the randomly cropped area will be for training data. If None no random
            crop will happen. Defaults to None.

        batch_size (int): The batch size that will be used to train the network. Defaults to 2.

        num_workers (int): How many threads will be spawn to load batches. Defaults to 8.

        device (Union[str,int]): The index of the GPU in the PCI_BUS_ID order or 'ddp' for Distributed Data Parallelism.

        prefetch_factor (int): How many batches each thread will try to keep as a buffer. Defaults to 6.

        validation_subjects (Union[str,Tuple[str, ...]]): Either one or multiple absolute paths to either a folder, csv
        or tsv. If it is a csv or tsv, it is expected that a "path" column is present and holds absolute paths to
        individual files. Those subjects will be used for the training dataset. For 'decoding' a Tuple of paths it is
        expected matching that the number of elements in num_embeddings each element should point to either folder or
        csv/tsv are pointing towards .npy files otherwise a single element is expected pointing towards .nii.gz files.

        eval_patch_size (Tuple[int,int,int]): How big the randomly cropped area will be for evaluation data.
        If None no random crop will happen. Defaults to None.

        eval_batch_size (int): The batch size that will be used to evaluate the network. Defaults to 1.

    Args:
        config (dict): Configuration dictionary that holds all the required parameters.

        logger (Logger): Logger that will be used to report DataLoaders parameters.

    Returns:
        DataLoader: Training DataLoader which has data augmentations

        DataLoader: Evaluation DataLoader for the validation data. No data augmentations.

        DataLoader: Evaluation DataLoader for the training data. No data augmentations.
    """

    def get_subjects(
        paths: Union[str, Tuple[str, ...]], mode: str, no_augmented_extractions: int
    ) -> List[Dict[str, str]]:
        if isinstance(paths, str):
            paths = [paths]
        else:
            paths = list(paths)

        files = []
        for path in paths:
            if os.path.isdir(path):
                files.append(
                    [os.path.join(path, os.fsdecode(f)) for f in os.listdir(path)]
                )
            elif os.path.isfile(path):
                if path.endswith(".csv"):
                    files.append(
                        pd.read_csv(filepath_or_buffer=path, sep=",")["path"].to_list()
                    )
                elif path.endswith(".tsv"):
                    files.append(
                        pd.read_csv(filepath_or_buffer=path, sep="\t")["path"].to_list()
                    )
            else:
                raise ValueError(
                    "Path is neither a folder (to source all the files inside) or a csv/tsv with file paths inside."
                )

        subjects = []
        if mode == VQVAEModes.DECODING.value:
            is_multi_level = len(files) > 1
            for file in zip(files) if is_multi_level else files[0]:
                subject = {}
                for idx, quantization in enumerate(file if is_multi_level else [file]):
                    if quantization.endswith(".npy"):
                        subject[f"quantization_{idx}"] = quantization
                    else:
                        raise ValueError(f"Path given is not a .npy file, but {file} ")

                subjects.append(subject)
        else:
            for file in files[0]:
                if file.endswith(".nii.gz"):
                    if (
                        no_augmented_extractions != 0
                        and mode == VQVAEModes.EXTRACTING.value
                    ):
                        for augmentation_id in range(no_augmented_extractions):
                            subjects.append(
                                {"MRI": file, "augmentation_id": int(augmentation_id)}
                            )
                    else:
                        subjects.append({"MRI": file})
                else:
                    raise ValueError(f"Path given is not a .nii.gz file, but {file} ")
        return subjects

    def get_transformations(
        mode: str,
        load_nii_canonical: bool,
        augmentation_probability: float,
        augmentation_strength: float,
        no_augmented_extractions: int,
        num_embeddings: int,
        normalize: bool,
        roi: Union[Tuple[int, ...], Tuple[Tuple[int, int], ...]],
        patch_size: Tuple[int, ...],
    ):
        is_augmented = (
            mode == VQVAEModes.TRAINING.value or no_augmented_extractions != 0
        )

        if config["mode"] == VQVAEModes.DECODING.value:
            keys = [f"quantization_{idx}" for idx in range(len(num_embeddings))]
            transform = [
                LoadImaged(keys=keys, reader="NumpyReader"),
                ToTensord(keys=keys),
            ]
        else:
            transform = [
                LoadImaged(
                    keys=["MRI"],
                    reader="NibabelReader",
                    as_closest_canonical=load_nii_canonical,
                ),
                AddChanneld(keys=["MRI"]),
            ]

            if normalize:
                transform += [ScaleIntensityd(keys=["MRI"], minv=0.0, maxv=1.0)]

            if roi:
                if type(roi[0]) is int:
                    transform += [CenterSpatialCropd(keys=["MRI"], roi_size=roi)]
                elif type(roi[0]) is tuple:
                    transform += [
                        SpatialCropd(
                            keys=["MRI"],
                            roi_start=[a[0] for a in roi],
                            roi_end=[a[1] for a in roi],
                        )
                    ]
                else:
                    raise ValueError(
                        f"roi should be either a Tuple with three ints like (0,1,2) or a Tuple with three Tuples that have "
                        f"two ints like ((0,1),(2,3),(4,5)). But received {roi}."
                    )

                transform += [
                    # This is here to guarantee no sample has lower spatial resolution than the ROI
                    # YOU SHOULD NOT RELY ON THIS TO CATCH YOU SLACK, ALWAYS CHECK THE SPATIAL SIZES
                    # OF YOU DATA PRIOR TO TRAINING ANY MODEL.
                    SpatialPadd(
                        keys=["MRI"],
                        spatial_size=roi
                        if type(roi[0]) is int
                        else [a[1] - a[0] for a in roi],
                        mode=NumpyPadMode.SYMMETRIC,
                    )
                ]

            if patch_size:
                transform += [
                    RandSpatialCropd(
                        keys=["MRI"],
                        roi_size=patch_size,
                        random_size=False,
                        random_center=True,
                    )
                ]

            if is_augmented:
                if patch_size:
                    # Patch based transformations
                    transform += [
                        RandFlipd(
                            keys=["MRI"], prob=augmentation_probability, spatial_axis=0
                        ),
                        RandFlipd(
                            keys=["MRI"], prob=augmentation_probability, spatial_axis=1
                        ),
                        RandFlipd(
                            keys=["MRI"], prob=augmentation_probability, spatial_axis=2
                        ),
                        RandRotate90d(
                            keys=["MRI"],
                            prob=augmentation_probability,
                            spatial_axes=(0, 1),
                        ),
                        RandRotate90d(
                            keys=["MRI"],
                            prob=augmentation_probability,
                            spatial_axes=(1, 2),
                        ),
                        RandRotate90d(
                            keys=["MRI"],
                            prob=augmentation_probability,
                            spatial_axes=(0, 2),
                        ),
                    ]
                else:
                    # Image transformations
                    transform += [
                        RandAffined(
                            keys=["MRI"],
                            prob=augmentation_probability,
                            rotate_range=[
                                0.04
                                + AugmentationStrengthScalers.AFFINEROTATE.value
                                * augmentation_strength
                            ]
                            * 3,
                            # Here int(round()) is used to guarantee an integer of the rounded value of the added
                            #   augmentation strength
                            translate_range=[
                                2
                                + int(
                                    round(
                                        AugmentationStrengthScalers.AFFINETRANSLATE.value
                                        * augmentation_strength
                                    )
                                )
                            ]
                            * 3,
                            scale_range=[
                                0.05
                                + AugmentationStrengthScalers.AFFINESCALE.value
                                * augmentation_strength
                            ]
                            * 3,
                            as_tensor_output=False,
                            padding_mode="zeros",
                            spatial_size=roi
                            if type(roi[0]) is int
                            else [a[1] - a[0] for a in roi],
                            cache_grid=False if roi is None else True,
                        )
                    ]

                # Patch/Image agnostic transformations
                transform += [
                    RandAdjustContrastd(
                        keys=["MRI"],
                        prob=augmentation_probability,
                        gamma=(
                            0.99
                            - AugmentationStrengthScalers.ADJUSTCONTRASTGAMMA.value
                            * augmentation_strength,
                            1.01
                            + AugmentationStrengthScalers.ADJUSTCONTRASTGAMMA.value
                            * augmentation_strength,
                        ),
                    ),
                    RandShiftIntensityd(
                        keys=["MRI"],
                        prob=augmentation_probability,
                        offsets=(
                            0.0,
                            0.05
                            + AugmentationStrengthScalers.SHIFTINTENSITYOFFSET.value
                            * augmentation_strength,
                        ),
                    ),
                    RandGaussianNoised(
                        keys=["MRI"],
                        prob=augmentation_probability,
                        mean=0.0,
                        std=0.02
                        + AugmentationStrengthScalers.GAUSSIANNOISESTD.value
                        * augmentation_strength,
                    ),
                ]

            transform += [
                ThresholdIntensityd(keys=["MRI"], threshold=1, above=False, cval=1.0),
                ThresholdIntensityd(keys=["MRI"], threshold=0, above=True, cval=0),
                ToTensord(keys=["MRI"]),
            ]

            transform = Compose(transform)

            tracer = TraceTransformsd(keys=["MRI"], composed_transforms=[transform])

            transform = [transform, tracer]

        return Compose(transform)

    training_subjects = get_subjects(
        paths=config["training_subjects"],
        mode=config["mode"],
        no_augmented_extractions=0,
    )

    training_transform = get_transformations(
        mode=config["mode"],
        load_nii_canonical=config["load_nii_canonical"],
        augmentation_probability=config["augmentation_probability"],
        augmentation_strength=config["augmentation_strength"],
        no_augmented_extractions=config.get("no_augmented_extractions", 0),
        num_embeddings=config["num_embeddings"],
        normalize=config.get("normalize", True),
        roi=config.get("roi", None),
        patch_size=config.get("patch_size", None),
    )

    training_dataset = Dataset(data=training_subjects, transform=training_transform)

    training_loader = DataLoader(
        training_dataset,
        batch_size=config.get("batch_size", 2),
        num_workers=config.get("num_workers", 8),
        # This is false due to the DistributedSampling handling the shuffling
        shuffle=config["device"] != "ddp",
        drop_last=True,
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 6),
        # Forcefully setting it to false due to this pull request
        # not being in PyTorch 1.7.1
        # https://github.com/pytorch/pytorch/pull/48543
        persistent_workers=False,
        sampler=DistributedSampler(dataset=training_dataset, shuffle=True, even_divisible=True)
        if config["device"] == "ddp"
        else None,
    )

    evaluation_subjects = get_subjects(
        paths=config["validation_subjects"],
        mode=config["mode"],
        no_augmented_extractions=config["no_augmented_extractions"],
    )

    evaluations_transform = get_transformations(
        mode=config["mode"],
        load_nii_canonical=config["load_nii_canonical"],
        augmentation_probability=config["augmentation_probability"],
        augmentation_strength=config["augmentation_strength"],
        no_augmented_extractions=config.get("no_augmented_extractions", 0),
        num_embeddings=config["num_embeddings"],
        normalize=config.get("normalize", True),
        roi=config.get("roi", None),
        patch_size=config.get("patch_size", None),
    )

    evaluation_dataset = Dataset(
        data=evaluation_subjects, transform=evaluations_transform
    )

    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=config.get("eval_batch_size", 1),
        num_workers=config.get("num_workers", 8),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 6),
        persistent_workers=False,
        sampler=DistributedSampler(
            dataset=evaluation_dataset, shuffle=False, even_divisible=False
        )
        if config["device"] == "ddp"
        else None,
    )

    training_evaluation_dataset = Dataset(
        data=training_subjects, transform=evaluations_transform
    )

    training_evaluation_loader = DataLoader(
        training_evaluation_dataset,
        batch_size=config.get("eval_batch_size", 1),
        num_workers=config.get("num_workers", 8),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 6),
        persistent_workers=False,
        sampler=DistributedSampler(
            dataset=evaluation_dataset, shuffle=False, even_divisible=False
        )
        if config["device"] == "ddp"
        else None,
    )

    if logger:
        logger.info("Dataflow setting:")
        logger.info("\tTraining:")
        if config.get("patch_size", None):
            logger.info(f"\t\tPatch Size: {config['patch_size']}")
        logger.info(f"\t\tLength: {len(training_loader)}")
        logger.info(f"\t\tBatch Size: {training_loader.batch_size}")
        logger.info(f"\t\tPin Memory: {training_loader.pin_memory}")
        logger.info(f"\t\tNumber of Workers: {training_loader.num_workers}")
        logger.info(f"\t\tPrefetch Factor: {training_loader.prefetch_factor}")
        logger.info("\tValidation:")
        if config.get("eval_patch_size", None):
            logger.info(f"\t\tPatch Size: {config['eval_patch_size']}")
        logger.info(f"\t\tLength: {len(evaluation_loader)}")
        logger.info(f"\t\tBatch Size: {evaluation_loader.batch_size}")
        logger.info(f"\t\tPin Memory: {evaluation_loader.pin_memory}")
        logger.info(f"\t\tNumber of Workers: {evaluation_loader.num_workers}")
        logger.info(f"\t\tPrefetch Factor: {evaluation_loader.prefetch_factor}")

    config["epoch_length"] = len(training_loader)

    if config["mode"] != VQVAEModes.DECODING.value:
        _, _, input_height, input_width, input_depth = next(iter(training_loader))[
            "MRI"
        ].shape
        config["input_shape"] = (input_height, input_width, input_depth)

    return training_loader, evaluation_loader, training_evaluation_loader


def get_ms_ssim_window(config: dict, logger: Logger = None) -> int:
    """
    Calculates the window size of the gaussian kernel for the MS-SSIM if the smallest dimension of the image is
    lower than 160 (requirement of the default parameters of MS-SSIM).

    It will first look for the 'eval_patch_size' since it has the last one applied, if not found it will look for 'roi'
    since all images are bing cropped or padded to that roi, and lastly it will look for 'input_shape'.

    Args:
        config (dict): Dictionary that holds the whole configuration
        logger (Logger): Logger for printing the logic

    Returns:
        int: Half of the maximum kernel size allowed or next odd int
    """
    if config["eval_patch_size"]:
        min_ps = min(config["eval_patch_size"])
    elif config["roi"]:
        if type(config["roi"][0]) is int:
            min_ps = min(config["roi"])
        elif type(config["roi"][0]) is tuple:
            min_ps = min([a[1] - a[0] for a in config["roi"]])
    else:
        min_ps = min(config["input_shape"])

    if min_ps > 160:
        win_size = 11
    else:
        win_size = floor(((min_ps / 2 ** 4) + 1) / 2)

        if win_size <= 1:
            raise ValueError(
                "Window size for MS-SSIM can't be calculated. Please increase patch_size's smallest dimension."
            )

        # Window size must be odd
        if win_size % 2 == 0:
            win_size += 1

    if logger:
        logger.info("MS-SSIM window calculation:")
        if config["eval_patch_size"]:
            logger.info(f"\tMinimum spatial dimension: {min_ps}")
        logger.info(f"\tWindow size {win_size}")

    return win_size


def prepare_batch(batch, device=None, non_blocking=False):
    """
    Prepare batch function that allows us to train an unsupervised mode by using the SupervisedTrainer
    """
    x_input = x_target = batch["MRI"]
    return _prepare_batch((x_input, x_target), device, non_blocking)


def prepare_decoding_batch(
    batch, num_quantization_levels, device=None, non_blocking=False
):
    """
    Prepare batch function that allows us to train an unsupervised mode by using the SupervisedTrainer
    """
    x_input = x_target = [
        convert_tensor(
            batch[f"quantization_{i}"].long(), device=device, non_blocking=non_blocking
        )
        for i in range(num_quantization_levels)
    ]
    return x_input, x_target


def get_batch_transform(
    mode: str,
    no_augmented_extractions: int,
    is_nii_based: bool,
    filename_or_objs_only: bool,
) -> Callable:
    """
    Batch transform generation, it handles the generation of the function for all modes. It also takes care of
    prepending the augmentation index to the filename.

    Args:
        mode (str): The running mode of the entry point. It can be either 'extracting' or 'decoding'.
        no_augmented_extractions (int): The number of augmentations per sample for extracting codes.
        is_nii_based (bool): Whether or not the batch data is based on nii input.
        filename_or_objs_only (bool): Whether or not we pass only the filename from the metadata.
    Returns:
        Batch transformations function
    """

    def batch_transform(batch: Dict) -> Dict:
        key = "quantization_0" if mode == VQVAEModes.DECODING.value else "MRI"

        if filename_or_objs_only:
            output_dict = {
                "filename_or_obj": deepcopy(
                    batch[f"{key}_meta_dict"]["filename_or_obj"]
                )
            }
        else:
            output_dict = deepcopy(batch[f"{key}_meta_dict"])

        if no_augmented_extractions > 0:
            file_extension = ".nii.gz" if is_nii_based else ".npy"
            output_dict["filename_or_obj"] = [
                f.replace(f"{file_extension}", f"_{i}{file_extension}")
                for f, i in zip(
                    output_dict["filename_or_obj"], batch["augmentation_id"]
                )
            ]

        return output_dict

    return batch_transform
