import collections.abc
import logging
from bisect import bisect_right
from copy import deepcopy
from enum import Enum
from typing import Callable, Union, Dict, Optional, List, Tuple

import numpy as np
import torch
from ignite.engine import Engine, Events
from monai.engines import Trainer
from monai.data.utils import create_file_basename
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module
from torch.nn.modules.loss import _Loss


class ParamSchedulerHandler:
    """
        General purpose scheduler for parameters values. By default it can schedule in a linear,
        exponential, step or multistep function. One can also Callables to have customized
        scheduling logic.
    """

    def __init__(
        self,
        parameter_setter: Callable,
        value_calculator: Union[str, Callable],
        vc_kwargs: Dict,
        epoch_level: bool = False,
        name: Optional[str] = None,
    ):
        """
        :param parameter_setter: Callable function that sets the required parameter
        :param value_calculator: Either a string ('linear', 'exponential', 'step' or 'multistep')
         or Callable for custom logic.
        :param vc_kwargs: Dictionary that stores the required parameters for the value_calculator.
        :param epoch_level: Call the scheduler every epoch or every iteration.
        :param name: Identifier of logging.logger to use, if None, defaulting to ``engine.logger``.
        """
        self.epoch_level = epoch_level

        self._calculators = {
            "linear": self._linear,
            "exponential": self._exponential,
            "step": self._step,
            "multistep": self._multistep,
        }

        self._parameter_setter = parameter_setter
        self._vc_kwargs = vc_kwargs
        self._value_calculator = self._get_value_calculator(
            value_calculator=value_calculator
        )

        self.logger = logging.getLogger(name)
        self._name = name

    def _get_value_calculator(self, value_calculator: Union[str, Callable]):
        if isinstance(value_calculator, str):
            return self._calculators[value_calculator]
        elif isinstance(value_calculator, Callable):
            return value_calculator
        else:
            raise ValueError(
                f"value_calculator must be either a string from {list(self._calculators.keys())} or a Callable."
            )

    def __call__(self, engine: Engine):
        if self.epoch_level:
            self._vc_kwargs["current_step"] = engine.state.epoch
        else:
            self._vc_kwargs["current_step"] = engine.state.iteration

        new_value = self._value_calculator(**self._vc_kwargs)
        self._parameter_setter(new_value)

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine that is used for training.
        """
        if self._name is None:
            self.logger = engine.logger

        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED, self)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    @staticmethod
    def _linear(
        initial_value: int,
        step_constant: int,
        step_max_value: int,
        max_value: float,
        current_step: int,
    ) -> float:
        """
        Keeps the parameter value to zero until step_zero steps passed and then linearly
        increases it to 1 until an additional step_one steps passed. Continues the trend
        until it reaches max_value.

        :param initial_value: Starting value of the parameter.
        :param step_constant: Step index until parameter's value is kept constant.
        :param step_max_value: Additional number of steps until parameter's value becomes max_value.
        :param max_value: Max parameter value.
        :param current_step: Current step index.
        :return: new parameter value
        """
        if current_step < step_constant:
            delta = initial_value
        elif current_step > step_max_value:
            delta = max_value - initial_value
        else:
            delta = (max_value - initial_value) * (
                (current_step - step_constant) / step_max_value
            )

        return initial_value + delta

    @staticmethod
    def _exponential(initial_value: float, gamma: float, current_step: int) -> float:
        """
        Decays the parameter value by gamma every step.

        Based on ExponentialLR from Pytorch
        https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py#L457

        :param initial_value: Starting value of the parameter.
        :param gamma: Multiplicative factor of parameter value decay.
        :param current_step: Current step index.
        :return: new parameter value
        """
        return initial_value * gamma ** current_step

    @staticmethod
    def _step(
        initial_value: float, gamma: float, step_size: int, current_step: int
    ) -> float:
        """
        Decays the parameter value by gamma every step_size.

        Based on StepLR from Pytorch.
        https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py#L377

        :param initial_value: Starting value of the parameter.
        :param gamma: Multiplicative factor of parameter value decay.
        :param step_size: Period of parameter value decay
        :param current_step: Current step index.
        :return: new parameter value
        """
        return initial_value * gamma ** (current_step // step_size)

    @staticmethod
    def _multistep(
        initial_value: float, gamma: float, milestones: List[int], current_step: int
    ) -> float:
        """
        Decays the parameter value by gamma once the number of steps reaches one of the milestones.

        Based on MultiStepLR from Pytorch.
        https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py#L424

        :param initial_value: Starting value of the parameter.
        :param gamma: Multiplicative factor of parameter value decay.
        :param milestones: List of step indices. Must be increasing.
        :param current_step: Current step index.
        :return: new parameter value
        """
        return initial_value * gamma ** bisect_right(milestones, current_step)


class TBSummaryTypes(Enum):
    SCALAR = "scalar"
    SCALARS = "scalars"
    HISTOGRAM = "histogram"
    IMAGE = "image"
    IMAGE_AXIAL = "image_axial"
    IMAGE_CORONAL = "image_coronal"
    IMAGE_SAGITTAL = "image_sagittal"
    IMAGE3_AXIAL = "image3_axial"
    IMAGE3_CORONAL = "image3_coronal"
    IMAGE3_SAGITTAL = "image3_sagittal"
    IMAGE3 = "image3"
    IMAGES = "images"
    IMAGE_WITH_BOXES = "image_with_boxes"
    FIGURE = "figure"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"


class TensorBoardHandler(object):
    """
    Update version the MONAI TensorBoard logging handler.

    It first tries to logg any SCALAR SUMMARIES found in engine.state.metrics dictionary.

    Following that it will iterate over possible summaries as defined in TBSummaryTypes and search for their existence
    in engine.state.output["summaries"].

    The following summaries type are tailored to medical imaging:
        - image_axial - mid axial slice of a 3D volume is saved as an image summary
        - image_coronal - mid coronal slice of a 3D volume is saved as an image summary
        - image_sagittal - mid coronal slice of a 3D volume is saved as an image summary
        - image3_axial - a GIF along the axial slice of the image
        - image3_coronal - a GIF along the coronal slice of the image
        - image3_sagittal - a GIF along the sagittal slice of the image

    Args:
        summary_writer (SummaryWriter): User can specify TensorBoard SummaryWriter. Default to create a new writer.

        log_dir (str): If using default SummaryWriter, write logs to this directory. Default is `./runs`.

        interval (int): Logs every N epochs or every N iterations. Defaults to 1.

        epoch_level (bool): Log content every N epochs or N iterations. `True` is epoch level, `False` is iteration
        level. Defaults to True.

        global_step_transform (Callable): Callable that is used to customize global step number for TensorBoard. For
        example, in evaluation, the evaluator engine needs to know current epoch from trainer. Defaults to lambda x: x

        clamp_images (bool): Whether we clamp the image based summaries. This is done so we do not have wrap around
        effect in TensorBoard. Defaults to True.

        clamp_range (Tuple[int,int]): To what range we clamp the image based summaries. Defaults to (-1.0, 1.0)
    """

    def __init__(
        self,
        summary_writer: Optional[SummaryWriter],
        log_dir: str = "./runs",
        interval: int = 1,
        epoch_level: bool = True,
        global_step_transform: Callable = lambda x: x,
        clamp_images: bool = True,
        clamp_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self._writer = (
            SummaryWriter(log_dir=log_dir) if summary_writer is None else summary_writer
        )
        self.interval = interval
        self.epoch_level = epoch_level
        self.global_step_transform = global_step_transform
        self.clamp_images = clamp_images
        self.clamp_range = clamp_range

    def attach(self, engine: Engine) -> None:
        if self.epoch_level:
            engine.add_event_handler(
                Events.EPOCH_COMPLETED(every=self.interval), self.log
            )
        else:
            engine.add_event_handler(
                Events.ITERATION_COMPLETED(every=self.interval), self.log
            )

    def log(self, engine: Engine) -> None:
        step = self.global_step_transform(
            engine.state.epoch if self.epoch_level else engine.state.iteration
        )

        try:
            for name, value in engine.state.metrics.items():
                self._writer.add_scalar(name, value, step)
            self._writer.flush()
        except AttributeError:
            pass

        summaries = engine.state.output["summaries"]
        if TBSummaryTypes.SCALAR in summaries:
            for name, value in summaries[TBSummaryTypes.SCALAR].items():
                self._writer.add_scalar(name, value, global_step=step)
        if TBSummaryTypes.SCALARS in summaries:
            for main_tag, tag_scalar_dict in summaries[TBSummaryTypes.SCALARS].items():
                self._writer.add_scalars(main_tag, tag_scalar_dict, global_step=step)
        if TBSummaryTypes.HISTOGRAM in summaries:
            for tag, values in summaries[TBSummaryTypes.HISTOGRAM].items():
                self._writer.add_histogram(tag=tag, values=values, global_step=step)
        if TBSummaryTypes.IMAGE in summaries:
            for tag, img_tensor in summaries[TBSummaryTypes.IMAGE].items():
                self._writer.add_image(
                    tag=tag,
                    img_tensor=self._prepare_image(img_tensor),
                    global_step=step,
                )
        if TBSummaryTypes.IMAGE_CORONAL in summaries:
            for tag, img_tensor in summaries[TBSummaryTypes.IMAGE_CORONAL].items():
                self._writer.add_images(
                    tag=tag,
                    img_tensor=torch.squeeze(
                        self._prepare_image(img_tensor)[
                            :, :, :, img_tensor.shape[3] // 2, :
                        ],
                        dim=2,
                    ).rot90(dims=[2,3]),
                    global_step=step,
                )
        if TBSummaryTypes.IMAGE_AXIAL in summaries:
            for tag, img_tensor in summaries[TBSummaryTypes.IMAGE_AXIAL].items():
                self._writer.add_images(
                    tag=tag,
                    img_tensor=torch.squeeze(
                        self._prepare_image(img_tensor)[
                            :, :, :, :, img_tensor.shape[4] // 2
                        ],
                        dim=3,
                    ).rot90(dims=[2,3]),
                    global_step=step,
                )
        if TBSummaryTypes.IMAGE_SAGITTAL in summaries:
            for tag, img_tensor in summaries[TBSummaryTypes.IMAGE_SAGITTAL].items():
                self._writer.add_images(
                    tag=tag,
                    img_tensor=self._prepare_image(img_tensor)[
                        :, :, img_tensor.shape[2] // 2, :, :
                    ].rot90(dims=[2,3]),
                    global_step=step,
                )
        if TBSummaryTypes.IMAGE3_CORONAL in summaries:
            for tag, vid_tensor in summaries[TBSummaryTypes.IMAGE3_CORONAL].items():
                self._writer.add_video(
                    tag=tag,
                    vid_tensor=self._prepare_image(
                        vid_tensor.permute(0, 3, 1, 2, 4).contiguous()
                    ).rot90(dims=[3,4]),
                    global_step=step,
                )
        if TBSummaryTypes.IMAGE3_AXIAL in summaries:
            for tag, vid_tensor in summaries[TBSummaryTypes.IMAGE3_AXIAL].items():
                self._writer.add_video(
                    tag=tag,
                    vid_tensor=self._prepare_image(
                        vid_tensor.permute(0, 4, 1, 2, 3).contiguous()
                    ).rot90(dims=[3,4]),
                    global_step=step,
                )
        if TBSummaryTypes.IMAGE3_SAGITTAL in summaries:
            for tag, vid_tensor in summaries[TBSummaryTypes.IMAGE3_SAGITTAL].items():
                self._writer.add_video(
                    tag=tag,
                    vid_tensor=self._prepare_image(
                        vid_tensor.permute(0, 2, 1, 3, 4).contiguous()
                    ).rot90(dims=[3,4]),
                    global_step=step,
                )
        if TBSummaryTypes.IMAGES in summaries:
            for tag, img_tensor in summaries[TBSummaryTypes.IMAGES].items():
                self._writer.add_images(
                    tag=tag,
                    img_tensor=self._prepare_image(img_tensor),
                    global_step=step,
                )
        if TBSummaryTypes.IMAGE_WITH_BOXES in summaries:
            for tag, args in summaries[TBSummaryTypes.IMAGE_WITH_BOXES].items():
                self._writer.add_image_with_boxes(
                    tag=tag,
                    img_tensor=self._prepare_image(args["img_tensor"]),
                    box_tensor=args["box_tensor"],
                    global_step=step,
                )
        if TBSummaryTypes.FIGURE in summaries:
            for tag, figure in summaries[TBSummaryTypes.FIGURE].items():
                self._writer.add_figure(tag=tag, figure=figure, global_step=step)
        if TBSummaryTypes.VIDEO in summaries:
            for tag, vid_tensor in summaries[TBSummaryTypes.VIDEO].items():
                self._writer.add_video(tag=tag, vid_tensor=vid_tensor, global_step=step)
        if TBSummaryTypes.AUDIO in summaries:
            for tag, snd_tensor in summaries[TBSummaryTypes.AUDIO].items():
                self._writer.add_audio(tag=tag, snd_tensor=snd_tensor, global_step=step)
        if TBSummaryTypes.TEXT in summaries:
            for tag, text_string in summaries[TBSummaryTypes.TEXT].items():
                self._writer.add_text(
                    tag=tag, text_string=text_string, global_step=step
                )

        self._writer.flush()

    def add_graph(self, model, input_to_model=None, verbose=False):
        self._writer.add_graph(
            model=model, input_to_model=input_to_model, verbose=verbose
        )

    def add_hparams(
        self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None
    ):
        self._writer.add_hparams(
            hparam_dict=hparam_dict,
            metric_dict=metric_dict,
            hparam_domain_discrete=hparam_domain_discrete,
            run_name=run_name,
        )

    def _prepare_image(self, image):
        if self.clamp_images:
            image = torch.clamp(image, min=self.clamp_range[0], max=self.clamp_range[1])

        return image


class LossSummaryHandler:
    """
    Handler that fetches the summaries the loss stores inside itself and update the summaries dictionary
    inside the ignite.engine.state.output

    Args:
        loss (Union[Module, _Loss]): Loss that implements .get_summaries() which returns a dictionary with summaries.
    """

    def __init__(self, loss: Union[Module, _Loss]):
        self.loss = loss

    def __call__(self, engine: Engine):
        self._update(
            accumulator=engine.state.output.get("summaries", {}),
            values=self.loss.get_summaries(),
        )

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def _update(self, accumulator:Dict, values:Dict):
        """
        Recursive update function which updates dictionary d with elements from dictionary u

        Args:
            accumulator (Dict): dictionary that accumulates summaries
            values (Dict): dictionary with new summaries

        Returns:
            Dict: updated accumulator
        """
        for k, v in values.items():
            if isinstance(v, collections.abc.Mapping):
                accumulator[k] = self._update(accumulator.get(k, {}), v)
            else:
                accumulator[k] = v
        return accumulator


class EvaluationHandler:
    """
    Lightweight wrapper that allows us to run evaluation as a handler.

    Args:
        evaluation_engine (Trainer): The trainer which runs the evaluation.
        evaluate_every (int): After how many epochs/iterations the evaluation is ran.
        epoch_level (bool):  Call the scheduler every epoch or every iteration.
    """

    def __init__(
        self, evaluation_engine: Trainer, evaluate_every: int, epoch_level: bool = True
    ):
        self.evaluation_engine = evaluation_engine
        self.epoch_level = epoch_level
        self.evaluate_every = evaluate_every

    def __call__(self, engine: Trainer):
        self.evaluation_engine.run()

    def attach(self, engine: Engine) -> None:
        if self.epoch_level:
            engine.add_event_handler(
                Events.EPOCH_COMPLETED(every=self.evaluate_every), self
            )
        else:
            engine.add_event_handler(
                Events.ITERATION_COMPLETED(every=self.evaluate_every), self
            )


class LoggingPreparationHandler:
    """
    Handler that guarantees the existence of the summary dictionary in the engine.state.output.
    That the place where the current codebase stores all the summaries.
    """
    def __init__(self):
        self.empty_summaries = {"summaries": dict()}

        for summary in TBSummaryTypes:
            self.empty_summaries[summary] = {}

    def __call__(self, engine: Engine):
        engine.state.output["summaries"] = deepcopy(self.empty_summaries)

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)


class NpySaver:
    """
    Event handler triggered on completing every iteration to save the arbitrary elements into npy files.
    Args:
        output_dir: output image directory.
        output_postfix: a string appended to all output file names.
        dtype: convert the image data to save to this data type. If None, keep the original type of data.
            It's used for Nifti format only.
        batch_transform: a callable that is used to transform the ignite.engine.batch into expected format to extract
            the meta_data dictionary.
        output_transform: a callable that is used to transform the ignite.engine.output into the form expected image
            data. The first dimension of this transform's output will be treated as the batch dimension. Each item in
            the batch will be saved individually.
        name: identifier of logging.logger to use, defaulting to `engine.logger`.
    """
    def __init__(
        self,
        output_dir: str = "./",
        output_postfix: str = "seg",
        dtype: Optional[np.dtype] = None,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        name: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.output_postfix = output_postfix
        self.dtype = dtype
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.output_ext = ".npy"
        self.logger = logging.getLogger(name)

        self._name = name
        self._data_index = 0

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self._name is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def __call__(self, engine: Engine) -> None:
        """
        This method assumes self.batch_transform will extract metadata from the input batch.
        Output file datatype is determined from ``engine.state.output.dtype``.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        meta_data = self.batch_transform(engine.state.batch)
        engine_output = self.output_transform(engine.state.output)
        self._save_batch(engine_output, meta_data)
        self.logger.info("saved all the model outputs into files.")

    def _save_batch(
        self,
        batch_data: Union[torch.Tensor, np.ndarray],
        meta_data: Optional[Dict] = None,
    ) -> None:
        """Save a batch of data into npy format files.

        Args:
            batch_data: target batch data content that save into npy format.
            meta_data: every key-value in the meta_data is corresponding to a batch of data.
        """
        for i, data in enumerate(batch_data):  # save a batch of files
            self.save(
                data, {k: meta_data[k][i] for k in meta_data} if meta_data else None
            )

    def save(
        self, data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None
    ) -> None:
        """
        Save data into a npy file.
        The meta_data could optionally have the following keys:

            - ``'filename_or_obj'`` -- for output file name creation, corresponding to filename or object.

        If meta_data is None, use the default index (starting from 0) as the filename.

        Args:
            data: target data content that to be saved as a npy format file.
            meta_data: the meta data information corresponding to the data.
        """
        filename = meta_data["filename_or_obj"] if meta_data else str(self._data_index)
        self._data_index += 1

        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        data = data.astype(self.dtype)

        filename = create_file_basename(self.output_postfix, filename, self.output_dir)
        filename = f"{filename}{self.output_ext}"

        np.save(file=filename, arr=data)


class MaxEpochsHandler:
    """
    Handler that allows a model to further be trained passed its initial max_epochs.

    This handler must be attached after the monai.handlers.CheckpointLoader was attached so it overwrites the max_epochs
    of the loaded checkpoint.

    Args:
        max_epochs (int): The new max_epochs.
    """
    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs

    def __call__(self, engine: Engine):
        engine.state.max_epochs = self.max_epochs

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.STARTED, self)
