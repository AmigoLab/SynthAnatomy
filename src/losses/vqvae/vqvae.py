from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from numpy import fromfunction
from torch.fft import fftn
from torch.nn.modules.loss import _Loss

from src.handlers.general import TBSummaryTypes


class MSELoss(_Loss):
    """
    Mean Squared Error loss.

    Args:
        size_average (bool): Deprecated (see reduction). By default, the losses are averaged over each loss element
            in the batch. Note that for some losses, there are multiple elements per sample. If the field
            size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is
            False. Default: True
        reduce (bool): Deprecated (see reduction). By default, the losses are averaged or summed over observations
            for each minibatch depending on size_average. When reduce is False, returns a loss per batch element
            instead and ignores size_average. Default: True
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no
            reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in
            the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being
            deprecated, and in the meantime, specifying either of those two args will override reduction.
            Default: 'mean'

    Attributes:
        self.summaries (Dict): Dictionary containing scalar summaries to be logged in TensorBoard

    """

    def __init__(
        self, size_average: bool = None, reduce: bool = None, reduction: str = "mean"
    ):
        super(MSELoss, self).__init__(size_average, reduce, reduction)

        try:
            assert reduction in ["sum", "mean"]
        except AssertionError:
            raise ValueError("Reduction must be either 'sum' or 'mean'")

        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    def forward(
        self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor
    ) -> torch.Tensor:
        y = y.float()
        y_pred = network_output["reconstruction"][0].float()
        q_losses = network_output["quantization_losses"]

        loss = F.mse_loss(y_pred, y)
        self.summaries[TBSummaryTypes.SCALAR]["Loss-MSE-Reconstruction"] = loss

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][
                f"Loss-MSE-VQ{idx}_Commitment_Cost"
            ] = q_loss

            loss += q_loss

        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries


class BaurLoss(_Loss):
    """
    Loss based on [1] with image gradients loss implementation based on [2]

    Args:
        size_average (bool): Deprecated (see reduction). By default, the losses are averaged over each loss element
            in the batch. Note that for some losses, there are multiple elements per sample. If the field
            size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is
            False. Default: True
        reduce (bool): Deprecated (see reduction). By default, the losses are averaged or summed over observations
            for each minibatch depending on size_average. When reduce is False, returns a loss per batch element
            instead and ignores size_average. Default: True
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no
            reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in
            the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being
            deprecated, and in the meantime, specifying either of those two args will override reduction.
            Default: 'mean'

    Attributes:
        self.dx (nn.ConstantPad3d): Shift on x axis to calculate the image gradients
        self.dy (nn.ConstantPad3d): Shift on y axis to calculate the image gradients
        self.dz (nn.ConstantPad3d): Shift on z axis to calculate the image gradients
        self.gdl_factor (float): Scaling factor for the image gradient component of the loss
        self.summaries (Dict): Dictionary containing scalar summaries to be logged in TensorBoard

    References:
        [1] Baur, C., Wiestler, B., Albarqouni, S., and Navab, N. 2019.
        Fusing unsupervised and supervised deep learning for white matter lesion segmentation.
        In International Conference on Medical Imaging with Deep Learning (pp. 63–72).

        [2] Micha'el Mathieu and Camille Couprie and Yann LeCun 2016.
        Deep multi-scale video prediction beyond mean square error.
        In 4th International Conference on Learning Representations, ICLR 2016,
        San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings.
        https://github.com/coupriec/VideoPredictionICLR2016/blob/802e902ec91c8c43eed4cbd0365eddf312339074/image_error_measures.lua#L123
    """

    def __init__(
        self, size_average: bool = None, reduce: bool = None, reduction: str = "mean"
    ):
        super(BaurLoss, self).__init__(size_average, reduce, reduction)

        try:
            assert reduction in ["sum", "mean"]
        except AssertionError:
            raise ValueError("Reduction must be either 'sum' or 'mean'")

        self.dx = nn.ConstantPad3d((1, -1, 0, 0, 0, 0), 0)
        self.dy = nn.ConstantPad3d((0, 0, 1, -1, 0, 0), 0)
        self.dz = nn.ConstantPad3d((0, 0, 0, 0, 1, -1), 0)
        self.gdl_factor: float = 0.0
        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    def forward(
        self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor
    ) -> torch.Tensor:
        y = y.float()
        y_pred = network_output["reconstruction"][0].float()
        q_losses = network_output["quantization_losses"]

        l1_loss = F.l1_loss(y_pred, y, reduction=self.reduction)
        self.summaries[TBSummaryTypes.SCALAR]["Loss-MAE-Reconstruction"] = l1_loss

        l2_loss = F.mse_loss(y_pred, y, reduction=self.reduction)
        self.summaries[TBSummaryTypes.SCALAR]["Loss-MSE-Reconstruction"] = l2_loss

        gdl_loss = (
            getattr(
                torch.abs(
                    torch.abs((self.dx(y) - y))[:, :, 1:-1, 1:-1, 1:-1]
                    - torch.abs((self.dx(y_pred) - y_pred))[:, :, 1:-1, 1:-1, 1:-1]
                )
                + torch.abs(
                    torch.abs((self.dy(y) - y))[:, :, 1:-1, 1:-1, 1:-1]
                    - torch.abs((self.dy(y_pred) - y_pred))[:, :, 1:-1, 1:-1, 1:-1]
                )
                + torch.abs(
                    torch.abs((self.dz(y) - y))[:, :, 1:-1, 1:-1, 1:-1]
                    - torch.abs((self.dz(y_pred) - y_pred))[:, :, 1:-1, 1:-1, 1:-1]
                ),
                self.reduction,
            )()
            * self.gdl_factor
        )

        self.summaries[TBSummaryTypes.SCALAR]["Loss-GDL-Reconstruction"] = gdl_loss

        self.summaries[TBSummaryTypes.SCALAR]["Auxiliary-GDL_Factor"] = self.gdl_factor

        loss = l1_loss + l2_loss + gdl_loss

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][
                f"Loss-MSE-VQ{idx}_Commitment_Cost"
            ] = q_loss

            loss += q_loss

        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries

    def get_gdl_factor(self) -> float:
        return self.gdl_factor

    def set_gdl_factor(self, gdl_factor: float) -> float:
        self.gdl_factor = gdl_factor

        return self.get_gdl_factor()


class SpectralLoss(_Loss):
    """
    Loss function that has a spectral component based on the amplitude and phase of FFT
    and a pixel component based on mean absolute error and mean squared error.

    Args:
        dimensions (int): Number of spatial dimensions.
        include_pixel_loss (bool): If the loss includes the pixel component as well
        fft_kwargs (Dict): Dictionary hold all FFT arguments that are to be used when calling torch.fft.fftn.
            Defaults to:  {'s': None, 'dims': tuple(range(1, self.dimensions + 2)), 'norm': 'ortho'}
        size_average (bool): Deprecated (see reduction). By default, the losses are averaged over each loss element
            in the batch. Note that for some losses, there are multiple elements per sample. If the field
            size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is
            False. Default: True
        reduce (bool): Deprecated (see reduction). By default, the losses are averaged or summed over observations
            for each minibatch depending on size_average. When reduce is False, returns a loss per batch element
            instead and ignores size_average. Default: True
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no
            reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in
            the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being
            deprecated, and in the meantime, specifying either of those two args will override reduction.
            Default: 'mean'

    Attributes:
        self.dimensions (int): Number of spatial dimensions.
        self.include_pixel_loss (bool): If the loss includes the pixel component as well
        self.pi (float): Pi value use in the fourier phase calculation
        self.fft_factor (float): Scaling factor of the spectral component of loss
        self.fft_kwargs (Dict): Dictionary containing key words arguments that will be passed to the torch.fft.fftn
            function call.
        self.summaries (Dict): Dictionary containing scalar summaries to be logged in TensorBoard

    References:
        [1] Takaki, S., Nakashika, T., Wang, X. and Yamagishi, J., 2019, May.
        STFT spectral loss for training a neural speech waveform model.
        In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 7065-7069). IEEE.
    """

    def __init__(
        self,
        dimensions: int,
        include_pixel_loss: bool = True,
        fft_kwargs: Dict = None,
        size_average: bool = True,
        reduce: bool = True,
        reduction: str = "mean",
    ):
        super(SpectralLoss, self).__init__(size_average, reduce, reduction)

        self.dimensions = dimensions
        self.include_pixel_loss = include_pixel_loss
        self.fft_factor: float = 1.0
        self.fft_kwargs = (
            {"s": None, "dim": tuple(range(1, self.dimensions + 2)), "norm": "ortho"}
            if fft_kwargs is None
            else fft_kwargs
        )
        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    def forward(
        self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor
    ) -> torch.Tensor:
        # Unpacking elements
        y = y.float()
        y_pred = network_output["reconstruction"][0].float()
        q_losses = network_output["quantization_losses"]

        # Calculating amplitudes and phases
        y_amplitude, y_phase = self._get_fft_amplitude_and_phase(y)
        yp_amplitude, yp_phase = self._get_fft_amplitude_and_phase(y_pred)

        # Ref 1 - Sec 2.2 - Equation 7
        amplitude_loss = 0.5 * F.mse_loss(yp_amplitude, y_amplitude)
        self.summaries[TBSummaryTypes.SCALAR][
            "Loss-Amplitude-Reconstruction"
        ] = amplitude_loss

        # Ref 1 - Sec 2.3 - Equation 10
        phase_loss = torch.mean(
            0.5 * torch.abs((1 - torch.exp(torch.abs(yp_phase - y_phase))) ** 2)
        )
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Phase-Reconstruction"] = phase_loss
        fft_loss = (amplitude_loss + phase_loss) * self.fft_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Spectral-Reconstruction"] = fft_loss
        self.summaries[TBSummaryTypes.SCALAR]["Auxiliary-FFT_Factor"] = self.fft_factor

        loss = fft_loss

        if self.include_pixel_loss:
            l2_loss = F.mse_loss(y_pred, y, reduction=self.reduction)
            self.summaries[TBSummaryTypes.SCALAR]["Loss-MSE-Reconstruction"] = l2_loss

            loss = loss + l2_loss

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][
                f"Loss-MSE-VQ{idx}_Commitment_Cost"
            ] = q_loss

            loss += q_loss

        return loss

    def _get_fft_amplitude_and_phase(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Manually calculating the amplitude and phase of the fourier transformations representation of the images

        Args:
            images (torch.Tensor): Images that are to undergo fftn

        Returns:
            torch.Tensor: fourier transformation amplitude
            torch.Tensor: fourier transformation phase

        """
        img_fft = fftn(input=images, **self.fft_kwargs)

        amplitude = torch.sqrt(img_fft.real ** 2 + img_fft.imag ** 2)
        phase = torch.atan2(img_fft.imag, img_fft.real)

        return amplitude, phase

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries

    def get_fft_factor(self) -> float:
        return self.fft_factor

    def set_fft_factor(self, fft_factor: float) -> float:
        self.fft_factor = fft_factor

        return self.get_fft_factor()


class HartleyLoss(_Loss):
    """
    Loss function that has a spectral component based on the Hartley representation and a pixel component based on
    mean absolute error and mean squared error.

    Args:
        dimensions (int): Dimensions: number of spatial dimensions.
        include_pixel_loss (bool): If the loss includes the pixel component as well
        size_average (bool): Deprecated (see reduction). By default, the losses are averaged over each loss element
            in the batch. Note that for some losses, there are multiple elements per sample. If the field
            size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is
            False. Default: True
        fft_kwargs (Dict): Dictionary hold all FFT arguments that are to be used when calling torch.fft.fftn.
            Defaults to:  {'s': None, 'dims': tuple(range(1, self.dimensions + 2)), 'norm': 'ortho'}
        prioritise_high_frequency (bool): Whether to increase the importance of the high frequencies based on the
            equation 11 from [1].
        reduce (bool): Deprecated (see reduction). By default, the losses are averaged or summed over observations
            for each minibatch depending on size_average. When reduce is False, returns a loss per batch element
            instead and ignores size_average. Default: True
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no
            reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in
            the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being
            deprecated, and in the meantime, specifying either of those two args will override reduction.
            Default: 'mean'

    Attributes:
        self.dimensions (int): Number of spatial dimensions.
        self.include_pixel_loss (bool): If the loss includes the pixel component as well
        self.fht_factor (float): Scaling factor of the spectral component of loss
        self.fht_weight (torch.Tensor): Weight tensor that scales the hartley transformation in favor of high
            frequencies.
        self.fft_kwargs (Dict): Dictionary containing key words arguments that will be passed to the torch.fft.fftn
            function call.
        self.prioritise_high_frequency (bool): Whether to increase the importance of the high frequencies
        self.summaries (Dict): Dictionary containing scalar summaries to be logged in TensorBoard

    References:
        [1] Xue, S., Qiu, W., Liu, F. and Jin, X., 2020.
        Faster image super-resolution by improved frequency-domain neural networks.
        Signal, Image and Video Processing, 14(2), pp.257-265.
    """

    def __init__(
        self,
        dimensions: int,
        include_pixel_loss: bool = True,
        fft_kwargs: Dict = None,
        prioritise_high_frequency: bool = True,
        size_average: bool = None,
        reduce: bool = None,
        reduction: str = "mean",
    ):
        super(HartleyLoss, self).__init__(size_average, reduce, reduction)

        self.dimensions = dimensions
        self.include_pixel_loss = include_pixel_loss
        self.fft_kwargs = (
            {"s": None, "dim": tuple(range(1, self.dimensions + 2)), "norm": "ortho"}
            if fft_kwargs is None
            else fft_kwargs
        )
        self.prioritise_high_frequency = prioritise_high_frequency
        self.fht_weight: torch.Tensor = torch.rand(1)
        self.fht_factor: float = 1.0
        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    def forward(
        self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor
    ) -> torch.Tensor:
        # Unpacking elements
        y = y.float()
        y_pred = network_output["reconstruction"][0].float()
        q_losses = network_output["quantization_losses"]

        # Calculating the Hartley representation
        y_fht = self._fhtn(images=y)
        y_pred_fht = self._fhtn(images=y_pred)

        if self.prioritise_high_frequency:
            weight = self._calculate_frequency_weight(images=y)
            y_fht = y_fht * weight
            y_pred_fht = y_pred_fht * weight

        # Ref 1 - Sec 3.4 - Eq 13
        # Deviation from a proper implementation due to scale imbalances
        # the Hartley loss would be 15+ orders of magnitude higher at the beginning
        # loss = 0.5 * torch.norm((y_pred_fht - y_fht) * weight, p="fro") ** 2
        loss = 0.5 * F.mse_loss(y_pred_fht, y_fht)
        loss = loss * self.fht_factor
        self.summaries[TBSummaryTypes.SCALAR][
            "Auxiliary-Hartley_Factor"
        ] = self.fht_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Hartley-Reconstruction"] = loss

        if self.include_pixel_loss:
            l2_loss = F.mse_loss(y_pred, y, reduction=self.reduction)
            self.summaries[TBSummaryTypes.SCALAR]["Loss-MSE-Reconstruction"] = l2_loss

            loss = loss + l2_loss

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][
                f"Loss-MSE-VQ{idx}_Commitment_Cost"
            ] = q_loss

            loss += q_loss

        return loss

    def _fhtn(self, images: torch.Tensor) -> torch.Tensor:
        """
        Calculates the hartley transformation of the given images based on the fourier transformation relation.

        Args:
            images (torch.Tensor): Images that are to undergo hartley transformation

        Returns:
            torch.Tensor: hartley transformation representation
        """
        img_fft = fftn(input=images, **self.fft_kwargs)
        ht = img_fft.real - img_fft.imag
        return ht

    def _calculate_frequency_weight(self, images: torch.Tensor) -> torch.Tensor:
        """
        Calculating a weight matrix that will be applied to scale the hartley representations in favour of
        high frequencies.

        Based on equation equation 11 from [1].

        Args:
            images (torch.Tensor): Images that are dictating the shape

        Returns:
            torch.Tensor: High frequency focused weighting

        Raises:
            ValueError: If the tensors fed for shape reference is not a 4D or 5D tensor
        """
        if images.shape != self.fht_weight.shape:
            # We have 2D images
            if len(images.shape) == 4:
                mx = images.shape[2]
                my = images.shape[3]

                def hartley_weight(x, y) -> float:
                    return (abs(mx / 2 - x) / (mx / 2)) ** 2 + (
                        abs(my / 2 - y) / (my / 2)
                    ) ** 2

            # We have 3D images
            elif len(images.shape) == 5:
                mx = images.shape[2]
                my = images.shape[3]
                mz = images.shape[4]

                def hartley_weight(x, y, z) -> float:
                    return (
                        (abs(mx / 2 - x) / (mx / 2)) ** 2
                        + (abs(my / 2 - y) / (my / 2)) ** 2
                        + (abs(mz / 2 - z) / (mz / 2)) ** 2
                    )

            else:
                raise ValueError(
                    "HartleyLoss is implemented only for 2D and 3D images."
                )

            weight = torch.from_numpy(
                fromfunction(hartley_weight, shape=images.shape[2:])
            ).to(images.device)

            weight = torch.exp(weight)

            weight = weight - torch.min(weight)
            weight = weight / torch.max(weight)
            weight = weight + 0.0001

            self.fht_weight = weight

        return self.fht_weight

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries

    def get_fht_factor(self) -> float:
        return self.fht_factor

    def set_fht_factor(self, fht_factor: float) -> float:
        self.fht_factor = fht_factor

        return self.get_fht_factor()


class JukeboxLoss(_Loss):
    """
    Loss function that has a spectral component based on the magnitude of FFT and a pixel component based on mean
    absolute error and mean squared error.

    Args:
        dimensions (int): Number of spatial dimensions.
        include_pixel_loss (bool): If the loss includes the pixel component as well
        fft_kwargs (Dict): Dictionary hold all FFT arguments that are to be used when calling torch.fft.fftn.
            Defaults to:  {'s': None, 'dims': tuple(range(1, self.dimensions + 2)), 'norm': 'ortho'}
        size_average (bool): Deprecated (see reduction). By default, the losses are averaged over each loss element
            in the batch. Note that for some losses, there are multiple elements per sample. If the field
            size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is
            False. Default: True
        reduce (bool): Deprecated (see reduction). By default, the losses are averaged or summed over observations
            for each minibatch depending on size_average. When reduce is False, returns a loss per batch element
            instead and ignores size_average. Default: True
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no
            reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in
            the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being
            deprecated, and in the meantime, specifying either of those two args will override reduction.
            Default: 'mean'

    Attributes:
        self.dimensions (int): Number of spatial dimensions.
        self.include_pixel_loss (bool): If the loss includes the pixel component as well
        self.fft_factor (float): Scaling factor of the spectral component of loss
        self.fft_kwargs (Dict): Dictionary containing key words arguments that will be passed to the torch.fft.fftn
            function call.
        self.summaries (Dict): Dictionary containing scalar summaries to be logged in TensorBoard

    References:
        [1] Dhariwal, P., Jun, H., Payne, C., Kim, J.W., Radford, A. and Sutskever, I., 2020.
        Jukebox: A generative model for music.
        arXiv preprint arXiv:2005.00341.
    """

    def __init__(
        self,
        dimensions: int,
        include_pixel_loss: bool = True,
        fft_kwargs: Dict = None,
        size_average: bool = True,
        reduce: bool = True,
        reduction: str = "mean",
    ):
        super(JukeboxLoss, self).__init__(size_average, reduce, reduction)

        self.dimensions = dimensions
        self.include_pixel_loss = include_pixel_loss
        self.fft_factor: float = 1.0
        self.fft_kwargs = (
            {"s": None, "dim": tuple(range(1, self.dimensions + 2)), "norm": "ortho"}
            if fft_kwargs is None
            else fft_kwargs
        )
        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    def forward(
        self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor
    ) -> torch.Tensor:
        # Unpacking elements
        y = y.float()
        y_pred = network_output["reconstruction"][0].float()
        q_losses = network_output["quantization_losses"]

        y_amplitude = self._get_fft_amplitude(y)
        y_pred_amplitude = self._get_fft_amplitude(y_pred)

        # Ref 1 - Section 3.3 - L_spec
        loss = F.mse_loss(y_pred_amplitude, y_amplitude) * self.fft_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Spectral-Reconstruction"] = loss
        self.summaries[TBSummaryTypes.SCALAR]["Auxiliary-FFT_Factor"] = self.fft_factor

        if self.include_pixel_loss:
            l2_loss = F.mse_loss(y_pred, y, reduction=self.reduction)
            self.summaries[TBSummaryTypes.SCALAR]["Loss-MSE-Reconstruction"] = l2_loss

            loss = loss + l2_loss

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][
                f"Loss-MSE-VQ{idx}_Commitment_Cost"
            ] = q_loss

            loss = loss + q_loss

        return loss

    def _get_fft_amplitude(self, images: torch.Tensor) -> torch.Tensor:
        """
        Manually calculating the amplitude of the fourier transformations representation of the images

        Args:
            images (torch.Tensor): Images that are to undergo fftn

        Returns:
            torch.Tensor: fourier transformation amplitude
        """
        img_fft = fftn(input=images, **self.fft_kwargs)

        amplitude = torch.sqrt(img_fft.real ** 2 + img_fft.imag ** 2)

        return amplitude

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries

    def get_fft_factor(self) -> float:
        return self.fft_factor

    def set_fft_factor(self, fft_factor: float) -> float:
        self.fft_factor = fft_factor

        return self.get_fft_factor()


class WaveGANLoss(_Loss):
    """
    Loss function that has a spectral component based on the magnitude of FFT and a pixel component based on mean
    absolute error and mean squared error.

    Args:
        dimensions (int): Number of spatial dimensions.
        include_pixel_loss (bool): If the loss includes the pixel component as well
        fft_kwargs (Dict): Dictionary hold all FFT arguments that are to be used when calling torch.fft.fftn.
            Defaults to:  {'s': None, 'dims': tuple(range(1, self.dimensions + 2)), 'norm': 'ortho'}
        size_average (bool): Deprecated (see reduction). By default, the losses are averaged over each loss element
            in the batch. Note that for some losses, there are multiple elements per sample. If the field
            size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is
            False. Default: True
        reduce (bool): Deprecated (see reduction). By default, the losses are averaged or summed over observations
            for each minibatch depending on size_average. When reduce is False, returns a loss per batch element
            instead and ignores size_average. Default: True
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no
            reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in
            the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being
            deprecated, and in the meantime, specifying either of those two args will override reduction.
            Default: 'mean'

    Attributes:
        self.dimensions (int): Number of spatial dimensions.
        self.include_pixel_loss (bool): If the loss includes the pixel component as well
        self.fft_factor (float): Scaling factor of the spectral component of loss
        self.fft_kwargs (Dict): Dictionary containing key words arguments that will be passed to the torch.fft.fftn
            function call.
        self.summaries (Dict): Dictionary containing scalar summaries to be logged in TensorBoard

    References:
        [1] Yamamoto, R., Song, E. and Kim, J.M., 2020, May.
        Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram.
        In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6199-6203). IEEE.
    """

    def __init__(
        self,
        dimensions: int,
        include_pixel_loss: bool = True,
        fft_kwargs: Dict = None,
        size_average: bool = True,
        reduce: bool = True,
        reduction: str = "mean",
    ):
        super(WaveGANLoss, self).__init__(size_average, reduce, reduction)

        self.dimensions = dimensions
        self.include_pixel_loss = include_pixel_loss
        self.fft_factor: float = 1.0
        self.fft_kwargs = (
            {"s": None, "dim": tuple(range(1, self.dimensions + 2)), "norm": "ortho"}
            if fft_kwargs is None
            else fft_kwargs
        )
        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    def forward(
        self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor
    ) -> torch.Tensor:
        # Unpacking elements
        y = y.float()
        y_pred = network_output["reconstruction"][0].float()
        q_losses = network_output["quantization_losses"]

        y_amplitude = self._get_fft_amplitude(y)
        y_pred_amplitude = self._get_fft_amplitude(y_pred)

        # Ref 1 - Sec 3.2 - Eq 4
        l_sc = torch.norm((y_amplitude - y_pred_amplitude), p="fro") / torch.norm(
            y_amplitude, p="fro"
        )
        self.summaries[TBSummaryTypes.SCALAR][
            "Loss-Spectral_Convergence-Reconstruction"
        ] = l_sc

        # Ref 1 - Sec 3.2 - Eq 5
        l_mag = F.l1_loss(torch.log(y_amplitude), torch.log(y_pred_amplitude))
        self.summaries[TBSummaryTypes.SCALAR][
            "Loss-Log_Magnitude-Reconstruction"
        ] = l_mag

        loss = (l_sc + l_mag) * self.fft_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Spectral-Reconstruction"] = loss
        self.summaries[TBSummaryTypes.SCALAR]["Auxiliary-FFT_Factor"] = self.fft_factor

        if self.include_pixel_loss:
            l2_loss = F.mse_loss(y_pred, y, reduction=self.reduction)
            self.summaries[TBSummaryTypes.SCALAR]["Loss-MSE-Reconstruction"] = l2_loss

            loss = loss + l2_loss

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][
                f"Loss-MSE-VQ{idx}_Commitment_Cost"
            ] = q_loss

            loss = loss + q_loss

        return loss

    def _get_fft_amplitude(self, images: torch.Tensor) -> torch.Tensor:
        """
        Manually calculating the amplitude of the fourier transformations representation of the images

        Args:
            images (torch.Tensor): Images that are to undergo fftn

        Returns:
            torch.Tensor: fourier transformation amplitude
        """
        img_fft = fftn(input=images, **self.fft_kwargs)

        amplitude = torch.sqrt(img_fft.real ** 2 + img_fft.imag ** 2)

        return amplitude

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries

    def get_fft_factor(self) -> float:
        return self.fft_factor

    def set_fft_factor(self, fft_factor: float) -> float:
        self.fft_factor = fft_factor

        return self.get_fft_factor()


# It is a torch.nn.Module to be able to move the network used for the perceptual loss to the desired compute devices
class PerceptualLoss(torch.nn.Module):
    """
    Perceptual loss based on the lpips library. The 3D implementation is based on a 2.5D approach where we batchify
    every spatial dimension one after another so we obtain better spatial consistency. There is also a pixel
    component as well.

    Args:
        dimensions (int): Dimensions: number of spatial dimensions.
        include_pixel_loss (bool): If the loss includes the pixel component as well
        is_fake_3d (bool): Whether we use 2.5D approach for a 3D perceptual loss
        drop_ratio (float): How many, as a ratio, slices we drop in the 2.5D approach
        lpips_kwargs (Dict): Dictionary containing key words arguments that will be passed to the LPIPS constructor.
            Defaults to: { 'pretrained': True, 'net': 'alex', 'version': '0.1', 'lpips': True, 'spatial': False,
            'pnet_rand': False,  'pnet_tune': False, 'use_dropout': True, 'model_path': None, 'eval_mode': True,
            'verbose': True}
        lpips_normalize (bool): Whether or not the input needs to be renormalised from [0,1] to [-1,1]

    Attributes:
        self.dimensions (int): Number of spatial dimensions.
        self.include_pixel_loss (bool): If the loss includes the pixel component as well
                self.lpips_kwargs (Dict): Dictionary containing key words arguments that will be passed to the LPIPS
            constructor function call.
        self.fake_3D_views (List[Tuple[Tuple[int,int,int,int,int],Tuple[int,int,int]]]): List of pairs for the 2.5D
            approach. The first element in every tuple is the required permutation for an axis and the second one
            hold the indices of the input image that dictate the shape of the bachified tensor.
        self.keep_ratio (float): Ratio of how many elements of the every 2.5D view we are using to calculate the
            loss. This allows for a memory & iteration speed vs information flow compromise.
        self.lpips_normalize (bool): Whether or not we renormalize from [0,1] to [-1,1]
        self.perceptual_function (Callable): Function that calculates the perceptual loss. For 2D and 2.5D is based
            LPIPS. 3D is not implemented yet.
        self.perceptual_factor (float): Scaling factor of the perceptual component of loss
        self.summaries (Dict): Dictionary containing scalar summaries to be logged in TensorBoard

    References:
        [1] Zhang, R., Isola, P., Efros, A.A., Shechtman, E. and Wang, O., 2018.
        The unreasonable effectiveness of deep features as a perceptual metric.
        In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 586-595).
    """

    def __init__(
        self,
        dimensions: int,
        include_pixel_loss: bool = True,
        is_fake_3d: bool = True,
        drop_ratio: float = 0.0,
        fake_3d_axis: Tuple[int, ...] = (2, 3, 4),
        lpips_kwargs: Dict = None,
        lpips_normalize: bool = True,
    ):
        super(PerceptualLoss, self).__init__()

        if not (dimensions in [2, 3]):
            raise NotImplementedError(
                "Perceptual loss is implemented only in 2D and 3D."
            )

        if dimensions == 3 and is_fake_3d is False:
            raise NotImplementedError("True 3D perceptual loss is not implemented yet.")

        self.dimensions = dimensions
        self.include_pixel_loss = include_pixel_loss
        self.lpips_kwargs = (
            {
                "pretrained": True,
                "net": "alex",
                "version": "0.1",
                "lpips": True,
                "spatial": False,
                "pnet_rand": False,
                "pnet_tune": False,
                "use_dropout": True,
                "model_path": None,
                "eval_mode": True,
                "verbose": False,
            }
            if lpips_kwargs is None
            else lpips_kwargs
        )
        # Here we store the permutations of the 5D tensor where we merge different axis into the batch dimension
        # and use the rest as spatial dimensions, we allow
        self.fake_3D_views = (
            (
                []
                + ([((0, 2, 1, 3, 4), (1, 3, 4))] if 2 in fake_3d_axis else [])
                + ([((0, 3, 1, 2, 4), (1, 2, 4))] if 3 in fake_3d_axis else [])
                + ([((0, 4, 1, 2, 3), (1, 2, 3))] if 4 in fake_3d_axis else [])
            )
            if is_fake_3d
            else None
        )
        # In case of being memory constraint for the 2.5D approach it allows to randomly drop some slices
        self.keep_ratio = 1 - drop_ratio
        self.lpips_normalize = lpips_normalize
        self.perceptual_function = (
            LPIPS(**self.lpips_kwargs) if self.dimensions == 2 or is_fake_3d else None
        )
        self.perceptual_factor = 0.001
        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    def forward(
        self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor
    ) -> torch.Tensor:
        # Unpacking elements
        y = y.float()
        y_pred = network_output["reconstruction"][0].float()
        q_losses = network_output["quantization_losses"]

        if self.dimensions == 3 and self.fake_3D_views:
            loss = torch.zeros(())

            for idx, fake_views in enumerate(self.fake_3D_views):
                p_loss = (
                    self._calculate_fake_3d_loss(
                        y=y,
                        y_pred=y_pred,
                        permute_dims=fake_views[0],
                        view_dims=fake_views[1],
                    )
                    * self.perceptual_factor
                )
                self.summaries[TBSummaryTypes.SCALAR][
                    f"Loss-Perceptual_{idx}-Reconstruction"
                ] = p_loss

                loss = loss + p_loss
        else:
            loss = (
                self.perceptual_function.forward(
                    y, y_pred, normalize=self.lpips_normalize
                )
                * self.perceptual_factor
            )
            self.summaries[TBSummaryTypes.SCALAR][
                "Loss-Perceptual-Reconstruction"
            ] = loss

        self.summaries[TBSummaryTypes.SCALAR][
            "Auxiliary-Perceptual_Factor"
        ] = self.perceptual_factor

        if self.include_pixel_loss:
            l2_loss = F.mse_loss(y_pred, y)
            self.summaries[TBSummaryTypes.SCALAR]["Loss-MSE-Reconstruction"] = l2_loss

            loss = loss + l2_loss

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][
                f"Loss-MSE-VQ{idx}_Commitment_Cost"
            ] = q_loss

            loss = loss + q_loss

        return loss

    def _calculate_fake_3d_loss(
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        permute_dims: Tuple[int, int, int, int, int],
        view_dims: Tuple[int, int, int],
    ):
        """
        Calculating perceptual loss after one spatial axis is batchified according to permute dims and
        we drop random slices as per self.keep_ratio.

        Args:
            y (torch.Tensor): Ground truth images
            y_pred (torch.Tensor): Predictions
            permute_dims (Tuple[int,int,int,int,int]): The order in which the permutation happens where the first
                to newly permuted dimensions are going to become the batch dimension
            view_dims (Tuple[int,int,int]): The channel dimension and two spatial dimensions that are being kept
                after the permutation.

        Returns:
            torch.Tensor: perceptual loss value on the given axis
        """
        # Reshaping the ground truth and prediction to be considered 2D
        y_slices = (
            y.permute(*permute_dims)
            .contiguous()
            .view(
                -1, y.shape[view_dims[0]], y.shape[view_dims[1]], y.shape[view_dims[2]]
            )
        )

        y_pred_slices = (
            y_pred.permute(*permute_dims)
            .contiguous()
            .view(
                -1,
                y_pred.shape[view_dims[0]],
                y_pred.shape[view_dims[1]],
                y_pred.shape[view_dims[2]],
            )
        )

        # Subsampling in case we are memory constrained
        indices = torch.randperm(y_pred_slices.shape[0], device=y_pred_slices.device)[
            : int(y_pred_slices.shape[0] * self.keep_ratio)
        ]

        y_pred_slices = y_pred_slices[indices]
        y_slices = y_slices[indices]

        # Calculating the 2.5D perceptual loss
        p_loss = torch.mean(
            self.perceptual_function.forward(
                y_slices, y_pred_slices, normalize=self.lpips_normalize
            )
        )

        return p_loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries

    def get_perceptual_factor(self) -> float:
        return self.perceptual_factor

    def set_perceptual_factor(self, perceptual_factor: float) -> float:
        self.perceptual_factor = perceptual_factor

        return self.get_perceptual_factor()


# It is a torch.nn.Module to be able to move the network used for the perceptual loss to the desired compute devices
class JukeboxPerceptualLoss(torch.nn.Module):
    """
    Loss made of three components:
        * Perceptual [1] - based on the lpips library. The 3D implementation is based on a 2.5D approach where we
            batchify every spatial dimension one after another so we obtain better spatial consistency.
        * Spectral [2] - based on the magnitude of FFT
        * Pixel - MSE

    Args:
        dimensions (int): Dimensions: number of spatial dimensions.
        include_pixel_loss (bool): If the loss includes the pixel component as well
        is_fake_3d (bool): Whether we use 2.5D approach for a 3D perceptual loss
        drop_ratio (float): How many, as a ratio, slices we drop in the 2.5D approach
        lpips_kwargs (Dict): Dictionary containing key words arguments that will be passed to the LPIPS constructor.
            Defaults to: { 'pretrained': True, 'net': 'alex', 'version': '0.1', 'lpips': True, 'spatial': False,
            'pnet_rand': False,  'pnet_tune': False, 'use_dropout': True, 'model_path': None, 'eval_mode': True,
            'verbose': True}
        lpips_normalize (bool): Whether or not the input needs to be renormalised from [0,1] to [-1,1]
        fft_kwargs (Dict): Dictionary hold all FFT arguments that are to be used when calling torch.fft.fftn.
            Defaults to:  {'s': None, 'dims': tuple(range(1, self.dimensions + 2)), 'norm': 'ortho'}

    Attributes:
        self.dimensions (int): Number of spatial dimensions.
        self.include_pixel_loss (bool): If the loss includes the pixel component as well
        self.lpips_kwargs (Dict): Dictionary containing key words arguments that will be passed to the LPIPS
            constructor function call.
        self.fake_3D_views (List[Tuple[Tuple[int,int,int,int,int],Tuple[int,int,int]]]): List of pairs for the 2.5D
            approach. The first element in every tuple is the required permutation for an axis and the second one
            hold the indices of the input image that dictate the shape of the bachified tensor.
        self.keep_ratio (float): Ratio of how many elements of the every 2.5D view we are using to calculate the
            loss. This allows for a memory & iteration speed vs information flow compromise.
        self.lpips_normalize (bool): Whether or not we renormalize from [0,1] to [-1,1]
        self.perceptual_function (Callable): Function that calculates the perceptual loss. For 2D and 2.5D is based
            LPIPS. 3D is not implemented yet.
        self.perceptual_factor (float): Scaling factor of the perceptual component of loss
        self.fft_factor (float): Scaling factor of the spectral component of loss
        self.fft_kwargs (Dict): Dictionary containing key words arguments that will be passed to the torch.fft.fftn
            function call.
        self.summaries (Dict): Dictionary containing scalar summaries to be logged in TensorBoard

    References:
        [1] Zhang, R., Isola, P., Efros, A.A., Shechtman, E. and Wang, O., 2018.
        The unreasonable effectiveness of deep features as a perceptual metric.
        In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 586-595).

        [2] Dhariwal, P., Jun, H., Payne, C., Kim, J.W., Radford, A. and Sutskever, I., 2020.
        Jukebox: A generative model for music.
        arXiv preprint arXiv:2005.00341.
    """

    def __init__(
        self,
        dimensions: int,
        include_pixel_loss: bool = True,
        is_fake_3d: bool = True,
        drop_ratio: float = 0.0,
        fake_3d_axis: Tuple[int, ...] = (2, 3, 4),
        lpips_kwargs: Dict = None,
        lpips_normalize: bool = True,
        fft_kwargs: Dict = None,
    ):
        super(JukeboxPerceptualLoss, self).__init__()

        if not (dimensions in [2, 3]):
            raise NotImplementedError(
                "Perceptual loss is implemented only in 2D and 3D."
            )

        if dimensions == 3 and is_fake_3d is False:
            raise NotImplementedError("True 3D perceptual loss is not implemented yet.")

        self.dimensions = dimensions

        self.include_pixel_loss = include_pixel_loss

        self.lpips_kwargs = (
            {
                "pretrained": True,
                "net": "alex",
                "version": "0.1",
                "lpips": True,
                "spatial": False,
                "pnet_rand": False,
                "pnet_tune": False,
                "use_dropout": True,
                "model_path": None,
                "eval_mode": True,
                "verbose": False,
            }
            if lpips_kwargs is None
            else lpips_kwargs
        )
        # Here we store the permutations of the 5D tensor where we merge different axis into the batch dimension
        # and use the rest as spatial dimensions, we allow
        self.fake_3D_views = (
            (
                []
                + ([((0, 2, 1, 3, 4), (1, 3, 4))] if 2 in fake_3d_axis else [])
                + ([((0, 3, 1, 2, 4), (1, 2, 4))] if 3 in fake_3d_axis else [])
                + ([((0, 4, 1, 2, 3), (1, 2, 3))] if 4 in fake_3d_axis else [])
            )
            if is_fake_3d
            else None
        )
        # In case of being memory constraint for the 2.5D approach it allows to randomly drop some slices
        self.keep_ratio = 1 - drop_ratio
        self.lpips_normalize = lpips_normalize
        self.perceptual_function = (
            LPIPS(**self.lpips_kwargs) if self.dimensions == 2 or is_fake_3d else None
        )
        self.perceptual_factor = 0.001

        self.fft_factor: float = 1.0
        self.fft_kwargs = (
            {"s": None, "dim": tuple(range(1, self.dimensions + 2)), "norm": "ortho"}
            if fft_kwargs is None
            else fft_kwargs
        )

        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    def forward(
        self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor
    ) -> torch.Tensor:
        # Unpacking elements
        y = y.float()
        y_pred = network_output["reconstruction"][0].float()
        q_losses = network_output["quantization_losses"]

        # ==== Spectral Component ====
        y_amplitude = self._get_fft_amplitude(y)
        y_pred_amplitude = self._get_fft_amplitude(y_pred)

        # Ref 2 - Section 3.3 - L_spec
        loss = F.mse_loss(y_pred_amplitude, y_amplitude) * self.fft_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Spectral-Reconstruction"] = loss
        self.summaries[TBSummaryTypes.SCALAR]["Auxiliary-FFT_Factor"] = self.fft_factor

        # ==== Perceptual Component ====
        if self.dimensions == 3 and self.fake_3D_views:
            for idx, fake_views in enumerate(self.fake_3D_views):
                p_loss = (
                    self._calculate_fake_3d_loss(
                        y=y,
                        y_pred=y_pred,
                        permute_dims=fake_views[0],
                        view_dims=fake_views[1],
                    )
                    * self.perceptual_factor
                )
                self.summaries[TBSummaryTypes.SCALAR][
                    f"Loss-Perceptual_{idx}-Reconstruction"
                ] = p_loss

                loss = loss + p_loss
        else:
            p_loss = (
                self.perceptual_function.forward(
                    y, y_pred, normalize=self.lpips_normalize
                )
                * self.perceptual_factor
            )
            self.summaries[TBSummaryTypes.SCALAR][
                "Loss-Perceptual-Reconstruction"
            ] = p_loss

            loss = loss + p_loss

        self.summaries[TBSummaryTypes.SCALAR][
            "Auxiliary-Perceptual_Factor"
        ] = self.perceptual_factor

        if self.include_pixel_loss:
            l2_loss = F.mse_loss(y_pred, y)
            self.summaries[TBSummaryTypes.SCALAR]["Loss-MSE-Reconstruction"] = l2_loss

            loss = loss + l2_loss

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][
                f"Loss-MSE-VQ{idx}_Commitment_Cost"
            ] = q_loss

            loss = loss + q_loss

        return loss

    def _calculate_fake_3d_loss(
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        permute_dims: Tuple[int, int, int, int, int],
        view_dims: Tuple[int, int, int],
    ):
        """
        Calculating perceptual loss after one spatial axis is batchified according to permute dims and
        we drop random slices as per self.keep_ratio.

        Args:
            y (torch.Tensor): Ground truth images
            y_pred (torch.Tensor): Predictions
            permute_dims (Tuple[int,int,int,int,int]): The order in which the permutation happens where the first
                to newly permuted dimensions are going to become the batch dimension
            view_dims (Tuple[int,int,int]): The channel dimension and two spatial dimensions that are being kept
                after the permutation.

        Returns:
            torch.Tensor: perceptual loss value on the given axis
        """
        # Reshaping the ground truth and prediction to be considered 2D
        y_slices = (
            y.permute(*permute_dims)
            .contiguous()
            .view(
                -1, y.shape[view_dims[0]], y.shape[view_dims[1]], y.shape[view_dims[2]]
            )
        )

        y_pred_slices = (
            y_pred.permute(*permute_dims)
            .contiguous()
            .view(
                -1,
                y_pred.shape[view_dims[0]],
                y_pred.shape[view_dims[1]],
                y_pred.shape[view_dims[2]],
            )
        )

        # Subsampling in case we are memory constrained
        indices = torch.randperm(y_pred_slices.shape[0], device=y_pred_slices.device)[
            : int(y_pred_slices.shape[0] * self.keep_ratio)
        ]

        y_pred_slices = y_pred_slices[indices]
        y_slices = y_slices[indices]

        # Calculating the 2.5D perceptual loss
        p_loss = torch.mean(
            self.perceptual_function.forward(
                y_slices, y_pred_slices, normalize=self.lpips_normalize
            )
        )

        return p_loss

    def _get_fft_amplitude(self, images: torch.Tensor) -> torch.Tensor:
        """
        Manually calculating the amplitude of the fourier transformations representation of the images

        Args:
            images (torch.Tensor): Images that are to undergo fftn

        Returns:
            torch.Tensor: fourier transformation amplitude
        """
        img_fft = fftn(input=images, **self.fft_kwargs)

        amplitude = torch.sqrt(img_fft.real ** 2 + img_fft.imag ** 2)

        return amplitude

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries

    def get_perceptual_factor(self) -> float:
        return self.perceptual_factor

    def set_perceptual_factor(self, perceptual_factor: float) -> float:
        self.perceptual_factor = perceptual_factor

        return self.get_perceptual_factor()

    def get_fft_factor(self) -> float:
        return self.fft_factor

    def set_fft_factor(self, fft_factor: float) -> float:
        self.fft_factor = fft_factor

        return self.get_fft_factor()


# It is a torch.nn.Module to be able to move the network used for the perceptual loss to the desired compute devices
class HartleyPerceptualLoss(torch.nn.Module):
    """
    Loss made of three components:
        * Perceptual [1] - based on the lpips library. The 3D implementation is based on a 2.5D approach where we
            batchify every spatial dimension one after another so we obtain better spatial consistency.
        * Spectral [2] - based on the magnitude of Hartley transformation
        * Pixel - MSE

    Args:
        dimensions (int): Dimensions: number of spatial dimensions.
        include_pixel_loss (bool): If the loss includes the pixel component as well
        is_fake_3d (bool): Whether we use 2.5D approach for a 3D perceptual loss
        drop_ratio (float): How many, as a ratio, slices we drop in the 2.5D approach
        lpips_kwargs (Dict): Dictionary containing key words arguments that will be passed to the LPIPS constructor.
            Defaults to: { 'pretrained': True, 'net': 'alex', 'version': '0.1', 'lpips': True, 'spatial': False,
            'pnet_rand': False,  'pnet_tune': False, 'use_dropout': True, 'model_path': None, 'eval_mode': True,
            'verbose': True}
        lpips_normalize (bool): Whether or not the input needs to be renormalised from [0,1] to [-1,1]
        fft_kwargs (Dict): Dictionary hold all FFT arguments that are to be used when calling torch.fft.fftn.
            Defaults to:  {'s': None, 'dims': tuple(range(1, self.dimensions + 2)), 'norm': 'ortho'}
        prioritise_high_frequency (bool): Whether to increase the importance of the high frequencies based on the
            equation 11 from [1].

    Attributes:
        self.dimensions (int): Number of spatial dimensions.
        self.include_pixel_loss (bool): If the loss includes the pixel component as well
                self.lpips_kwargs (Dict): Dictionary containing key words arguments that will be passed to the LPIPS
            constructor function call.
        self.fake_3D_views (List[Tuple[Tuple[int,int,int,int,int],Tuple[int,int,int]]]): List of pairs for the 2.5D
            approach. The first element in every tuple is the required permutation for an axis and the second one
            hold the indices of the input image that dictate the shape of the bachified tensor.
        self.keep_ratio (float): Ratio of how many elements of the every 2.5D view we are using to calculate the
            loss. This allows for a memory & iteration speed vs information flow compromise.
        self.lpips_normalize (bool): Whether or not we renormalize from [0,1] to [-1,1]
        self.perceptual_function (Callable): Function that calculates the perceptual loss. For 2D and 2.5D is based
            LPIPS. 3D is not implemented yet.
        self.perceptual_factor (float): Scaling factor of the perceptual component of loss
                    self.fht_factor (float): Scaling factor of the spectral component of loss
        self.fht_weight (torch.Tensor): Weight tensor that scales the hartley transformation in favor of high
            frequencies.
        self.fft_kwargs (Dict): Dictionary containing key words arguments that will be passed to the torch.fft.fftn
            function call.
        self.prioritise_high_frequency (bool): Whether to increase the importance of the high frequencies
        self.summaries (Dict): Dictionary containing scalar summaries to be logged in TensorBoard

    References:
        [1] Zhang, R., Isola, P., Efros, A.A., Shechtman, E. and Wang, O., 2018.
        The unreasonable effectiveness of deep features as a perceptual metric.
        In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 586-595).

        [2] Xue, S., Qiu, W., Liu, F. and Jin, X., 2020.
        Faster image super-resolution by improved frequency-domain neural networks.
        Signal, Image and Video Processing, 14(2), pp.257-265.
    """

    def __init__(
        self,
        dimensions: int,
        include_pixel_loss: bool = True,
        is_fake_3d: bool = True,
        drop_ratio: float = 0.0,
        fake_3d_axis: Tuple[int, ...] = (2, 3, 4),
        lpips_kwargs: Dict = None,
        lpips_normalize: bool = True,
        fft_kwargs: Dict = None,
        prioritise_high_frequency: bool = True,
    ):
        super(HartleyPerceptualLoss, self).__init__()

        if not (dimensions in [2, 3]):
            raise NotImplementedError(
                "Perceptual loss is implemented only in 2D and 3D."
            )

        if dimensions == 3 and is_fake_3d is False:
            raise NotImplementedError("True 3D perceptual loss is not implemented yet.")

        self.dimensions = dimensions

        self.include_pixel_loss = include_pixel_loss

        self.lpips_kwargs = (
            {
                "pretrained": True,
                "net": "alex",
                "version": "0.1",
                "lpips": True,
                "spatial": False,
                "pnet_rand": False,
                "pnet_tune": False,
                "use_dropout": True,
                "model_path": None,
                "eval_mode": True,
                "verbose": False,
            }
            if lpips_kwargs is None
            else lpips_kwargs
        )
        # Here we store the permutations of the 5D tensor where we merge different axis into the batch dimension
        # and use the rest as spatial dimensions, we allow
        self.fake_3D_views = (
            (
                []
                + ([((0, 2, 1, 3, 4), (1, 3, 4))] if 2 in fake_3d_axis else [])
                + ([((0, 3, 1, 2, 4), (1, 2, 4))] if 3 in fake_3d_axis else [])
                + ([((0, 4, 1, 2, 3), (1, 2, 3))] if 4 in fake_3d_axis else [])
            )
            if is_fake_3d
            else None
        )
        # In case of being memory constraint for the 2.5D approach it allows to randomly drop some slices
        self.keep_ratio = 1 - drop_ratio
        self.lpips_normalize = lpips_normalize
        self.perceptual_function = (
            LPIPS(**self.lpips_kwargs) if self.dimensions == 2 or is_fake_3d else None
        )
        self.perceptual_factor = 0.001

        self.fft_kwargs = (
            {"s": None, "dim": tuple(range(1, self.dimensions + 2)), "norm": "ortho"}
            if fft_kwargs is None
            else fft_kwargs
        )
        self.prioritise_high_frequency = prioritise_high_frequency
        self.fht_weight: torch.Tensor = torch.rand(1)
        self.fht_factor: float = 1.0

        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    def forward(
        self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor
    ) -> torch.Tensor:
        # Unpacking elements
        y = y.float()
        y_pred = network_output["reconstruction"][0].float()
        q_losses = network_output["quantization_losses"]

        # ==== Spectral Component ====
        # Calculating the Hartley representation
        y_fht = self._fhtn(images=y)
        y_pred_fht = self._fhtn(images=y_pred)

        if self.prioritise_high_frequency:
            weight = self._calculate_frequency_weight(images=y)
            y_fht = y_fht * weight
            y_pred_fht = y_pred_fht * weight

        # Ref 2 - Sec 3.4 - Eq 13
        # Deviation from a proper implementation due to scale imbalances
        # the Hartley loss would be 15+ orders of magnitude higher at the beginning
        # loss = 0.5 * torch.norm((y_pred_fht - y_fht) * weight, p="fro") ** 2
        loss = 0.5 * F.mse_loss(y_pred_fht, y_fht)
        loss = loss * self.fht_factor
        self.summaries[TBSummaryTypes.SCALAR][
            "Auxiliary-Hartley_Factor"
        ] = self.fht_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Hartley-Reconstruction"] = loss
        # ==== Perceptual Component ====
        if self.dimensions == 3 and self.fake_3D_views:
            for idx, fake_views in enumerate(self.fake_3D_views):
                p_loss = self._calculate_fake_3d_loss(
                    y=y,
                    y_pred=y_pred,
                    permute_dims=fake_views[0],
                    view_dims=fake_views[1],
                )

                p_loss = p_loss * self.perceptual_factor

                self.summaries[TBSummaryTypes.SCALAR][
                    f"Loss-Perceptual_{idx}-Reconstruction"
                ] = p_loss

                loss = loss + p_loss
        else:
            p_loss = (
                self.perceptual_function.forward(
                    y, y_pred, normalize=self.lpips_normalize
                )
                * self.perceptual_factor
            )
            self.summaries[TBSummaryTypes.SCALAR][
                "Loss-Perceptual-Reconstruction"
            ] = p_loss

            loss = loss + p_loss

        self.summaries[TBSummaryTypes.SCALAR][
            "Auxiliary-Perceptual_Factor"
        ] = self.perceptual_factor

        if self.include_pixel_loss:
            l2_loss = F.mse_loss(y_pred, y)
            self.summaries[TBSummaryTypes.SCALAR]["Loss-MSE-Reconstruction"] = l2_loss

            loss = loss + l2_loss

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][
                f"Loss-MSE-VQ{idx}_Commitment_Cost"
            ] = q_loss

            loss = loss + q_loss

        return loss

    def _calculate_fake_3d_loss(
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        permute_dims: Tuple[int, int, int, int, int],
        view_dims: Tuple[int, int, int],
    ):
        """
        Calculating perceptual loss after one spatial axis is batchified according to permute dims and
        we drop random slices as per self.keep_ratio.

        Args:
            y (torch.Tensor): Ground truth images
            y_pred (torch.Tensor): Predictions
            permute_dims (Tuple[int,int,int,int,int]): The order in which the permutation happens where the first
                to newly permuted dimensions are going to become the batch dimension
            view_dims (Tuple[int,int,int]): The channel dimension and two spatial dimensions that are being kept
                after the permutation.

        Returns:
            torch.Tensor: perceptual loss value on the given axis
        """
        # Reshaping the ground truth and prediction to be considered 2D
        y_slices = (
            y.permute(*permute_dims)
            .contiguous()
            .view(
                -1, y.shape[view_dims[0]], y.shape[view_dims[1]], y.shape[view_dims[2]]
            )
        )

        y_pred_slices = (
            y_pred.permute(*permute_dims)
            .contiguous()
            .view(
                -1,
                y_pred.shape[view_dims[0]],
                y_pred.shape[view_dims[1]],
                y_pred.shape[view_dims[2]],
            )
        )

        # Subsampling in case we are memory constrained
        indices = torch.randperm(y_pred_slices.shape[0], device=y_pred_slices.device)[
            : int(y_pred_slices.shape[0] * self.keep_ratio)
        ]

        y_pred_slices = y_pred_slices[indices]
        y_slices = y_slices[indices]

        # Calculating the 2.5D perceptual loss
        p_loss = torch.mean(
            self.perceptual_function.forward(
                y_slices, y_pred_slices, normalize=self.lpips_normalize
            )
        )

        return p_loss

    def _fhtn(self, images: torch.Tensor) -> torch.Tensor:
        """
        Calculates the hartley transformation of the given images based on the fourier transformation relation.

        Args:
            images (torch.Tensor): Images that are to undergo hartley transformation

        Returns:
            torch.Tensor: hartley transformation representation
        """
        img_fft = fftn(input=images, **self.fft_kwargs)
        ht = img_fft.real - img_fft.imag
        return ht

    def _calculate_frequency_weight(self, images: torch.Tensor) -> torch.Tensor:
        """
        Calculating a weight matrix that will be applied to scale the hartley representations in favour of
        high frequencies.

        Based on equation equation 11 from [1].

        Args:
            images (torch.Tensor): Images that are dictating the shape

        Returns:
            torch.Tensor: High frequency focused weighting

        Raises:
            ValueError: If the tensors fed for shape reference is not a 4D or 5D tensor
        """
        if images.shape != self.fht_weight.shape:
            # We have 2D images
            if len(images.shape) == 4:
                mx = images.shape[2]
                my = images.shape[3]

                def hartley_weight(x, y) -> float:
                    return (abs(mx / 2 - x) / (mx / 2)) ** 2 + (
                        abs(my / 2 - y) / (my / 2)
                    ) ** 2

            # We have 3D images
            elif len(images.shape) == 5:
                mx = images.shape[2]
                my = images.shape[3]
                mz = images.shape[4]

                def hartley_weight(x, y, z) -> float:
                    return (
                        (abs(mx / 2 - x) / (mx / 2)) ** 2
                        + (abs(my / 2 - y) / (my / 2)) ** 2
                        + (abs(mz / 2 - z) / (mz / 2)) ** 2
                    )

            else:
                raise ValueError(
                    "HartleyLoss is implemented only for 2D and 3D images."
                )

            weight = torch.from_numpy(
                fromfunction(hartley_weight, shape=images.shape[2:])
            ).to(images.device)

            weight = torch.exp(weight)

            weight = weight - torch.min(weight)
            weight = weight / torch.max(weight)
            weight = weight + 0.0001

            self.fht_weight = weight

        return self.fht_weight

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries

    def get_perceptual_factor(self) -> float:
        return self.perceptual_factor

    def set_perceptual_factor(self, perceptual_factor: float) -> float:
        self.perceptual_factor = perceptual_factor

        return self.get_perceptual_factor()

    def get_fht_factor(self) -> float:
        return self.fht_factor

    def set_fht_factor(self, fht_factor: float) -> float:
        self.fht_factor = fht_factor

        return self.get_fht_factor()


class BaselineLoss(torch.nn.Module):
    def __init__(self):
        super(BaselineLoss, self).__init__()

        self.pixel_factor = 1.0

        self.perceptual_factor = 0.002
        self.n_slices = 512
        self.perceptual_function = LPIPS(net="squeeze")

        self.fft_factor = 1.0

        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    def forward(
        self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor
    ) -> torch.Tensor:
        # Unpacking elements
        x = y.float()
        y = network_output["reconstruction"][0].float()
        q_losses = network_output["quantization_losses"]

        loss = (
            self._calculate_pixel_loss(x, y)
            + self._calculate_frequency_loss(x, y)
            + self._calculate_perceptual_loss(x, y)
        )

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][
                f"Loss-MSE-VQ{idx}_Commitment_Cost"
            ] = q_loss

            loss = loss + q_loss

        return loss

    def _calculate_frequency_loss(self, x, y) -> torch.Tensor:
        def fft_abs(t):
            img_torch = (t + 1.0) / 2.0
            fft = fftn(img_torch, norm="ortho")
            return torch.abs(fft)

        loss = F.mse_loss(fft_abs(x), fft_abs(y))
        loss = loss * self.fft_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Jukebox-Reconstruction"] = loss

        return loss

    def _calculate_pixel_loss(self, x, y) -> torch.Tensor:
        loss = F.l1_loss(x, y)
        loss = loss * self.pixel_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-MAE-Reconstruction"] = loss

        return loss

    def _calculate_perceptual_loss(self, x, y) -> torch.Tensor:
        # Sagital
        x_2d = (
            x.permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(-1, x.shape[1], x.shape[3], x.shape[4])
        )
        y_2d = (
            y.permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(-1, y.shape[1], y.shape[3], y.shape[4])
        )
        indices = torch.randperm(x_2d.size(0))[: self.n_slices]
        selected_x_2d = x_2d[indices]
        selected_y_2d = y_2d[indices]

        p_loss_sagital = torch.mean(
            self.perceptual_function.forward(
                selected_x_2d.float(), selected_y_2d.float()
            )
        )
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual_Sagittal-Reconstruction"] = p_loss_sagital

        # Axial
        x_2d = (
            x.permute(0, 4, 1, 2, 3)
            .contiguous()
            .view(-1, x.shape[1], x.shape[2], x.shape[3])
        )
        y_2d = (
            y.permute(0, 4, 1, 2, 3)
            .contiguous()
            .view(-1, y.shape[1], y.shape[2], y.shape[3])
        )
        indices = torch.randperm(x_2d.size(0))[: self.n_slices]
        selected_x_2d = x_2d[indices]
        selected_y_2d = y_2d[indices]

        p_loss_axial = torch.mean(
            self.perceptual_function.forward(
                selected_x_2d.float(), selected_y_2d.float()
            )
        )
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual_Axial-Reconstruction"] = p_loss_axial

        # Coronal
        x_2d = (
            x.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(-1, x.shape[1], x.shape[2], x.shape[4])
        )
        y_2d = (
            y.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(-1, y.shape[1], y.shape[2], y.shape[4])
        )
        indices = torch.randperm(x_2d.size(0))[: self.n_slices]
        selected_x_2d = x_2d[indices]
        selected_y_2d = y_2d[indices]

        p_loss_coronal = torch.mean(
            self.perceptual_function.forward(
                selected_x_2d.float(), selected_y_2d.float()
            )
        )
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual_Coronal-Reconstruction"] = p_loss_coronal

        loss = p_loss_sagital + p_loss_axial + p_loss_coronal
        loss = loss * self.perceptual_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual-Reconstruction"] = loss

        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries
