from enum import Enum
from typing import Union, Tuple, List

import numpy as np
import torch

from gilbert.gilbert2d import gilbert2d
from gilbert.gilbert3d import gilbert3d


class OrderingType(Enum):
    RASTER_SCAN = "raster_scan"
    S_CURVE = "s_curve"
    RANDOM = "random"
    HILBERT = "hilbert_curve"


class OrderingTransformations(Enum):
    ROTATE_90 = "rotate_90"
    TRANSPOSE = "transpose"
    REFLECT = "reflect"


class Ordering:
    def __init__(
        self,
        ordering_type: str,
        spatial_dims: int,
        dimensions: Union[Tuple[int, int, int], Tuple[int, int, int, int]],
        reflected_spatial_dims: Union[Tuple[bool, bool], Tuple[bool, bool, bool]],
        transpositions_axes: Union[
            Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int], ...]
        ],
        rot90_axes: Union[
            Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int], ...]
        ],
        transformation_order: Tuple[str, ...] = (
            OrderingTransformations.TRANSPOSE.value,
            OrderingTransformations.ROTATE_90.value,
            OrderingTransformations.REFLECT.value,
        ),
    ):
        super().__init__()
        self.ordering_type = ordering_type

        assert self.ordering_type in [
            e.value for e in OrderingType
        ], f"ordering_type must be one of the following {[e.value for e in OrderingType]}, but got {self.ordering_type}."

        self.spatial_dims = spatial_dims
        self.dimensions = dimensions

        assert len(dimensions) == self.spatial_dims + 1, f"Dimensions must have length {self.spatial_dims + 1}."

        self.reflected_spatial_dims = reflected_spatial_dims
        self.transpositions_axes = transpositions_axes
        self.rot90_axes = rot90_axes
        if len(set(transformation_order)) != len(transformation_order):
            raise ValueError(
                f"No duplicates are allowed. Received {transformation_order}."
            )

        for transformation in transformation_order:
            if transformation not in [t.value for t in OrderingTransformations]:
                raise ValueError(
                    f"Valid transformations are {[t.value for t in OrderingTransformations]} but received {transformation}."
                )
        self.transformation_order = transformation_order

        self.template = self._create_template()
        self._sequence_ordering = self._create_ordering()
        self._revert_sequence_ordering = np.argsort(self._sequence_ordering)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x[self._sequence_ordering]

        return x

    def get_sequence_ordering(self) -> np.ndarray:
        return self._sequence_ordering

    def get_revert_sequence_ordering(self) -> np.ndarray:
        return self._revert_sequence_ordering

    def _create_ordering(self):
        self.template = self._transform_template()
        order = self._order_template(template=self.template)

        return order

    def _create_template(self) -> np.ndarray:
        spatial_dimensions = self.dimensions[1:]
        template = np.arange(np.prod(spatial_dimensions)).reshape(*spatial_dimensions)

        return template

    def _transform_template(self)->np.ndarray:
        for transformation in self.transformation_order:
            if transformation == OrderingTransformations.TRANSPOSE.value:
                self.template = self._transpose_template(template=self.template)
            elif transformation == OrderingTransformations.ROTATE_90.value:
                self.template = self._rot90_template(template=self.template)
            elif transformation == OrderingTransformations.REFLECT.value:
                self.template = self._flip_template(template=self.template)

        return self.template
    def _transpose_template(self, template: np.ndarray) -> np.ndarray:
        for axes in self.transpositions_axes:
            template = np.transpose(template, axes=axes)

        return template

    def _flip_template(self, template: np.ndarray) -> np.ndarray:
        for axis, to_reflect in enumerate(self.reflected_spatial_dims):
            template = np.flip(template, axis=axis) if to_reflect else template

        return template

    def _rot90_template(self, template: np.ndarray) -> np.ndarray:
        for axes in self.rot90_axes:
            template = np.rot90(template, axes=axes)

        return template

    def _order_template(self, template: np.ndarray) -> np.ndarray:
        depths = None
        if self.spatial_dims == 2:
            rows, columns = template.shape[0], template.shape[1]
        else:
            rows, columns, depths = (
                template.shape[0],
                template.shape[1],
                template.shape[2],
            )

        sequence = eval(f"self.{self.ordering_type}_idx")(rows, columns, depths)

        ordering = np.array([template[tuple(e)] for e in sequence])

        return ordering

    @staticmethod
    def raster_scan_idx(rows: int, cols: int, depths: int = None) -> np.ndarray:
        idx = []

        for r in range(rows):
            for c in range(cols):
                if depths:
                    for d in range(depths):
                        idx.append((r, c, d))
                else:
                    idx.append((r, c))

        idx = np.array(idx)

        return idx

    @staticmethod
    def s_curve_idx(rows: int, cols: int, depths: int = None) -> np.ndarray:
        idx = []

        for r in range(rows):
            col_idx = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            for c in col_idx:
                if depths:
                    depth_idx = (
                        range(depths) if c % 2 == 0 else range(depths - 1, -1, -1)
                    )

                    for d in depth_idx:
                        idx.append((r, c, d))
                else:
                    idx.append((r, c))

        idx = np.array(idx)

        return idx

    @staticmethod
    def random_idx(rows: int, cols: int, depths: int = None) -> np.ndarray:
        idx = []

        for r in range(rows):
            for c in range(cols):
                if depths:
                    for d in range(depths):
                        idx.append((r, c, d))
                else:
                    idx.append((r, c))

        idx = np.array(idx)
        np.random.shuffle(idx)

        return idx

    @staticmethod
    def hilbert_curve_idx(rows: int, cols: int, depths: int = None) -> np.ndarray:
        t = list(gilbert3d(rows, cols, depths) if depths else gilbert2d(rows, cols))
        idx = np.array(t)

        return idx
