from typing import Collection, Dict, Hashable, Mapping, Union

import numpy as np
from monai.config import KeysCollection
from monai.transforms.compose import Compose
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.utils.misc import ensure_tuple


class TraceTransformsd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        composed_transforms: Union[Collection[Compose], Compose],
        trace_key_postfix: str = "trace_dict",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.composed_transforms = ensure_tuple(composed_transforms)
        self.trace_key_postfix = trace_key_postfix

        assert len(self.keys) == len(self.composed_transforms), (
            f"The keys and composed_transforms must have the same length but got keys of length {len(self.keys)} and "
            f"composed_transforms of length {len(self.composed_transforms)}"
        )

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:

        d = dict(data)

        for k, ct in zip(self.keys, self.composed_transforms):
            trace = {
                str(transform.__class__).split(".")[-1]: transform._do_transform
                if isinstance(transform, RandomizableTransform)
                else True
                for transform in ct.transforms
            }

            d[f"{k}_{self.trace_key_postfix}"] = trace

        return d
