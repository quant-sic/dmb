import json
from typing import Any

import numpy as np


class WormInputParametersDecoder(json.JSONDecoder):
    """Decoder for the input parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, object_hook=self._object_hook, **kwargs)

    @staticmethod
    def _object_hook(obj: Any) -> Any:
        for key, value in obj.items():
            if key in ["mu", "t_hop", "U_on", "V_nn"]:
                obj[key] = np.array(value)

        return obj
