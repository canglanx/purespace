"""
Wireframe rendering for three-dimensional geometry.

Approach:
To extract edges where faces intersect, render adjacent faces with
distinct colors and isolate the intersection lines via post-processing.
The final result may require compositing multiple intermediate render passes.

Note:
Blender's built-in "freestyle" can achieve similar effects, but it often
produces rough lines and may incorrectly render edges between coplanar faces.
"""

import cv2
import logging
import numpy as np
import os
import tempfile

from datetime import datetime
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ValidationInfo,
)
from typing import List, Literal, Self

from purespace.renderer.engine import (
    build_case,
    clear_case,
    init_scene,
    render_case_by_views,
)
from purespace.renderer.post_process import pipeline
from purespace.renderer.utils import (
    get_heights_z,
    get_reverse_params,
    remap_view_name,
    wrap_view,
)


logger = logging.getLogger(__name__)


ViewType = Literal["iso", "top", "right", "front", "iso-left", "iso-right", "iso-reverse"]


class RenderParams(BaseModel):
    is_strict: bool = Field(..., exclude=True)
    heights: List[int]
    corners: List[List[List[int]]]
    view: ViewType
    size: int = Field(..., ge=128, le=1024)

    @field_validator("heights")
    @classmethod
    def validate_heights(
        cls, values: List[int], info: ValidationInfo
    ) -> List[int]:
        # --- Common constraints ---
        if not values:
            raise ValueError("'heights' cannot be empty")
        for h in values:
            if h <= 0:
                raise ValueError(f"'heights' must be greater than 0, got {h}")
        if not info.data.get("is_strict", True):
            return values
        # --- Strict constraints ---
        for h in values:
            if not (4 <= h <= 20):
                raise ValueError(f"'heights' must be between 4 and 20, got {h}")
        sum_values = sum(values)
        if sum_values != 20:
            raise ValueError(f"sum of 'heights' must be exactly 20, got {sum_values}")
        return values

    @field_validator("corners")
    @classmethod
    def validate_corners(
        cls, values: List[List[List[int]]], info: ValidationInfo
    ) -> List[List[List[int]]]:
        # --- Common constraints ---
        if not values:
            raise ValueError("'corners' cannot be empty")
        if not info.data.get("is_strict", True):
            return values
        # --- Strict constraints ---
        for level in values:
            for pt in level:
                if not (4 <= pt[0] <= 20) or not (4 <= pt[1] <= 20):
                    raise ValueError(
                        f"'corners' x and y must be between 4 and 20, got {pt}"
                    )
            for pt_idx in range(len(level) - 1):
                x_curr = level[pt_idx][0]
                y_curr = level[pt_idx][1]
                x_next = level[pt_idx + 1][0]
                y_next = level[pt_idx + 1][1]
                if x_next - x_curr < 4:
                    raise ValueError(
                        f"'corners' x must increase by >= 4, got {x_curr} -> {x_next}"
                    )
                if y_curr - y_next < 4:
                    raise ValueError(
                        f"'corners' y must decrease by >= 4, got {y_curr} -> {y_next}"
                    )
        return values

    @model_validator(mode="after")
    def validate_num_levels(self) -> Self:
        if len(self.heights) != len(self.corners):
            raise ValueError("'heights' and 'corners' must have the same length")
        return self


class Renderer():
    def __init__(self) -> None:
        init_scene()

    def render(
        self,
        heights: List[int],
        corners: List[List[List[int]]],
        view: ViewType = "iso",
        size: int = 256,
        is_strict: bool = True,
    ) -> np.ndarray:
        """
        Returns:
            np.ndarray: A single-channel grayscale image,
            shape=(size, size), dtype=np.uint8.
        """
        params = RenderParams(
            is_strict=is_strict,
            heights=heights,
            corners=corners,
            view=view,
            size=size,
        )
        return self._render(params)

    def _render(self, params: RenderParams) -> np.ndarray:
        heights = get_heights_z(params.heights)
        levels = params.corners
        view = remap_view_name(params.view)
        size = params.size

        # Process for iso-reverse view
        if view == "iso-r":
            levels, heights = get_reverse_params(levels, heights)
            view = "iso"

        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir_raw = os.path.join(temp_dir, "raw")
            save_dir_process = os.path.join(temp_dir, "process")
            os.makedirs(save_dir_raw, exist_ok=True)
            os.makedirs(save_dir_process, exist_ok=True)
            now = datetime.now()
            case_name = now.strftime("%Y%m%d%H%M%S") + f"{now.microsecond:06d}"

            # Build case in blender
            build_case(levels, heights)

            # Render case in blender, will save images to "save_dir_raw"
            render_case_by_views(
                views=wrap_view(view),
                save_dir=save_dir_raw,
                case_name=case_name,
            )

            # Post-process images
            results = pipeline(
                views=wrap_view(view),
                data_dir=save_dir_raw,
                case_name=case_name,
            )

            # Clear case in blender
            clear_case()

        output = results.get(view, None)
        if output is None:
            raise ValueError(f"no output for view {view}")
        output = cv2.resize(output, (size, size))
        return output


if __name__ == "__main__":

    test_params_raw = {
        "bound_size": 20,
        "min_size": 4,
        "num_levels": 3,
        "num_corners": [2, 3, 2],
        "ratio": [0.7, 0.1, 0.2],
        "level_heights": [8, 6, 6],
        "level_heights_z": [8, 14, 20],
        "corners": [
            [[16, 20], [20, 6]], [[4, 20], [11, 14], [20, 6]], [[7, 10], [16, 6]]
        ]
    }

    renderer = Renderer()

    for view in [
        "iso", "top", "right", "front", "iso-left", "iso-right", "iso-reverse"
    ]:
        test_params = {
            "heights": test_params_raw["level_heights"],
            "corners": test_params_raw["corners"],
            "view": view,
            "size": 512,
        }
        img_arr = renderer.render(**test_params)
        cv2.imshow(f"output-{view}", img_arr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
