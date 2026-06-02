import json
import logging
import os
import string
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import cv2

from purespace.data_generation.geometry import (
    gen_heights,
    gen_negative_corners,
    gen_positive_corners,
)
from purespace.data_generation.schemas import (
    BatchConfig, Case, Task, View
)
from purespace.data_generation.utils import (
    assemble_question,
    is_negative_finished,
    is_same_image,
    render_question,
)
from purespace.renderer import Renderer


logger = logging.getLogger(__name__)


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(MODULE_DIR)
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)


def run_data_generation(config_dict: Dict[str, Any]) -> None:
    save_dir = os.path.join(
        os.path.abspath(
            config_dict.get("save_dir", os.path.join(PROJECT_DIR, "outputs"))
        ),
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(save_dir, exist_ok=True)

    enable_question_demo = config_dict.get("enable_question_demo", False)

    tasks = config_dict.get("tasks", [])
    if not tasks:
        logger.warning("No generation tasks found in configuration")
        return

    # One task is generated as one batch
    for task_idx, task_config in enumerate(tasks):
        logger.info("Processing generation task [%s/%s]", task_idx + 1, len(tasks))
        batch_config = BatchConfig(
            save_dir=save_dir,
            enable_question_demo=enable_question_demo,
            num_levels=task_config.get("num_levels"),
            num_corners=task_config.get("num_corners"),
            num_samples=task_config.get("num_samples")
        )

        # Generate batch samples
        cnt_samples = Generator(batch_config).gen_batch()

        logger.info(
            "Task [%s/%s] finished, generated %s samples",
            task_idx + 1, len(tasks), cnt_samples,
        )


class Generator():
    def __init__(self, config: BatchConfig) -> None:
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.renderer = Renderer()

    def gen_batch(self) -> int:
        cnt_samples, tgt_samples = 0, self.config.num_samples
        cnt_retry, max_retry = 0, self.config.max_retry_sample

        # Log interval for progress tracking
        log_interval = max(1, tgt_samples // 10)

        # For deduplication in positive case generation
        existing_cases = []

        while cnt_samples < tgt_samples and cnt_retry < max_retry:
            sample_flag, sample_case, err_msg = self.gen_sample(
                f"{cnt_samples:06d}", existing_cases,
            )
            if not sample_flag or sample_case is None:
                cnt_retry += 1
                logger.debug(
                    "%s, retrying (%s/%s)...",
                    err_msg,
                    cnt_retry,
                    max_retry,
                )
                continue

            cnt_samples += 1
            cnt_retry = 0
            existing_cases.append(sample_case)

            if cnt_samples % log_interval == 0:
                logger.info(
                    "Task progress: %s/%s (%s%%)",
                    cnt_samples,
                    tgt_samples,
                    int(cnt_samples / tgt_samples * 100),
                )

        if cnt_samples < tgt_samples:
            logger.warning(
                "Task stopped, generated %s samples out of %s required",
                cnt_samples, tgt_samples,
            )
        return cnt_samples

    def gen_sample(
        self, sample_name: str, existing_cases: List[Case]
    ) -> Tuple[bool, Optional[Case], str]:

        # Generate positive case
        positive_flag, positive_case = self.gen_positive_case(existing_cases)
        if not positive_flag or positive_case is None:
            return False, None, "Failed to generate positive case"

        # Generate negative cases, and render positive and negative cases
        negative_flag, negative_cases = self.gen_negative_cases(
            positive_case, self.config.num_options - 1
        )
        if not negative_flag or negative_cases is None:
            return False, None, "Failed to generate negative cases"

        # Save data
        self.save_sample_data(positive_case, negative_cases, sample_name)

        heights, corners = positive_case.heights, positive_case.corners
        return True, Case(heights=heights, corners=corners), ""

    def gen_positive_case(
        self, existing_cases: List[Case]
    ) -> Tuple[bool, Optional[Case]]:

        heights = gen_heights(
            self.config.num_levels, self.config.bound, self.config.minimum
        )
        flag, corners = gen_positive_corners(
            self.config.num_levels,
            self.config.num_corners,
            self.config.bound,
            self.config.minimum,
            self.config.ratio,
            self.config.max_retry_level,
            self.config.max_retry_case,
            [case.corners for case in existing_cases],
        )

        if not flag or not heights or not corners:
            return False, None

        return True, Case(heights=heights, corners=corners)

    def gen_negative_cases(
        self, positive_case: Case, target: int
    ) -> Tuple[bool, Optional[List[Case]]]:

        negatives = gen_negative_corners(
            positive_case.corners,
            self.config.bound,
            self.config.minimum,
        )
        raw_negative_cases = [
            Case(heights=positive_case.heights[:], corners=corners)
            for corners in negatives
        ]
        if len(raw_negative_cases) < target:
            logger.debug(
                "Negative cases insufficient, current=%s, target=%s",
                len(raw_negative_cases),
                target,
            )
            return False, None

        # Render positive case
        for view in View:
            img_render = self.renderer.render(
                positive_case.heights,
                positive_case.corners,
                view.value,
                self.config.render_size,
            )
            positive_case.renders[view] = img_render

        # Deduplication for negative cases, especially image comparison
        negative_cases = []
        is_finished, unfinished_views = is_negative_finished(negative_cases, target)
        while raw_negative_cases and not is_finished:
            curr_negative_case = raw_negative_cases.pop()

            # Greedy for rendering
            to_use = False
            for view in unfinished_views:
                img_render = self.renderer.render(
                    curr_negative_case.heights,
                    curr_negative_case.corners,
                    view.value,
                    self.config.render_size,
                )
                is_same = False
                img_anchors = [positive_case.renders[view]] + [
                    case.renders[view] for case in negative_cases
                    if case.renders[view] is not None
                ]
                for img_anchor in img_anchors:
                    if is_same_image(
                        img_anchor, img_render, view.diff_threshold
                    ):
                        is_same = True
                        break
                # Only add valid views for each negative case
                if not is_same:
                    curr_negative_case.renders[view] = img_render
                    to_use = True

            if to_use:
                negative_cases.append(curr_negative_case)
                is_finished, unfinished_views = is_negative_finished(
                    negative_cases, target
                )

        if not is_finished:
            return False, None

        return True, negative_cases

    def save_sample_data(
        self,
        positive_case: Case,
        negative_cases: List[Case],
        sample_name: str,
    ) -> None:
        num_levels = self.config.num_levels
        num_corners = self.config.num_corners
        num_options = self.config.num_options

        save_data_dir = os.path.join(self.config.save_dir, "images")
        setting_name = f"l{num_levels}_c{''.join(map(str, num_corners))}_{self.timestamp}"
        save_sample_dir = os.path.join(save_data_dir, setting_name, sample_name)
        os.makedirs(save_sample_dir, exist_ok=True)

        # Positive
        for view in View:
            if positive_case.renders[view] is None:  # Should not happen
                continue
            cv2.imwrite(
                os.path.join(save_sample_dir, f"{sample_name}_{view.save_name}.jpg"),
                positive_case.renders[view]
            )

        # Negative
        view_img_idx = {view: 0 for view in View}
        for case in negative_cases:
            for view in View:
                # Some views can be None for some cases
                if case.renders[view] is not None:
                    cv2.imwrite(
                        os.path.join(
                            save_sample_dir,
                            f"{sample_name}_{view.save_name}_{view_img_idx[view]}.jpg"
                        ),
                        case.renders[view],
                    )
                    view_img_idx[view] += 1

        # Metadata
        metadata = {
            "bound_size": self.config.bound,
            "min_size": self.config.minimum,
            "num_levels": self.config.num_levels,
            "num_corners": self.config.num_corners,
            "ratio": self.config.ratio,
            "heights": positive_case.heights,
            "corners": positive_case.corners,
        }
        with open(
            os.path.join(save_sample_dir, f"{sample_name}_metadata.json"), "w"
        ) as f:
            json.dump(metadata, f)

        # Question demo
        if self.config.enable_question_demo:
            save_demo_dir = os.path.join(save_sample_dir, "demo")
            os.makedirs(save_demo_dir, exist_ok=True)
            for task in Task:
                question = assemble_question(
                    positive_case, negative_cases, task, num_options
                )
                labels = list(string.ascii_uppercase[:num_options])
                question_img = render_question(
                    question,
                    labels,
                    font_path=os.path.join(
                        PACKAGE_DIR, "assets/fonts/Arimo-Bold.ttf"
                    ),
                )
                answer_label = labels[question.answer]
                cv2.imwrite(
                    os.path.join(
                        save_demo_dir,
                        f"{sample_name}_{task.value}_answer_{answer_label}.jpg"
                    ),
                    question_img,
                )
