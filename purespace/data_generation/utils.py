import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, List, Tuple

from purespace.data_generation.schemas import (
    Case, Question, Task, View
)


def is_negative_finished(
    negative_cases: List[Case], target: int
) -> Tuple[bool, List[View]]:
    view_progress = {view: 0 for view in View}
    unfinished_views = []
    for case in negative_cases:
        for view, value in case.renders.items():
            if value is not None:
                view_progress[view] += 1
    for view, progress in view_progress.items():
        if progress < target:
            unfinished_views.append(view)
    return len(unfinished_views) == 0, unfinished_views


def is_same_image(img1: np.ndarray, img2: np.ndarray, thres: int) -> bool:
    _, img1 = cv2.threshold(img1, 200, 255, cv2.THRESH_BINARY_INV)
    _, img2 = cv2.threshold(img2, 200, 255, cv2.THRESH_BINARY_INV)

    if img1.shape != img2.shape:
        return False

    fg_mask = np.logical_or(img1 > 0, img2 > 0)  # Foreground mask

    diff_img = np.zeros_like(img1)
    diff_img[(img1 != img2) & fg_mask] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    diff_img = cv2.morphologyEx(diff_img, cv2.MORPH_OPEN, kernel)

    diff_px_num = np.sum(diff_img == 255)
    flag = diff_px_num < thres

    return flag

# --------

def distribute_rotation_options(
    num_options: int, difficulty: int
) -> Dict[str, Any]:

    # We need 1 correct opts, m hard opts, and n-m-1 easy opts.
    # We have two pools, rotl and rotr, each containing
    # 1 correct opts, >= m hard opts, and >= n-m-1 easy opts.
    # We need to distribute these opts fairly.

    n, m = num_options, difficulty
    assert n > m

    options = {
        "correct": {
            "rotl": 0,
            "rotr": 0,
        },
        "hard": {
            "rotl": m // 2,
            "rotr": m // 2,
        },
        "easy": {
            "rotl": (n - m - 1) // 2,
            "rotr": (n - m - 1) // 2,
        },
    }

    def _switch_pool(pool):
        _mapping = {"rotl": "rotr", "rotr": "rotl"}
        return _mapping[pool]

    temp_pool = random.choice(["rotl", "rotr"])
    options["correct"][temp_pool] += 1

    if m % 2 != 0:
        temp_pool = _switch_pool(temp_pool)
        options["hard"][temp_pool] += 1
        if (n - m - 1) % 2 != 0:
            temp_pool = _switch_pool(temp_pool)
            options["easy"][temp_pool] += 1
    elif (n - m - 1) % 2 != 0:
        temp_pool = _switch_pool(temp_pool)
        options["easy"][temp_pool] += 1

    return options


def assemble_question(
    positive_case: Case,
    negative_cases: List[Case],
    task: Task,
    num_options: int,
) -> Question:

    stem = positive_case.renders[View.ISO]

    if task == Task.ROTATION:
        wrong_rotl_pool = [
            case.renders[View.ISO_LEFT] for case in negative_cases
            if case.renders[View.ISO_LEFT] is not None
        ]
        wrong_rotr_pool = [
            case.renders[View.ISO_RIGHT] for case in negative_cases
            if case.renders[View.ISO_RIGHT] is not None
        ]
        random.shuffle(wrong_rotl_pool)
        random.shuffle(wrong_rotr_pool)

        option_pool = {
            "correct": {
                "rotl": [positive_case.renders[View.ISO_LEFT]],
                "rotr": [positive_case.renders[View.ISO_RIGHT]],
            },
            "hard": {
                "rotl": wrong_rotl_pool,
                "rotr": wrong_rotr_pool,
            },
            "easy": {
                "rotl": [],
                "rotr": [],
            },
        }
        tgt_n_opts = distribute_rotation_options(
            num_options=num_options, difficulty=num_options-1
        )
        if tgt_n_opts["correct"]["rotl"] != 0:
            correct_option = option_pool["correct"]["rotl"][0]
        else:
            correct_option = option_pool["correct"]["rotr"][0]

        wrong_options = []
        if tgt_n_opts["hard"]["rotl"] != 0:
            wrong_options += option_pool["hard"]["rotl"][:tgt_n_opts["hard"]["rotl"]]
        if tgt_n_opts["hard"]["rotr"] != 0:
            wrong_options += option_pool["hard"]["rotr"][:tgt_n_opts["hard"]["rotr"]]

    elif task == Task.PROJECTION:
        correct_option = positive_case.renders[View.TOP]
        wrong_options = [
            case.renders[View.TOP] for case in negative_cases
            if case.renders[View.TOP] is not None
        ]
        random.shuffle(wrong_options)
        wrong_options = wrong_options[:num_options-1]

    elif task == Task.COMPLETION:
        correct_option = positive_case.renders[View.ISO_REVERSE]
        wrong_options = [
            case.renders[View.ISO_REVERSE] for case in negative_cases
            if case.renders[View.ISO_REVERSE] is not None
        ]
        random.shuffle(wrong_options)
        wrong_options = wrong_options[:num_options-1]

    # Shuffle options and assign answer
    ready_options = [correct_option] + wrong_options
    option_indices = list(range(num_options))
    shuffled_indices = option_indices[:]
    random.shuffle(shuffled_indices)
    answer = shuffled_indices.index(0)
    shuffled_options = [
        ready_options[shuffled_indices[i]] for i in range(num_options)
    ]

    question = Question(
        task=task,
        stem=stem,
        options=shuffled_options,
        answer=answer,
    )
    return question


def render_question(
    question: Question, labels: List[str], font_path: str
) -> np.ndarray:
    # Load images
    img_stem = Image.fromarray(
        cv2.cvtColor(question.stem, cv2.COLOR_BGR2RGB)
    )
    img_options = [
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for img in question.options
    ]
    # Load fonts
    letter_font = ImageFont.truetype(font_path, 70)
    text_font = ImageFont.truetype(font_path, 85)

    # Image size
    n_options = len(question.options)
    spacing = 40  # H space between options
    option_w, option_h = img_options[0].size
    grid_w = option_w * n_options + spacing * (n_options - 1)

    text_height = 120  # Text height
    spacing_text_options = 60  # V space between text and options
    canvas_width = max(img_stem.width, grid_w)
    canvas_height = (
        img_stem.height
        + text_height
        + spacing_text_options
        + option_h
    )

    # Make image
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    # Insert stem image
    canvas.paste(img_stem, ((canvas_width - img_stem.width) // 2, 0))

    # Insert text
    text_y = img_stem.height
    text_size = draw.textbbox((0, 0), question.task.text, font=text_font)
    text_width = text_size[2] - text_size[0]
    text_x = (canvas_width - text_width) // 2
    draw.text((text_x, text_y), question.task.text, fill="black", font=text_font)

    # Insert option images and labels
    for idx, img in enumerate(img_options):
        x = idx * (option_w + spacing)
        y = text_y + text_height + spacing_text_options
        canvas.paste(img, (x, y))
        draw.text((x + 45, y + 10), labels[idx], fill="black", font=letter_font)

    # Padding
    padding_y = 20
    final_canvas = Image.new(
        "RGB", (canvas_width, canvas_height + padding_y * 2), "white"
    )
    final_canvas.paste(canvas, (0, padding_y))

    final_canvas_arr = np.array(final_canvas)
    question_img = cv2.cvtColor(final_canvas_arr, cv2.COLOR_RGB2BGR)
    return question_img
