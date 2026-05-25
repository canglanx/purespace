import cv2
import numpy as np
import os

from typing import List, Dict


line_detector = cv2.createLineSegmentDetector()


def detect_edges(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines, _, _, _ = line_detector.detect(img)
    edges = np.zeros_like(img, dtype=np.uint8)
    if lines is None:
        return edges
    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        cv2.line(
            edges,
            (x1, y1),
            (x2, y2),
            color=255,  # type: ignore
            thickness=10,
            lineType=cv2.LINE_AA
        )
    return edges


def merge_edges(img_lst: List[np.ndarray]) -> np.ndarray:
    if not img_lst:
        raise ValueError("'img_lst' cannot be empty")
    fixed_shape = img_lst[0].shape
    for idx in range(len(img_lst)):
        if img_lst[idx].shape != fixed_shape:
            img_lst[idx] = cv2.resize(
                img_lst[idx], (fixed_shape[1], fixed_shape[0])
            )  # type: ignore
    merged_edges = img_lst[0].copy()
    for img in img_lst[1:]:
        merged_edges = cv2.bitwise_or(merged_edges, img)
    return merged_edges


def center_bbox(
    img_gray: np.ndarray,
    if_horizontal: bool = True,
    if_vertical: bool = False,
) -> np.ndarray:
    # Center horizontally by default,
    # leaving the vertical position unchanged

    if len(img_gray.shape) != 2:
        raise ValueError("input image must be grayscale")

    mask = img_gray < 10  # Foreground is black
    if not np.any(mask) or (not if_horizontal and not if_vertical):
        return img_gray.copy()

    coords = np.column_stack(np.where(mask))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    foreground = img_gray[y0:y1, x0:x1]
    fh, fw = foreground.shape

    h, w = img_gray.shape
    canvas = np.full((h, w), 255, dtype=np.uint8)

    start_y = y0 if not if_vertical else max((h - fh) // 2, 0)
    start_x = x0 if not if_horizontal else max((w - fw) // 2, 0)

    canvas[start_y:start_y + fh, start_x:start_x + fw] = foreground
    return canvas


def single_pipeline(img_path: str) -> np.ndarray:
    return center_bbox(cv2.bitwise_not(detect_edges(img_path)))


def merge_pipeline(img_path_lst: List[str]) -> np.ndarray:
    result = center_bbox(
        cv2.bitwise_not(
            merge_edges([detect_edges(img_path) for img_path in img_path_lst])
        )
    )
    return result


def pipeline(
    views: Dict[str, List[str | None]],
    data_dir: str,
    case_name: str,
) -> Dict[str, np.ndarray]:
    single_name_lst = views["single"]
    merge_name_lst = views["merge"]

    results = {}
    for name in single_name_lst:
        results[name] = single_pipeline(
            os.path.join(data_dir, f"{case_name}-{name}.jpg")
        )
    for name in merge_name_lst:
        results[name] = merge_pipeline(
            [
                os.path.join(
                    data_dir, f"{case_name}-{name}-{axis}.jpg"
                ) for axis in ["x", "y", "z"]
            ]
        )
    return results
