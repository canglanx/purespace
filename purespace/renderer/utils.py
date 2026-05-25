from typing import List, Tuple, Dict


def get_heights_z(heights: List[int]) -> List[int]:
    heights_z = []
    add_z = 0
    for h in heights:
        add_z += h
        heights_z.append(add_z)
    return heights_z


def remap_view_name(view: str) -> str:
    if view == "iso-left":
        return "iso-b"
    elif view == "iso-right":
        return "iso-a"
    elif view == "iso-reverse":
        return "iso-r"
    else:
        return view


def wrap_view(view: str) -> Dict[str, List[str | None]]:
    # Categorize "view" types into "single" and "merge" pipelines
    if view in ["iso", "top", "right", "front"]:
        return {"single": [view], "merge": []}
    if view in ["iso-a", "iso-b"]:
        return {"single": [], "merge": [view]}
    return {"single": [], "merge": []}


def get_reverse_corners(
    corners: List[List[int]],
    bound: int = 20,
) -> List[List[int]]:
    # Concave corners
    concave_corners = []
    temp_corners = []
    if corners[0][1] != bound:
        temp_corners.append([0, -1])
    temp_corners.extend(corners)
    if corners[-1][0] != bound:
        temp_corners.append([-1, 0])
    for i in range(len(temp_corners) - 1):
        concave_corners.append([temp_corners[i][0], temp_corners[i + 1][1]])
    # Symmetric
    reverse_corners = [
        [bound - pt[1], bound - pt[0]] for pt in concave_corners
    ]
    return reverse_corners


def get_reverse_params(
    levels: List[List[List[int]]],
    heights: List[int],
    bound: int = 20,
) -> Tuple[List[List[List[int]]], List[int]]:

    _heights = [heights[0]]
    for i in range(len(heights) - 1):
        _heights.append(heights[i + 1] - heights[i])
    reverse_heights = _heights[::-1]
    reverse_heights_z = get_heights_z(reverse_heights)

    reverse_levels = []
    for level_corners in levels[::-1]:
        reverse_levels.append(get_reverse_corners(level_corners, bound))

    return reverse_levels, reverse_heights_z
