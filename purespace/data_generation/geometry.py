"""
Note:
Internal algorithms use corners in List[List[Tuple[int, int]]] format,
while external use List[List[List[int]]] format.
"""

import copy
import logging
import random
from typing import List, Optional, Tuple


logger = logging.getLogger(__name__)


Point = Tuple[int, int]
Level = List[Point]
CornersInternal = List[Level]
CornersExternal = List[List[List[int]]]


def tuple2list(corners: CornersInternal) -> CornersExternal:
    return [[list(point) for point in level] for level in corners]


def list2tuple(corners: CornersExternal) -> CornersInternal:
    return [[(x, y) for x, y in level] for level in corners]

# ---------------------

def gen_heights(
    num_levels: int, bound: int = 20, minimum: int = 4
) -> List[int]:
    assert num_levels * minimum <= bound
    heights = [minimum] * num_levels
    remaining = bound - sum(heights)
    for _ in range(remaining):
        idx = random.randint(0, num_levels - 1)
        heights[idx] += 1

    random.shuffle(heights)
    return heights

# ---------------------

def gen_positive_corners(
    num_levels: int,
    num_corners: List[int],
    bound: int = 20,
    minimum: int = 4,
    ratio: Optional[List[float]] = None,
    max_retry_level: int = 50,
    max_retry_case: int = 50,
    existing_cases: Optional[List[CornersExternal]] = None,
) -> Tuple[bool, Optional[CornersExternal]]:

    # Initialize default ratio
    if ratio is None:
        ratio = [0.7, 0.1, 0.2]

    for cnt_retry_case in range(max_retry_case):
        positive_corners: CornersInternal = []

        # For base level, the lower level has only one corner (bound, bound)
        lower_level_corners = [(bound, bound)]

        is_success = True
        err_msg = ""

        # Generate positive corners level by level
        for level_idx in range(num_levels):
            is_base_level = level_idx == 0
            is_top_level = level_idx == num_levels - 1

            # Target number of corners in this level
            tgt_n_corners = num_corners[level_idx]

            # Since corners are generated one-by-one
            order = list(range(tgt_n_corners))
            random.shuffle(order)

            level_flag = False
            level_corners = []
            cnt_retry_level = 0

            while not level_flag and cnt_retry_level < max_retry_level:
                # Core logic
                level_flag, level_corners = gen_level(
                    tgt_n_corners,
                    order,
                    lower_level_corners,
                    is_base_level,
                    is_top_level,
                    bound,
                    minimum,
                    ratio,
                )
                cnt_retry_level += 1

            # After retrying for several times, still failed, then
            # need to generate from base level again.
            if not level_flag or level_corners is None:
                is_success = False
                err_msg = (
                    f"Failed to generate level [{level_idx+1}/{num_levels}]"
                )
                break

            positive_corners.append(level_corners)
            lower_level_corners = copy.deepcopy(level_corners)

        if not is_success:
            logger.debug(
                "%s, retrying (%s/%s)...",
                err_msg,
                cnt_retry_case + 1,
                max_retry_case,
            )
            continue

        # Convert for external use
        positive_corners_ext: CornersExternal = tuple2list(positive_corners)

        # Deduplication
        if existing_cases and positive_corners_ext in existing_cases:
            logger.debug(
                "Generated case already exists, retrying (%s/%s)...",
                cnt_retry_case + 1,
                max_retry_case,
            )
            continue

        return True, positive_corners_ext

    return False, None


def gen_level(
    num_corners: int,
    order: List[int],
    lower_level: Level,
    is_base: bool = False,
    is_top: bool = False,
    bound: int = 20,
    minimum: int = 4,
    ratio: Optional[List[float]] = None,
    max_retry: int = 50,
) -> Tuple[bool, Optional[Level]]:

    # Initialize default ratio
    if ratio is None:
        ratio = [0.7, 0.1, 0.2]

    # Get the available interior / edge / convex corner points
    (
        _, avail_interior_points, avail_edge_points, avail_corner_points
    ) = find_avail_points(lower_level, bound, minimum)

    # Start choosing points, considering the already-chosen ones of this level
    for _ in range(max_retry):
        chosen_corners: List[Optional[Point]] = [None] * num_corners

        is_success = True

        for order_idx in range(num_corners):
            # Here the corner indices 0 ~ n refer to
            # the coordinates that near (0, bound) to near (bound, 0),
            # which looks like:
            #          .  -> idx=0, near (0, bound)
            #        .    -> idx=1
            #      .      -> idx=2
            #    .        -> idx=3
            #  .          -> idx=4, near (bound, 0)
            corner_idx = order[order_idx]

            # Range constraints derived from the order
            x_min = (corner_idx + 1) * minimum
            x_max = bound - (num_corners - corner_idx - 1) * minimum
            y_min = (num_corners - corner_idx) * minimum
            y_max = bound - corner_idx * minimum

            # Range constraints derived from the already-chosen corners
            for j in range(corner_idx - 1, -1, -1):
                corner_j = chosen_corners[j]
                if corner_j is not None:
                    margin = corner_idx - j
                    x_min = max(x_min, corner_j[0] + margin * minimum)
                    y_max = min(y_max, corner_j[1] - margin * minimum)
                    break
            for j in range(corner_idx + 1, num_corners, +1):
                corner_j = chosen_corners[j]
                if corner_j is not None:
                    margin = j - corner_idx
                    x_max = min(x_max, corner_j[0] - margin * minimum)
                    y_min = max(y_min, corner_j[1] + margin * minimum)
                    break

            # Get the real available candidate points
            candidate_points_interior = [
                point for point in avail_interior_points
                if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
            ]
            candidate_points_edge = [
                point for point in avail_edge_points
                if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
            ]
            candidate_points_corner = [
                point for point in avail_corner_points
                if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
            ]

            # Additional constraints for the top level
            if is_top:
                candidate_points_edge = [
                    point for point in candidate_points_edge
                    if not (point[0] == bound or point[1] == bound)
                ]
                candidate_points_corner = [
                    point for point in candidate_points_corner
                    if not (point[0] == bound or point[1] == bound)
                ]

            # Choose one pool from interior / edge / corner pools,
            # with basic ratio [0.7, 0.1, 0.2]
            candidate_pools = [
                candidate_points_interior,
                candidate_points_edge,
                candidate_points_corner,
            ]
            ratio_lst = []
            for cand_idx, cand in enumerate(candidate_pools):
                if cand:
                    ratio_lst.append(ratio[cand_idx])
                else:
                    ratio_lst.append(0)

            if sum(ratio_lst) == 0:
                is_success = False
                break

            # For base-level, there must be a point on each edge
            if is_base and corner_idx in [0, num_corners-1]:
                ratio_lst = [0, 1, 0]

            # Random choose one pool
            choice = random.choices(
                population=[0, 1, 2],
                weights=[r/sum(ratio_lst) for r in ratio_lst],
                k=1,
            )[0]
            candidate_points = candidate_pools[choice]

            # Tend to choose outer points
            chosen_corner_point = random.choices(
                candidate_points, weights=[x+y for x, y in candidate_points], k=1
            )[0]
            chosen_corners[corner_idx] = chosen_corner_point

        if not is_success:
            continue

        final_corners = [cn for cn in chosen_corners if cn is not None]

        # Must ensure the level count, as adjacent identical levels
        # will automatically merge into one level,
        # and cause an incorrect total number of levels.
        if final_corners == lower_level:
            continue

        return True, final_corners

    return False, None


def find_avail_points(
    lower_level: Level,
    bound: int = 20,
    minimum: int = 4,
) -> Tuple[List[Point], List[Point], List[Point], List[Point]]:

    # Get the boundary of this level, i.e.,
    # the corner points and the edge points,
    # according to the corners of the lower level

    # Corner points (part of boundary)
    corner_points = copy.deepcopy(lower_level)

    # Edge points (part of boundary) do not contain corner points
    x_temp, y_temp = 0, lower_level[0][1]
    edge_points = [(x_temp, y_temp)]
    for i, point in enumerate(lower_level):
        x_temp += 1
        while x_temp < point[0]:
            edge_points.append((x_temp, y_temp))
            x_temp += 1
        if i == len(lower_level) - 1:
            y_end = 0
        else:
            y_end = lower_level[i + 1][1]
        while y_temp > y_end:
            y_temp -= 1
            edge_points.append((x_temp, y_temp))

    # Get the available interior points,
    # only considering the distance to the edges
    all_points = [(i, j) for i in range(bound + 1) for j in range(bound + 1)]
    inner_points = []
    for corner in corner_points:
        for point in all_points:
            if (
                point[0] < corner[0]
                and point[1] < corner[1]
                and point not in inner_points
            ):
                inner_points.append(point)
    outer_points = [point for point in all_points if point not in inner_points]

    # Some additional constraints for image aesthetics
    avail_interior_points = []
    for point in inner_points:
        if point[0] + minimum > bound or point[1] + minimum > bound:
            continue
        if (
            (point[0] + minimum - 1, point[1]) in outer_points
            or (point[0], point[1] + minimum - 1) in outer_points
            or (point[0] + minimum - 1, point[1] + minimum - 1) in outer_points
        ):
            continue
        avail_interior_points.append(point)

    # Get the available edge points,
    # only considering the distance to other edges
    avail_edge_points = filter_avail_edge_points(
        edge_points, corner_points, minimum
    )
    # Get the available convex corner points
    avail_corner_points = corner_points

    # Concat all the available points
    all_avail_points = (
        avail_interior_points
        + avail_edge_points
        + avail_corner_points
    )
    all_avail_points = copy.deepcopy(all_avail_points)

    return (
        all_avail_points,
        avail_interior_points,
        avail_edge_points,
        avail_corner_points,
    )


def filter_avail_edge_points(
    edge_points: List[Point],
    corner_points: List[Point],
    minimum: int = 4,
) -> List[Point]:
    # To avoid the chosen edge point be too closer to another edge

    edge_x_lst = [point[0] for point in corner_points]
    edge_y_lst = [point[1] for point in corner_points]

    avail_edge_points = []
    for point in edge_points:
        is_avail = True
        # Current point at a concave corner
        if point[0] in edge_x_lst and point[1] in edge_y_lst:
            avail_edge_points.append(point)
            continue
        # Current point on a vertical edge
        elif point[0] in edge_x_lst:
            for y in edge_y_lst:
                if point[1] - minimum < y < point[1] + minimum:
                    is_avail = False
                    break
            if is_avail:
                avail_edge_points.append(point)
        # Current point on a horizontal edge
        elif point[1] in edge_y_lst:
            for x in edge_x_lst:
                if point[0] - minimum < x < point[0] + minimum:
                    is_avail = False
                    break
            if is_avail:
                avail_edge_points.append(point)
        else:
            raise ValueError(f"wrong edge point {point}")

    return avail_edge_points

# ---------------------

def gen_negative_corners(
    positive_corners: CornersExternal, bound: int = 20, minimum: int = 4
) -> List[CornersExternal]:

    # One positive case can make more than one negative cases
    negative_cases: List[CornersInternal] = []

    # Modify on one of the levels
    all_levels = list2tuple(positive_corners)

    # Expansion - choose one corner to move outer
    negative_cases.extend(gen_expansion_base_level(all_levels, bound, minimum))
    negative_cases.extend(gen_expansion_upper_level(all_levels, bound, minimum))

    # Contraction - choose one corner to move inner
    negative_cases.extend(gen_contraction(all_levels, bound, minimum))

    # Deduplication
    unique_negative_cases: List[CornersInternal] = []
    for case in negative_cases:
        if case not in unique_negative_cases:
            unique_negative_cases.append(case)

    # Additional constraints for the top and base levels
    final_negative_cases = filter_valid_top_and_base(unique_negative_cases, bound)

    random.shuffle(final_negative_cases)
    negative_cases_ext = [tuple2list(case) for case in final_negative_cases]
    return negative_cases_ext


def gen_expansion_base_level(
    positive_corners: CornersInternal, bound: int = 20, minimum: int = 4
) -> List[CornersInternal]:

    new_negative_cases = []
    for level_idx in range(1):  # Only for the base level
        chosen_level_idx = level_idx
        old_chosen_level = positive_corners[chosen_level_idx]

        # Get naive available points according to the lower level
        avail_points = [
            (i, j) for i in range(bound + 1) for j in range(bound + 1)
            if (i, j) != (bound, bound)
        ]

        # For each corner on this level
        num_corners = len(old_chosen_level)
        for corner_idx in range(num_corners):
            chosen_corner_idx = corner_idx
            old_chosen_corner = old_chosen_level[chosen_corner_idx]

            # Add naive available expand-to points
            avail_exp_points = []
            if chosen_corner_idx >= 1:  # expand on y-axis
                check_point = (
                    old_chosen_corner[0],  # x
                    old_chosen_level[chosen_corner_idx - 1][1],  # y
                )
                if check_point in avail_points:
                    avail_exp_points.append(check_point)
            if chosen_corner_idx <= num_corners - 2:  # expand on x-axis
                check_point = (
                    old_chosen_level[chosen_corner_idx + 1][0],  # x
                    old_chosen_corner[1],  # y
                )
                if check_point in avail_points:
                    avail_exp_points.append(check_point)

            avail_exp_points = sorted(list(set(avail_exp_points)))

            # Get upper level for validation below
            if chosen_level_idx < len(positive_corners) - 1:
                upper_level = positive_corners[chosen_level_idx + 1]
            else:
                upper_level = None

            for exp_point in avail_exp_points:
                # Replace the old corner with the expanded new one
                modified_level = copy.deepcopy(old_chosen_level)
                modified_level[chosen_corner_idx] = exp_point
                # Validate the modified level
                is_valid, new_chosen_level = clean_and_validate_modification(
                    modified_level, bound, minimum, upper_level, is_contraction=False,
                )
                # Add to output
                if is_valid:
                    new_all_levels = copy.deepcopy(positive_corners)
                    new_all_levels[chosen_level_idx] = new_chosen_level
                    new_negative_cases.append(new_all_levels)

    return new_negative_cases


def gen_expansion_upper_level(
    positive_corners: CornersInternal, bound: int = 20, minimum: int = 4
) -> List[CornersInternal]:

    new_negative_cases = []
    for level_idx in range(1, len(positive_corners)):
        chosen_level_idx = level_idx
        old_chosen_level = positive_corners[chosen_level_idx]

        # Get naive available points according to the lower level
        lower_level_corners = positive_corners[chosen_level_idx - 1]
        avail_points, _, avail_edge_points, avail_corner_points = find_avail_points(
            lower_level_corners, bound, minimum
        )

        # For each corner on this level
        num_corners = len(old_chosen_level)
        for corner_idx in range(num_corners):
            chosen_corner_idx = corner_idx
            old_chosen_corner = old_chosen_level[chosen_corner_idx]

            # Find the naive available expand-to points
            avail_exp_points_ec, avail_exp_points_sc = [], []

            # Add naive available expand-to points
            # on the edges and corners (ec) of the lower level
            for pt in avail_edge_points + avail_corner_points:
                if pt[0] == old_chosen_corner[0] and pt[1] > old_chosen_corner[1]:
                    if chosen_corner_idx >= 1:
                        y_bd = old_chosen_level[chosen_corner_idx - 1][1]
                    else:
                        y_bd = float("inf")
                    if pt[1] <= y_bd:
                        avail_exp_points_ec.append(pt)
                if pt[1] == old_chosen_corner[1] and pt[0] > old_chosen_corner[0]:
                    if chosen_corner_idx <= num_corners - 2:
                        x_bd = old_chosen_level[chosen_corner_idx + 1][0]
                    else:
                        x_bd = float("inf")
                    if pt[0] <= x_bd:
                        avail_exp_points_ec.append(pt)

            # For better effect, we only remain the most outer point to expand to.
            # Also, expansion must cause some changes on the relation between
            # edge lines. Otherwise, the question can be confusing.
            old_x = old_chosen_corner[0]
            old_y = old_chosen_corner[1]
            x_outer = old_x
            y_outer = old_y
            for p in avail_exp_points_ec:
                if p[0] == old_x:
                    y_outer = max(y_outer, p[1])
                elif p[1] == old_y:
                    x_outer = max(x_outer, p[0])
            avail_exp_points_ec = []
            if y_outer > old_y:
                avail_exp_points_ec.append((old_x, y_outer))
            if x_outer > old_x:
                avail_exp_points_ec.append((x_outer, old_y))

            # Add naive available expand-to points on the corners of the same level (sc)
            if chosen_corner_idx >= 1:  # expand on y-axis
                check_point = (
                    old_chosen_corner[0],  # x
                    old_chosen_level[chosen_corner_idx - 1][1],  # y
                )
                if check_point in avail_points:
                    avail_exp_points_sc.append(check_point)
            if chosen_corner_idx <= num_corners - 2:  # expand on x-axis
                check_point = (
                    old_chosen_level[chosen_corner_idx + 1][0],  # x
                    old_chosen_corner[1],  # y
                )
                if check_point in avail_points:
                    avail_exp_points_sc.append(check_point)

            # All naive available points
            avail_exp_points = avail_exp_points_ec + avail_exp_points_sc
            avail_exp_points = sorted(list(set(avail_exp_points)))

            # Get upper level for validation below
            if chosen_level_idx < len(positive_corners) - 1:
                upper_level = positive_corners[chosen_level_idx + 1]
            else:
                upper_level = None

            for exp_point in avail_exp_points:
                # Replace the old corner with the expanded new one
                modified_level = copy.deepcopy(old_chosen_level)
                modified_level[chosen_corner_idx] = exp_point
                # Validate the modified level
                is_valid, new_chosen_level = clean_and_validate_modification(
                    modified_level, bound, minimum, upper_level, is_contraction=False,
                )
                # Add to output
                if is_valid:
                    new_all_levels = copy.deepcopy(positive_corners)
                    new_all_levels[chosen_level_idx] = new_chosen_level
                    new_negative_cases.append(new_all_levels)

    return new_negative_cases


def gen_contraction(
    positive_corners: CornersInternal, bound: int = 20, minimum: int = 4
) -> List[CornersInternal]:

    new_negative_cases = []
    for level_idx in range(0, len(positive_corners)):
        chosen_level_idx = level_idx
        old_chosen_level = positive_corners[chosen_level_idx]

        # x-coords and y-coords that can be contracted from
        lower_level_x_bds, lower_level_y_bds = [], []
        if chosen_level_idx > 0:
            lower_level_corners = positive_corners[chosen_level_idx - 1]
            for point in lower_level_corners:
                lower_level_x_bds.append(point[0])
                lower_level_y_bds.append(point[1])

        if chosen_level_idx == 0:
            avail_points = [
                (i, j) for i in range(bound + 1) for j in range(bound + 1)
                if (i, j) != (bound, bound)
            ]
        else:
            avail_points, _, _, _ = find_avail_points(
                lower_level_corners, bound, minimum
            )

        # For each corner on this level
        num_corners = len(old_chosen_level)
        for corner_idx in range(num_corners):
            chosen_corner_idx = corner_idx
            old_chosen_corner = old_chosen_level[chosen_corner_idx]

            # Some corners cannot be contracted
            if chosen_level_idx < len(positive_corners) - 1:
                if old_chosen_corner in positive_corners[chosen_level_idx + 1]:
                    continue
            if (
                chosen_level_idx == 0
                and chosen_corner_idx in [0, num_corners - 1]
            ):
                continue
            is_stack_x, is_stack_y = False, False
            if chosen_level_idx > 0:
                if old_chosen_corner[0] in lower_level_x_bds:
                    is_stack_x = True
                if old_chosen_corner[1] in lower_level_y_bds:
                    is_stack_y = True
            is_stack = is_stack_x or is_stack_y
            if (
                chosen_level_idx == len(positive_corners) - 1
                and num_corners == 1
                and not is_stack
            ):
                continue

            # Add naive available contract-to points
            avail_con_points = []

            # x-coords and y-coords that can be contracted to
            old_chosen_corner_x_0 = 0
            if chosen_corner_idx > 0:
                old_chosen_corner_x_0 = old_chosen_level[chosen_corner_idx - 1][0]
            old_chosen_corner_y_0 = 0
            if chosen_corner_idx < num_corners - 1:
                old_chosen_corner_y_0 = old_chosen_level[chosen_corner_idx + 1][1]
            upper_level_x_bds, upper_level_y_bds = [], []
            if chosen_level_idx < len(positive_corners) - 1:
                upper_level_corners = positive_corners[chosen_level_idx + 1]
                for point in upper_level_corners:
                    if point[1] >= old_chosen_corner_y_0:
                        upper_level_x_bds.append(point[0])
                    if point[0] >= old_chosen_corner_x_0:
                        upper_level_y_bds.append(point[1])

            # x-direction
            if chosen_corner_idx == 0:
                same_level_x_bd = 0
            else:
                same_level_x_bd = old_chosen_level[chosen_corner_idx - 1][0]

            if old_chosen_corner[0] not in upper_level_x_bds:
                middle_avail_points = []
                cur_x, cur_y = old_chosen_corner[0] - 1, old_chosen_corner[1]
                while cur_x != same_level_x_bd and cur_x not in upper_level_x_bds:
                    if is_stack_x and old_chosen_corner[0] - cur_x >= minimum:
                        middle_avail_points.append((cur_x, cur_y))
                    cur_x -= 1

                if num_corners == 1 and cur_x == 0:
                    pass
                elif (cur_x, cur_y) in avail_points:
                    avail_con_points.append((cur_x, cur_y))

                if is_stack_x:
                    # Only add one middle available point
                    random.shuffle(middle_avail_points)
                    for point in middle_avail_points:
                        if point[0] - cur_x >= minimum and point in avail_points:
                            avail_con_points.append(point)
                            break

            # y-direction
            if chosen_corner_idx == num_corners - 1:
                same_level_y_bd = 0
            else:
                same_level_y_bd = old_chosen_level[chosen_corner_idx + 1][1]

            if old_chosen_corner[1] not in upper_level_y_bds:
                middle_avail_points = []
                cur_x, cur_y = old_chosen_corner[0], old_chosen_corner[1] - 1
                while cur_y != same_level_y_bd and cur_y not in upper_level_y_bds:
                    if is_stack_y and old_chosen_corner[1] - cur_y >= minimum:
                        middle_avail_points.append((cur_x, cur_y))
                    cur_y -= 1

                if num_corners == 1 and cur_y == 0:
                    pass
                elif (cur_x, cur_y) in avail_points:
                    avail_con_points.append((cur_x, cur_y))

                if is_stack_y:
                    # Only add one middle available point
                    random.shuffle(middle_avail_points)
                    for point in middle_avail_points:
                        if point[1] - cur_y >= minimum and point in avail_points:
                            avail_con_points.append(point)
                            break

            avail_con_points = sorted(list(set(avail_con_points)))

            # Get upper level for validation below
            if chosen_level_idx < len(positive_corners) - 1:
                upper_level = positive_corners[chosen_level_idx + 1]
            else:
                upper_level = None

            for con_point in avail_con_points:
                # Replace the old corner with the contracted new one
                modified_level = copy.deepcopy(old_chosen_level)
                modified_level[chosen_corner_idx] = con_point
                # Validate the modified level
                is_valid, new_chosen_level = clean_and_validate_modification(
                    modified_level, bound, minimum, upper_level, is_contraction=True,
                )
                # Add to output
                if is_valid:
                    new_all_levels = copy.deepcopy(positive_corners)
                    new_all_levels[chosen_level_idx] = new_chosen_level
                    new_negative_cases.append(new_all_levels)

    return new_negative_cases


def clean_and_validate_modification(
    modified_level: Level,
    bound: int = 20,
    minimum: int = 4,
    upper_level: Optional[Level] = None,
    is_contraction: bool = False,
) -> Tuple[bool, Level]:

    # Modification may cause redundant corners, i.e.,
    # corners with the same x-coord or the same y-coord.
    # Note that this kind of situation can only happen
    # on excatly one corner pair.
    redundant_corners = []
    valid_corners = []
    for p in modified_level:
        if redundant_corners:
            break
        if is_contraction:
            if p[0] == 0:
                redundant_corners.append(p)
                break
            elif p[1] == 0:
                redundant_corners.append(p)
                break
        for pp in modified_level:
            if p[0] == pp[0] and p[1] < pp[1]:
                redundant_corners.append(p)
                break
            elif p[1] == pp[1] and p[0] < pp[0]:
                redundant_corners.append(p)
                break

    for p in modified_level:
        if p not in redundant_corners:
            valid_corners.append(p)
    cleaned_level = valid_corners

    # Although we choose the available points above,
    # but they are only "naive" available.
    # Because we have not check the "minumum" from
    # the same level and the upper level after modification.
    # Hence, check if "real" available, and then add to output.
    is_valid = True

    # Check if this level is correct internally
    if is_valid and len(cleaned_level) > 1:
        for i in range(len(cleaned_level) - 1):
            if (
                abs(cleaned_level[i][0] - cleaned_level[i+1][0]) < minimum
                or abs(cleaned_level[i][1] - cleaned_level[i+1][1]) < minimum
            ):
                is_valid = False
                break

    # Check if this level agrees with the upper level
    if is_valid and upper_level is not None:
        upper_avail_points, _, _, _ = find_avail_points(
            cleaned_level, bound, minimum
        )
        for p in upper_level:
            if p not in upper_avail_points:
                is_valid = False
                break

    return is_valid, cleaned_level


def filter_valid_top_and_base(
    negative_cases: List[CornersInternal], bound: int = 20
) -> List[CornersInternal]:
    final_negative_cases = []
    for case in negative_cases:
        is_valid = True

        top_level = case[-1]
        for point in top_level:
            if point[0] == bound or point[1] == bound:
                is_valid = False
                break
        if not is_valid:
            continue

        base_level = case[0]
        if len(base_level) == 1:
            is_valid = False
        elif base_level[0][1] != bound or base_level[-1][0] != bound:
            is_valid = False
        if not is_valid:
            continue
        final_negative_cases.append(case)
    return final_negative_cases
