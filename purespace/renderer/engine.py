# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

import logging
import math
import os
from typing import Any, Dict, List, Tuple

import bpy
import mathutils
import numpy as np


logger = logging.getLogger(__name__)


def clear_scene() -> None:
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat, do_unlink=True)
    for cam in bpy.data.cameras:
        bpy.data.cameras.remove(cam, do_unlink=True)
    for light in bpy.data.lights:
        bpy.data.lights.remove(light, do_unlink=True)


def create_camera(
    name: str,
    location: Tuple[int, int, int],
    target: Tuple[int, int, int],
    cam_type: str = "ORTHO",
    ortho_scale: int = 40,
) -> Any:
    bpy.ops.object.camera_add(location=location, rotation=(0, 0, 0))
    cam = bpy.context.active_object
    cam.name = name

    cam.data.type = cam_type  # ["PERSP", "ORTHO"]
    cam.data.ortho_scale = ortho_scale

    cam_target = mathutils.Vector(target)
    cam_direction = cam_target - cam.location
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = cam_direction.to_track_quat("-Z", "Y")
    return cam


def init_scene(render_size: int = 1024) -> None:
    clear_scene()

    # Global scene settings
    scene = bpy.context.scene
    scene.render.resolution_x = render_size
    scene.render.resolution_y = render_size
    scene.render.use_freestyle = False
    scene.render.image_settings.file_format = "JPEG"

    # Make sure the rendered color can be controlled
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.exposure = 0

    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)

    # ***************************************************
    # ISO-view and ISOA-view and ISOB-view
    create_camera(name="CAM_ISO", location=(30, 30, 30), target=(10, 10, 10))
    create_camera(name="CAM_ISOA", location=(-10, 30, 30), target=(10, 10, 10))
    create_camera(name="CAM_ISOB", location=(30, -10, 30), target=(10, 10, 10))

    # Create the material "MAT_ISO" for the three cameras above.
    # Its feature is: Assigning different grayscale values to
    # the faces with normal vectors along the X / Y / Z axis respectively.

    mat = bpy.data.materials.new(name="MAT_ISO")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    geometry = nodes.new("ShaderNodeNewGeometry")
    separate = nodes.new("ShaderNodeSeparateXYZ")

    abs_x = nodes.new("ShaderNodeMath")
    abs_y = nodes.new("ShaderNodeMath")
    abs_z = nodes.new("ShaderNodeMath")
    abs_x.operation = "ABSOLUTE"
    abs_y.operation = "ABSOLUTE"
    abs_z.operation = "ABSOLUTE"

    greater_yx = nodes.new("ShaderNodeMath")  # Y > X
    greater_yx.operation = "GREATER_THAN"
    greater_zy = nodes.new("ShaderNodeMath")  # Z > Y
    greater_zy.operation = "GREATER_THAN"
    greater_zxy = nodes.new("ShaderNodeMath")  # Z > max(X, Y)
    greater_zxy.operation = "GREATER_THAN"

    emission_x = nodes.new("ShaderNodeEmission")
    emission_y = nodes.new("ShaderNodeEmission")
    emission_z = nodes.new("ShaderNodeEmission")

    emission_x.inputs["Color"].default_value = (0.3, 0.3, 0.3, 1)
    emission_y.inputs["Color"].default_value = (0.6, 0.6, 0.6, 1)
    emission_z.inputs["Color"].default_value = (0.9, 0.9, 0.9, 1)

    mix_yx = nodes.new("ShaderNodeMixShader")  # X or Y
    mix_final = nodes.new("ShaderNodeMixShader")  # Z or XY

    links.new(geometry.outputs["Normal"], separate.inputs[0])
    links.new(separate.outputs["X"], abs_x.inputs[0])
    links.new(separate.outputs["Y"], abs_y.inputs[0])
    links.new(separate.outputs["Z"], abs_z.inputs[0])

    links.new(abs_y.outputs[0], greater_yx.inputs[0])
    links.new(abs_x.outputs[0], greater_yx.inputs[1])

    links.new(abs_z.outputs[0], greater_zy.inputs[0])
    links.new(abs_y.outputs[0], greater_zy.inputs[1])

    links.new(greater_yx.outputs[0], mix_yx.inputs[0])
    links.new(emission_x.outputs[0], mix_yx.inputs[1])
    links.new(emission_y.outputs[0], mix_yx.inputs[2])

    links.new(greater_zy.outputs[0], mix_final.inputs[0])
    links.new(mix_yx.outputs[0], mix_final.inputs[1])
    links.new(emission_z.outputs[0], mix_final.inputs[2])

    links.new(mix_final.outputs[0], output.inputs["Surface"])

    # ***************************************************
    # FRONT-view and TOP-view and RIGHT-view
    create_camera(name="CAM_FRONT", location=(30, 10, 10), target=(10, 10, 10))
    create_camera(name="CAM_RIGHT", location=(10, 30, 10), target=(10, 10, 10))
    cam_top = create_camera(name="CAM_TOP", location=(10, 10, 30), target=(10, 10, 10))
    cam_top.rotation_mode = "XYZ"
    cam_top.rotation_euler.z += math.radians(90)

    # Create the material "MAT_X" for the "CAM_FRONT" camera.
    # Its feature is: Only targeting the faces with normal vectors along the X axis,
    # assigning different grayscale values to the faces with different X coordinates.

    mat = bpy.data.materials.new("MAT_X")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")

    emission_major = nodes.new("ShaderNodeEmission")
    emission_other = nodes.new("ShaderNodeEmission")
    emission_other.inputs["Color"].default_value = (0, 0, 0, 1)

    mix_shader = nodes.new("ShaderNodeMixShader")

    geometry = nodes.new("ShaderNodeNewGeometry")
    separate_normal = nodes.new("ShaderNodeSeparateXYZ")

    math_abs = nodes.new("ShaderNodeMath")
    math_abs.operation = "ABSOLUTE"

    math_greater = nodes.new("ShaderNodeMath")
    math_greater.operation = "GREATER_THAN"
    math_greater.inputs[1].default_value = 0.95

    position = nodes.new("ShaderNodeNewGeometry")
    separate_pos = nodes.new("ShaderNodeSeparateXYZ")

    map_range = nodes.new("ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = 20
    map_range.inputs["From Max"].default_value = 0

    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].color = (0.2, 0.2, 0.2, 1)
    color_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)

    links.new(geometry.outputs["Normal"], separate_normal.inputs[0])
    links.new(separate_normal.outputs["X"], math_abs.inputs[0])
    links.new(math_abs.outputs[0], math_greater.inputs[0])
    links.new(math_greater.outputs[0], mix_shader.inputs["Fac"])

    links.new(position.outputs["Position"], separate_pos.inputs[0])
    links.new(separate_pos.outputs["X"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], emission_major.inputs["Color"])

    links.new(emission_major.outputs["Emission"], mix_shader.inputs[2])
    links.new(emission_other.outputs["Emission"], mix_shader.inputs[1])
    links.new(mix_shader.outputs[0], output.inputs["Surface"])

    # Create the material "MAT_Y" for the "CAM_RIGHT" camera.
    # Its feature is: Only targeting the faces with normal vectors along the Y axis,
    # assigning different grayscale values to the faces with different Y coordinates.

    mat = bpy.data.materials.new("MAT_Y")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")

    emission_major = nodes.new("ShaderNodeEmission")
    emission_other = nodes.new("ShaderNodeEmission")
    emission_other.inputs["Color"].default_value = (0, 0, 0, 1)

    mix_shader = nodes.new("ShaderNodeMixShader")

    geometry = nodes.new("ShaderNodeNewGeometry")
    separate_normal = nodes.new("ShaderNodeSeparateXYZ")

    math_abs = nodes.new("ShaderNodeMath")
    math_abs.operation = "ABSOLUTE"

    math_greater = nodes.new("ShaderNodeMath")
    math_greater.operation = "GREATER_THAN"
    math_greater.inputs[1].default_value = 0.95

    position = nodes.new("ShaderNodeNewGeometry")
    separate_pos = nodes.new("ShaderNodeSeparateXYZ")

    map_range = nodes.new("ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = 0
    map_range.inputs["From Max"].default_value = 20

    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].color = (0.2, 0.2, 0.2, 1)
    color_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)

    links.new(geometry.outputs["Normal"], separate_normal.inputs[0])
    links.new(separate_normal.outputs["Y"], math_abs.inputs[0])
    links.new(math_abs.outputs[0], math_greater.inputs[0])
    links.new(math_greater.outputs[0], mix_shader.inputs["Fac"])

    links.new(position.outputs["Position"], separate_pos.inputs[0])
    links.new(separate_pos.outputs["Y"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], emission_major.inputs["Color"])

    links.new(emission_major.outputs["Emission"], mix_shader.inputs[2])
    links.new(emission_other.outputs["Emission"], mix_shader.inputs[1])
    links.new(mix_shader.outputs[0], output.inputs["Surface"])

    # Create the material "MAT_Z" for the "CAM_TOP" camera.
    # Its feature is: Only targeting the faces with normal vectors along the Z axis,
    # assigning different grayscale values to the faces with different Z coordinates.

    mat = bpy.data.materials.new("MAT_Z")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")

    emission_major = nodes.new("ShaderNodeEmission")
    emission_other = nodes.new("ShaderNodeEmission")
    emission_other.inputs["Color"].default_value = (0, 0, 0, 1)

    mix_shader = nodes.new("ShaderNodeMixShader")

    geometry = nodes.new("ShaderNodeNewGeometry")
    separate_normal = nodes.new("ShaderNodeSeparateXYZ")

    math_abs = nodes.new("ShaderNodeMath")
    math_abs.operation = "ABSOLUTE"

    math_greater = nodes.new("ShaderNodeMath")
    math_greater.operation = "GREATER_THAN"
    math_greater.inputs[1].default_value = 0.95

    position = nodes.new("ShaderNodeNewGeometry")
    separate_pos = nodes.new("ShaderNodeSeparateXYZ")

    map_range = nodes.new("ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = 0
    map_range.inputs["From Max"].default_value = 20

    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].color = (0.2, 0.2, 0.2, 1)
    color_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)

    links.new(geometry.outputs["Normal"], separate_normal.inputs[0])
    links.new(separate_normal.outputs["Z"], math_abs.inputs[0])
    links.new(math_abs.outputs[0], math_greater.inputs[0])
    links.new(math_greater.outputs[0], mix_shader.inputs["Fac"])

    links.new(position.outputs["Position"], separate_pos.inputs[0])
    links.new(separate_pos.outputs["Z"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], emission_major.inputs["Color"])

    links.new(emission_major.outputs["Emission"], mix_shader.inputs[2])
    links.new(emission_other.outputs["Emission"], mix_shader.inputs[1])
    links.new(mix_shader.outputs[0], output.inputs["Surface"])

# ******************************************************

def clear_case(prefix: str = "MODEL") -> None:
    for obj in list(bpy.data.objects):
        if obj.name.startswith(prefix):
            bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh, do_unlink=True)


def create_cuboid(p1: List[int], p2: List[int], name: str | None = None) -> Any:
    p1_arr = np.array(p1)
    p2_arr = np.array(p2)

    center = (p1_arr + p2_arr) / 2
    length = np.abs(p2_arr - p1_arr)

    bpy.ops.mesh.primitive_cube_add(size=1, location=center.tolist())
    obj_create = bpy.context.active_object
    obj_create.scale = length.tolist()

    if name is not None:
        obj_create.name = name
    return obj_create


def build_case(
    levels: List[List[List[int]]],
    heights: List[int],
    prefix: str = "MODEL"
) -> None:
    # Must apply a consistent "prefix" name to all objects
    clear_case(prefix)

    for level_idx, level_corners in enumerate(levels):
        start_x, start_y = 0, 0
        start_z = 0 if level_idx == 0 else heights[level_idx - 1]

        for corner_idx, corner_point in enumerate(level_corners):
            obj_name = f"{prefix}_{level_idx}_{corner_idx}"
            create_cuboid(
                p1=[start_x, start_y, start_z],
                p2=[corner_point[0], corner_point[1], heights[level_idx]],
                name=obj_name,
            )
            start_x, start_y = corner_point[0], 0

# ******************************************************

def render_case(
    prefix: str, cam_name: str, mat_name: str, save_path: str
) -> None:
    mat_render  = bpy.data.materials.get(mat_name)
    for obj in list(bpy.data.objects):
        if obj.name.startswith(prefix):
            obj_render = obj
            if obj_render.data.materials:
                obj_render.data.materials[0] = mat_render
            else:
                obj_render.data.materials.append(mat_render)
    bpy.context.scene.camera = bpy.data.objects[cam_name]
    bpy.context.scene.render.filepath = save_path
    bpy.ops.render.render(write_still=True)


def render_case_by_views(
    views: Dict[str, List[str | None]],
    save_dir: str,
    case_name: str,
    prefix: str = "MODEL"
) -> None:
    # All views that need to render
    all_views = views["single"] + views["merge"]

    if "iso-a" in all_views:
        # ***********************************************
        # Render ISO-A-view image
        # *********************
        # Render ISO-A-X,
        # where X means colors change by X coordinates
        render_case(
            prefix=prefix,
            cam_name="CAM_ISOA",
            mat_name="MAT_X",
            save_path=os.path.join(save_dir, f"{case_name}-iso-a-x.jpg")
        )
        # *********************
        # Render ISO-A-Y,
        # where Y means colors change by Y coordinates
        render_case(
            prefix=prefix,
            cam_name="CAM_ISOA",
            mat_name="MAT_Y",
            save_path=os.path.join(save_dir, f"{case_name}-iso-a-y.jpg")
        )
        # *********************
        # Render ISO-A-Z,
        # where Z means colors change by Z coordinates
        render_case(
            prefix=prefix,
            cam_name="CAM_ISOA",
            mat_name="MAT_Z",
            save_path=os.path.join(save_dir, f"{case_name}-iso-a-z.jpg")
        )

    if "iso-b" in all_views:
        # ***********************************************
        # Render ISO-B-view image
        # *********************
        # Render ISO-B-X,
        # where X means colors change by X coordinates
        render_case(
            prefix=prefix,
            cam_name="CAM_ISOB",
            mat_name="MAT_X",
            save_path=os.path.join(save_dir, f"{case_name}-iso-b-x.jpg")
        )
        # *********************
        # Render ISO-B-Y,
        # where Y means colors change by Y coordinates
        render_case(
            prefix=prefix,
            cam_name="CAM_ISOB",
            mat_name="MAT_Y",
            save_path=os.path.join(save_dir, f"{case_name}-iso-b-y.jpg")
        )
        # *********************
        # Render ISO-B-Z,
        # where Z means colors change by Z coordinates
        render_case(
            prefix=prefix,
            cam_name="CAM_ISOB",
            mat_name="MAT_Z",
            save_path=os.path.join(save_dir, f"{case_name}-iso-b-z.jpg")
        )

    if "front" in all_views:
        # ***********************************************
        # Render FRONT-view image
        render_case(
            prefix=prefix,
            cam_name="CAM_FRONT",
            mat_name="MAT_X",
            save_path=os.path.join(save_dir, f"{case_name}-front.jpg")
        )

    if "right" in all_views:
        # ***********************************************
        # Render RIGHT-view image
        render_case(
            prefix=prefix,
            cam_name="CAM_RIGHT",
            mat_name="MAT_Y",
            save_path=os.path.join(save_dir, f"{case_name}-right.jpg")
        )

    if "top" in all_views:
        # ***********************************************
        # Render TOP-view image
        render_case(
            prefix=prefix,
            cam_name="CAM_TOP",
            mat_name="MAT_Z",
            save_path=os.path.join(save_dir, f"{case_name}-top.jpg")
        )

    if "iso" in all_views:
        # ***********************************************
        # Render ISO-view image
        render_case(
            prefix=prefix,
            cam_name="CAM_ISO",
            mat_name="MAT_ISO",
            save_path=os.path.join(save_dir, f"{case_name}-iso.jpg")
        )
