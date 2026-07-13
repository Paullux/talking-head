"""
Bake the armature-driven viseme/idle poses into MORPH TARGETS and export a
web-ready .glb (mesh + shape keys + textures, no skeleton needed at runtime).

  blender --background base.blend --python bake_visemes_glb.py -- out.glb

three.js then blends mesh.morphTargetInfluences["A".."X","blink",...] live.
"""
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viseme_poses import VISEMES, ALL_POSED_BONES  # noqa: E402

# extra idle morphs (same bones idle_motion.py drives)
LID_UP = ["lid.T.L", "lid.T.L.001", "lid.T.R", "lid.T.R.001"]
LID_LO = ["lid.B.L", "lid.B.L.001", "lid.B.R", "lid.B.R.001"]
BROW = ["brow.T.L", "brow.T.L.001", "brow.B.L", "brow.T.R", "brow.T.R.001", "brow.B.R"]

IDLE_MORPHS = {
    "blink": {**{b: (0, 0, -0.032) for b in LID_UP}, **{b: (0, 0, 0.015) for b in LID_LO}},
    "browUp": {b: (0, 0, 0.010) for b in BROW},
    "browFurrow": {"brow.T.L": (-0.004, 0, -0.007), "brow.T.R": (0.004, 0, -0.007),
                   "brow.T.L.001": (-0.003, 0, -0.005), "brow.T.R.001": (0.003, 0, -0.005)},
}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out_glb = argv[0]

    mesh = bpy.data.objects["model"]
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    arm_mod = next(m for m in mesh.modifiers if m.type == "ARMATURE")

    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.mode_set(mode="OBJECT")

    # Basis = rest mesh
    if not mesh.data.shape_keys:
        mesh.shape_key_add(name="Basis", from_mix=False)

    def local(pb, g):
        return pb.bone.matrix_local.to_3x3().inverted() @ Vector(g)

    def set_pose(pose):
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode="POSE")
        for pbn in arm.pose.bones:      # reset everything first
            pbn.location = (0, 0, 0)
        for bn, g in pose.items():
            pb = arm.pose.bones.get(bn)
            if pb:
                pb.location = local(pb, g)
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.objects.active = mesh

    def bake(name, pose):
        set_pose(pose)
        before = set(k.name for k in mesh.data.shape_keys.key_blocks)
        bpy.ops.object.modifier_apply_as_shapekey(keep_modifier=True, modifier=arm_mod.name)
        new = (set(k.name for k in mesh.data.shape_keys.key_blocks) - before).pop()
        mesh.data.shape_keys.key_blocks[new].name = name
        print("baked morph:", name)

    for vis in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        bake(vis, VISEMES[vis])
    for name, pose in IDLE_MORPHS.items():
        bake(name, pose)

    # back to rest, drop the skeleton -> pure morph-target mesh
    set_pose({})
    mesh.modifiers.remove(arm_mod)
    mesh.parent = None
    for k in mesh.data.shape_keys.key_blocks:
        k.value = 0.0

    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.export_scene.gltf(
        filepath=out_glb,
        export_format="GLB",
        use_selection=True,
        export_morph=True,
        export_morph_normal=False,
        export_apply=False,
        export_yup=True,
    )
    print("EXPORTED", out_glb)
    print("morphs:", [k.name for k in mesh.data.shape_keys.key_blocks])


if __name__ == "__main__":
    main()
