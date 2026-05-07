"""Generate soft-gripper FR3 URDFs (single-arm + bimanual).

Reads the existing FR3 URDFs and produces parallel ``*_soft`` variants
where the Franka parallel-jaw fingers have been swapped for a pair of
custom soft-rubber fingers (``soft_gripper_finger.stl``).

The soft finger STL has its base at ``z=0`` and tip at ``z=-0.123 m``
in the mesh's own frame.  Mounting the mesh inside each finger link
with ``rpy = (pi, 0, 0)`` rotates 180° about X so that:

  • the finger tip extends along ``+Z`` (matching the Franka convention
    where each finger sticks out from the hand in ``+Z``);
  • the inner pinching surface points in ``+Y`` of the link frame,
    so the existing ``rpy = (0, 0, pi)`` rotation on
    ``fr3_finger_joint2`` still mirrors the right finger correctly.

A small ``±half_gap`` ``y`` offset on each finger joint pre-positions
the two fingers at a fixed open gap (no actuated DOF — the joints
remain ``fixed``, since the planner doesn't search over gripper
opening).

Outputs (idempotent — re-run any time):

  bimanual_franka_planning/resources/robot/single_fr3_soft/
    single_fr3_soft.urdf            # planning URDF (mesh collision)
    single_fr3_soft_spherized.urdf  # FK-gen input (sphere collision)
    single_fr3_soft.srdf            # collision-pair filters
    meshes/  viz_meshes/            # mesh copies (incl. soft finger)

  bimanual_franka_planning/resources/robot/bimanual_fr3_soft/
    bimanual_fr3_soft.urdf
    bimanual_fr3_soft_spherized.urdf
    bimanual_fr3_soft_viz.urdf
    bimanual_fr3_soft.srdf
    meshes/  viz_meshes/

  assets/bimanual_fr3_description/urdf/
    bimanual_fr3_soft.urdf
    bimanual_fr3_soft.srdf
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets" / "bimanual_fr3_description"
RES = ROOT / "bimanual_franka_planning" / "resources" / "robot"
SOFT_FINGER_STL = ROOT / "soft_gripper_finger.stl"

# ── Soft-finger geometry constants ────────────────────────────────────

# Half the gap between the two finger inner surfaces (m).  Fingers are
# 25.8 mm thick along Y, so a 50 mm gap (25 mm half-gap) puts the
# inner faces of each finger at hand y = ±0.025 and the outer faces
# at hand y = ±(0.025 + 0.0258) ≈ ±0.0508 — entirely on its own side
# of the centreline (no overlap across the gap).
HALF_GAP = 0.025

# Mounting offset of each finger on the hand body (z direction) —
# unchanged from the Franka original, so the hand mesh is reused.
FINGER_Z_OFFSET = 0.0584

# Pose of the soft-finger mesh inside its link.
#
#   The authored STL has cross-section 40 mm (x) × 25.8 mm (y) and a
#   123 mm long axis along -z.  We need:
#     • the long axis pointing **+z** (away from the hand body),
#     • the **wide face (40 mm)** to be the gripping pad — i.e.
#       parallel to the open/close direction (link y), so when the
#       gripper closes on an object, the broad rubber surface
#       contacts it (this is the whole point of a soft finger),
#     • the finger centred on x=0 with the inner pinching face at
#       link y=0 (so the joint y-offset directly equals the inner-
#       face position and the gap between fingers stays a clean
#       2·HALF_GAP).
#
#   Compose:
#     • Rx(π) flips -y/-z to +y/+z (extends in +z, material to +y),
#     • Rz(π/2) twists 90° around the finger's long axis so the wide
#       40-mm face goes from x to y (the gripping direction).
#   The xyz translation re-centres the resulting cross-section: the
#   90° twist puts material at link x ∈ [-0.0258, 0], so xyz x
#   compensates by +0.0129.
FINGER_MESH_RPY = "3.141592653589793 0 1.5707963267948966"
FINGER_MESH_XYZ = "0.0129 0 0"

# Sphere collision approximation of the rectangular soft finger
# (now 25.8 × 40 × 123 mm in link x/y/z after the twist).  After the
# visual/collision <origin> above, the finger material occupies (in
# link-local coordinates)
#   x ∈ [-0.0129, +0.0129], y ∈ [0, 0.04], z ∈ [0, 0.123]
# Six spheres along the spine envelope the volume conservatively for
# swept-collision checking.  Radius is the half-diagonal of the
# 25.8 × 40 mm cross-section (≈ 0.0238 m) rounded down a touch — at
# r=0.024 the spheres fully envelope the cross-section.
SOFT_FINGER_SPHERES: list[tuple[tuple[float, float, float], float]] = [
    ((0.0, 0.02, 0.014), 0.024),
    ((0.0, 0.02, 0.038), 0.024),
    ((0.0, 0.02, 0.062), 0.024),
    ((0.0, 0.02, 0.086), 0.024),
    ((0.0, 0.02, 0.108), 0.024),
    ((0.0, 0.02, 0.122), 0.020),
]

# Soft finger inertial — order-of-magnitude rubber finger; not
# planner-relevant but keeps the URDF complete for downstream sims.
# Center of mass placed at the geometric centroid in link frame.
FINGER_INERTIAL_XYZ = (0.0, 0.02, 0.06)
FINGER_INERTIAL_MASS = 0.05
FINGER_INERTIAL_IXX = 6.0e-5
FINGER_INERTIAL_IYY = 6.0e-5
FINGER_INERTIAL_IZZ = 2.0e-5


# ── XML utilities ─────────────────────────────────────────────────────


def _parse(path: Path) -> ET.ElementTree:
    return ET.parse(path)


def _serialize(tree: ET.ElementTree) -> str:
    root = tree.getroot()
    _indent(root)
    return '<?xml version="1.0" ?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def _indent(el: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(el):
        if not el.text or not el.text.strip():
            el.text = indent + "  "
        for child in el:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
        if not el.tail or not el.tail.strip():
            el.tail = indent
    elif level and (not el.tail or not el.tail.strip()):
        el.tail = indent


def _write_urdf(tree: ET.ElementTree, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_serialize(tree))
    print(f"wrote {path.relative_to(ROOT)}")


# ── Finger link / joint patching ──────────────────────────────────────


def _make_visual(name: str) -> ET.Element:
    visual = ET.Element("visual", attrib={"name": name})
    origin = ET.SubElement(visual, "origin")
    origin.set("rpy", FINGER_MESH_RPY)
    origin.set("xyz", FINGER_MESH_XYZ)
    geom = ET.SubElement(visual, "geometry")
    mesh = ET.SubElement(geom, "mesh")
    mesh.set("filename", "package://meshes/robot_ee/franka_hand_soft/visual/finger.stl")
    return visual


def _make_mesh_collision() -> ET.Element:
    col = ET.Element("collision")
    origin = ET.SubElement(col, "origin")
    origin.set("rpy", FINGER_MESH_RPY)
    origin.set("xyz", FINGER_MESH_XYZ)
    geom = ET.SubElement(col, "geometry")
    mesh = ET.SubElement(geom, "mesh")
    mesh.set(
        "filename", "package://meshes/robot_ee/franka_hand_soft/collision/finger.stl"
    )
    return col


def _make_sphere_collisions() -> list[ET.Element]:
    out = []
    for (cx, cy, cz), r in SOFT_FINGER_SPHERES:
        col = ET.Element("collision")
        geom = ET.SubElement(col, "geometry")
        sphere = ET.SubElement(geom, "sphere")
        sphere.set("radius", f"{r}")
        origin = ET.SubElement(col, "origin")
        origin.set("xyz", f"{cx} {cy} {cz}")
        origin.set("rpy", "0 0 0")
        out.append(col)
    return out


def _make_inertial() -> ET.Element:
    inertial = ET.Element("inertial")
    origin = ET.SubElement(inertial, "origin")
    origin.set("rpy", "0 0 0")
    origin.set("xyz", " ".join(str(v) for v in FINGER_INERTIAL_XYZ))
    mass = ET.SubElement(inertial, "mass")
    mass.set("value", str(FINGER_INERTIAL_MASS))
    inertia = ET.SubElement(inertial, "inertia")
    inertia.set("ixx", str(FINGER_INERTIAL_IXX))
    inertia.set("ixy", "0.0")
    inertia.set("ixz", "0.0")
    inertia.set("iyy", str(FINGER_INERTIAL_IYY))
    inertia.set("iyz", "0.0")
    inertia.set("izz", str(FINGER_INERTIAL_IZZ))
    return inertial


def _replace_finger_link(link: ET.Element, *, mode: str, visual_name: str) -> None:
    """Strip the existing finger geometry and stamp the soft-finger one.

    ``mode``:
      "mesh"   — visual mesh + mesh collision (planning/visualisation URDF)
      "sphere" — visual mesh + sphere collisions (FK-gen / spherized URDF)
      "viz"    — visual mesh, no collision at all (visualisation-only URDF)
    """
    # Drop everything inside the link.
    for child in list(link):
        link.remove(child)

    link.append(_make_visual(visual_name))

    if mode == "mesh":
        link.append(_make_mesh_collision())
    elif mode == "sphere":
        for sphere in _make_sphere_collisions():
            link.append(sphere)
    elif mode == "viz":
        pass
    else:
        raise ValueError(f"unknown mode {mode!r}")

    link.append(_make_inertial())


def _patch_finger_joint(joint: ET.Element, *, side: str) -> None:
    """Pre-position a fixed finger joint so the two soft fingers sit at
    a constant open gap.  ``side`` is "left" or "right" — Franka naming.

    Inside each finger link the soft mesh's material lives in
    ``link.y ∈ [0, +0.0258]`` (positive y, after the visual/collision
    rpy flip).  To keep the LEFT finger entirely on the +y side of the
    hand and the RIGHT finger entirely on the -y side (as in the
    standard Franka convention), we offset the joint origins:

      • left  joint: ``xyz="0 +HALF_GAP 0.0584"`` rpy=0 → material in
        hand.y ∈ [+HALF_GAP, +HALF_GAP+0.0258]
      • right joint: ``xyz="0 -HALF_GAP 0.0584"`` rpy=π about z →
        the rpy flip turns link.+y into hand.-y, putting material in
        hand.y ∈ [-HALF_GAP-0.0258, -HALF_GAP]

    Both fingertips therefore sit on their own side of the centreline
    with a 2·HALF_GAP gap between the inner pinching faces.
    """
    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin")
    if side == "left":
        origin.set("xyz", f"0 {HALF_GAP} {FINGER_Z_OFFSET}")
        origin.set("rpy", "0 0 0")
    elif side == "right":
        origin.set("xyz", f"0 -{HALF_GAP} {FINGER_Z_OFFSET}")
        origin.set("rpy", "0 0 3.141592653589793")
    else:
        raise ValueError(f"unknown side {side!r}")
    # Fingers stay fixed — no actuated gripper DOF in the planner.
    joint.set("type", "fixed")
    for tag in ("axis", "limit", "dynamics", "mimic", "safety_controller"):
        for child in list(joint.findall(tag)):
            joint.remove(child)


def _patch_finger_pair(
    root: ET.Element,
    *,
    left_link: str,
    right_link: str,
    left_joint: str,
    right_joint: str,
    visual_name_left: str,
    visual_name_right: str,
    mode: str,
) -> None:
    """Patch one (left, right) finger pair in ``root`` to use the soft finger."""
    by_name: dict[tuple[str, str], ET.Element] = {}
    for el in root.iter():
        if el.tag in ("link", "joint") and "name" in el.attrib:
            by_name[(el.tag, el.attrib["name"])] = el

    _replace_finger_link(by_name[("link", left_link)], mode=mode, visual_name=visual_name_left)
    _replace_finger_link(by_name[("link", right_link)], mode=mode, visual_name=visual_name_right)
    _patch_finger_joint(by_name[("joint", left_joint)], side="left")
    _patch_finger_joint(by_name[("joint", right_joint)], side="right")


# ── Mesh staging ──────────────────────────────────────────────────────


def _mirror_meshes(src_dir: Path, dst_dir: Path) -> None:
    """Copy ``src_dir`` (resource tree of an existing fr3 robot) into
    ``dst_dir`` and add the soft finger mesh under
    ``meshes/robot_ee/franka_hand_soft/`` (and ``viz_meshes/``)."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    for sub in ("meshes", "viz_meshes"):
        soft_visual = dst_dir / sub / "robot_ee" / "franka_hand_soft" / "visual"
        soft_collision = dst_dir / sub / "robot_ee" / "franka_hand_soft" / "collision"
        soft_visual.mkdir(parents=True, exist_ok=True)
        soft_collision.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOFT_FINGER_STL, soft_visual / "finger.stl")
        shutil.copy2(SOFT_FINGER_STL, soft_collision / "finger.stl")


# ── Single-arm soft FR3 ───────────────────────────────────────────────


def build_single_fr3_soft() -> None:
    src = RES / "single_fr3"
    dst = RES / "single_fr3_soft"
    _mirror_meshes(src, dst)

    # Drop the meshes/ subtree of the *plain* franka_hand_white finger
    # (we replaced it everywhere, no link references it now).  Leaves
    # the franka_hand_white/hand mesh in place since the soft variant
    # still uses the original Franka palm.
    for sub in ("meshes", "viz_meshes"):
        for finger_dir in (
            dst / sub / "robot_ee" / "franka_hand_white" / "visual",
            dst / sub / "robot_ee" / "franka_hand_white" / "collision",
        ):
            stale = finger_dir / "finger.dae"
            if stale.exists():
                stale.unlink()

    # Remove the original-named URDFs (we'll write _soft variants).
    for f in dst.glob("single_fr3*.urdf"):
        f.unlink()
    for f in dst.glob("single_fr3*.srdf"):
        f.unlink()

    def _patch_single(src_urdf: Path, dst_urdf: Path, mode: str) -> None:
        tree = _parse(src_urdf)
        _patch_finger_pair(
            tree.getroot(),
            left_link="fr3_leftfinger",
            right_link="fr3_rightfinger",
            left_joint="fr3_finger_joint1",
            right_joint="fr3_finger_joint2",
            visual_name_left="fr3_leftfinger_visual",
            visual_name_right="fr3_rightfinger_visual",
            mode=mode,
        )
        # Rename robot tag for clarity ("fr3_soft" instead of "fr3").
        tree.getroot().set("name", "fr3_soft")
        _write_urdf(tree, dst_urdf)

    _patch_single(src / "single_fr3.urdf", dst / "single_fr3_soft.urdf", mode="mesh")
    _patch_single(
        src / "single_fr3_spherized.urdf",
        dst / "single_fr3_soft_spherized.urdf",
        mode="sphere",
    )

    # SRDF: copy verbatim — collision-disable rules reference link names
    # we kept (fr3_leftfinger / fr3_rightfinger / fr3_hand) so no rewrite
    # is required.  Just rename the robot for cleanliness.
    srdf_src = (src / "single_fr3.srdf").read_text()
    srdf_dst = srdf_src.replace('<robot name="fr3">', '<robot name="fr3_soft">')
    (dst / "single_fr3_soft.srdf").write_text(srdf_dst)
    print(f"wrote {(dst / 'single_fr3_soft.srdf').relative_to(ROOT)}")


# ── Bimanual soft FR3 ─────────────────────────────────────────────────


def build_bimanual_fr3_soft() -> None:
    src = RES / "bimanual_fr3"
    dst = RES / "bimanual_fr3_soft"
    _mirror_meshes(src, dst)

    for sub in ("meshes", "viz_meshes"):
        for finger_dir in (
            dst / sub / "robot_ee" / "franka_hand_white" / "visual",
            dst / sub / "robot_ee" / "franka_hand_white" / "collision",
        ):
            stale = finger_dir / "finger.dae"
            if stale.exists():
                stale.unlink()

    for f in dst.glob("bimanual_fr3*.urdf"):
        f.unlink()
    for f in dst.glob("bimanual_fr3*.srdf"):
        f.unlink()

    def _patch_bimanual(src_urdf: Path, dst_urdf: Path, mode: str) -> None:
        tree = _parse(src_urdf)
        for prefix in ("fr3_left_", "fr3_right_"):
            _patch_finger_pair(
                tree.getroot(),
                left_link=f"{prefix}leftfinger",
                right_link=f"{prefix}rightfinger",
                left_joint=f"{prefix}finger_joint1",
                right_joint=f"{prefix}finger_joint2",
                visual_name_left=f"{prefix}leftfinger_visual",
                visual_name_right=f"{prefix}rightfinger_visual",
                mode=mode,
            )
        tree.getroot().set("name", "bimanual_fr3_soft")
        _write_urdf(tree, dst_urdf)

    _patch_bimanual(
        src / "bimanual_fr3.urdf", dst / "bimanual_fr3_soft.urdf", mode="mesh"
    )
    _patch_bimanual(
        src / "bimanual_fr3_spherized.urdf",
        dst / "bimanual_fr3_soft_spherized.urdf",
        mode="sphere",
    )
    _patch_bimanual(
        src / "bimanual_fr3_viz.urdf",
        dst / "bimanual_fr3_soft_viz.urdf",
        mode="viz",
    )

    srdf_src = (src / "bimanual_fr3.srdf").read_text()
    srdf_dst = srdf_src.replace(
        '<robot name="bimanual_fr3">', '<robot name="bimanual_fr3_soft">'
    )
    (dst / "bimanual_fr3_soft.srdf").write_text(srdf_dst)
    print(f"wrote {(dst / 'bimanual_fr3_soft.srdf').relative_to(ROOT)}")

    # Mirror the urdf+srdf into the assets/ tree too — keeps the asset
    # package layout symmetrical with the non-soft bimanual variant.
    asset_urdf = ASSETS / "urdf" / "bimanual_fr3_soft.urdf"
    asset_srdf = ASSETS / "urdf" / "bimanual_fr3_soft.srdf"
    asset_urdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dst / "bimanual_fr3_soft.urdf", asset_urdf)
    shutil.copy2(dst / "bimanual_fr3_soft.srdf", asset_srdf)
    print(f"wrote {asset_urdf.relative_to(ROOT)}")
    print(f"wrote {asset_srdf.relative_to(ROOT)}")


def main() -> None:
    if not SOFT_FINGER_STL.exists():
        raise FileNotFoundError(
            f"soft_gripper_finger.stl not found at {SOFT_FINGER_STL}; "
            "drop it in the repo root before running this tool."
        )
    build_single_fr3_soft()
    build_bimanual_fr3_soft()


if __name__ == "__main__":
    main()
