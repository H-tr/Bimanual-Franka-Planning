"""Compose a bimanual FR3 URDF from cricket's single-arm FR3 source.

Reads ``third_party/cricket/resources/fr3/fr3.urdf`` and stamps two
prefix-renamed copies (``fr3_left_*`` and ``fr3_right_*``) into one
``<robot name="bimanual_fr3">`` document with a common ``world`` root,
two fixed offset joints positioning each arm base, and a virtual
``world_camera`` frame between the arms.

The two prismatic finger joints are converted to ``fixed`` joints so
the planner sees exactly 7 actuated DOFs per arm = 14 total.

This script is run once during repo setup; the generated URDF is then
checked in and the build pipeline (decompose / spherize / cricket FK)
operates on it like any other robot description.

Run:
    python tools/build_bimanual_urdf.py

Outputs:
    assets/bimanual_fr3_description/urdf/bimanual_fr3.urdf
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
CRICKET_FR3 = ROOT / "third_party" / "cricket" / "resources" / "fr3"
SRC_URDF = CRICKET_FR3 / "fr3.urdf"
SRC_URDF_SPHERIZED = CRICKET_FR3 / "fr3_spherized.urdf"
SRC_SRDF = CRICKET_FR3 / "fr3.srdf"

ASSETS_OUT = ROOT / "assets" / "bimanual_fr3_description"
OUT_URDF = ASSETS_OUT / "urdf" / "bimanual_fr3.urdf"
OUT_SRDF = ASSETS_OUT / "urdf" / "bimanual_fr3.srdf"

# Resource copies (consumed by the planning Python package + cricket FK gen).
RESOURCES_OUT = ROOT / "bimanual_franka_planning" / "resources" / "robot" / "bimanual_fr3"
OUT_URDF_RESOURCES = RESOURCES_OUT / "bimanual_fr3.urdf"
OUT_URDF_SPHERIZED = RESOURCES_OUT / "bimanual_fr3_spherized.urdf"
OUT_URDF_VIZ = RESOURCES_OUT / "bimanual_fr3_viz.urdf"
OUT_SRDF_RESOURCES = RESOURCES_OUT / "bimanual_fr3.srdf"

# Bimanual cell geometry: two arms 0.8 m apart, both facing forward (+x).
LEFT_BASE_XYZ = (0.0, 0.4, 0.0)
LEFT_BASE_RPY = (0.0, 0.0, 0.0)
RIGHT_BASE_XYZ = (0.0, -0.4, 0.0)
RIGHT_BASE_RPY = (0.0, 0.0, 0.0)

# Virtual head-mounted camera between the arms.
CAMERA_XYZ = (0.5, 0.0, 0.6)
CAMERA_RPY = (0.0, 0.3, 0.0)


SOURCE_PREFIX = "fr3_"


def _retarget_name(value: str, prefix: str, known: set[str]) -> str:
    """Replace the source ``fr3_`` prefix with the side-specific one."""
    if value not in known:
        return value
    if value.startswith(SOURCE_PREFIX):
        return prefix + value[len(SOURCE_PREFIX):]
    return prefix + value


def _collect_link_joint_names(arm_root: ET.Element) -> tuple[set[str], set[str]]:
    links = {el.attrib["name"] for el in arm_root.findall("link") if "name" in el.attrib}
    joints = {
        el.attrib["name"] for el in arm_root.findall("joint") if "name" in el.attrib
    }
    return links, joints


def _rename(arm_root: ET.Element, prefix: str) -> None:
    """Replace the source ``fr3_`` prefix with ``prefix`` on every link/joint
    name and every parent/child/mimic reference."""
    links, joints = _collect_link_joint_names(arm_root)

    for el in arm_root.iter():
        if el.tag == "link" and "name" in el.attrib:
            el.attrib["name"] = _retarget_name(el.attrib["name"], prefix, links)
        if el.tag == "joint":
            if "name" in el.attrib:
                el.attrib["name"] = _retarget_name(el.attrib["name"], prefix, joints)
            parent = el.find("parent")
            if parent is not None and "link" in parent.attrib:
                parent.attrib["link"] = _retarget_name(
                    parent.attrib["link"], prefix, links
                )
            child = el.find("child")
            if child is not None and "link" in child.attrib:
                child.attrib["link"] = _retarget_name(
                    child.attrib["link"], prefix, links
                )
            mimic = el.find("mimic")
            if mimic is not None and "joint" in mimic.attrib:
                mimic.attrib["joint"] = _retarget_name(
                    mimic.attrib["joint"], prefix, joints
                )


def _retarget_meshes(arm_root: ET.Element) -> None:
    """Rewrite ``package://franka_description/meshes/...`` → ``package://meshes/...``.

    Matches the convention used by the resource URDFs in
    ``bimanual_franka_planning/resources/robot/bimanual_fr3/`` so the
    PyBullet env's package resolver only needs to walk up a few
    directories to find a ``meshes/`` folder — no ROS install required.
    """
    for mesh in arm_root.iter("mesh"):
        fn = mesh.attrib.get("filename", "")
        if fn.startswith("package://franka_description/meshes/"):
            mesh.attrib["filename"] = fn.replace(
                "package://franka_description/meshes/",
                "package://meshes/",
                1,
            )


def _drop_finger_dof(arm_root: ET.Element) -> None:
    """Convert the two prismatic finger joints to fixed.

    Keeps the gripper geometry (and its visual meshes) attached for
    rendering / collision against the environment, but removes the two
    extra DOFs from the planner's view.  After this step, every arm
    contributes exactly 7 actuated joints.
    """
    for joint in arm_root.iter("joint"):
        name = joint.attrib.get("name", "")
        if "finger_joint" in name:
            joint.attrib["type"] = "fixed"
            for tag in ("axis", "limit", "dynamics", "mimic", "safety_controller"):
                child = joint.find(tag)
                while child is not None:
                    joint.remove(child)
                    child = joint.find(tag)


def _strip_world_root(arm_root: ET.Element) -> None:
    """Remove the existing ``base`` link + ``fr3_base_joint`` so each arm
    becomes a free chain rooted at ``fr3_link0`` — we'll wire it under our
    own world frame next."""
    to_remove = []
    for el in arm_root:
        if el.tag == "link" and el.attrib.get("name") == "base":
            to_remove.append(el)
        if el.tag == "joint" and el.attrib.get("name") == "fr3_base_joint":
            to_remove.append(el)
    for el in to_remove:
        arm_root.remove(el)


def _arm_copy(prefix: str, src_urdf: Path = SRC_URDF) -> ET.Element:
    tree = ET.parse(src_urdf)
    arm_root = tree.getroot()
    _strip_world_root(arm_root)
    _drop_finger_dof(arm_root)
    _rename(arm_root, prefix)
    _retarget_meshes(arm_root)
    return arm_root


def _origin(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> ET.Element:
    el = ET.Element("origin")
    el.set("xyz", " ".join(str(v) for v in xyz))
    el.set("rpy", " ".join(str(v) for v in rpy))
    return el


def _fixed_joint(
    name: str,
    parent: str,
    child: str,
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> ET.Element:
    j = ET.Element("joint", attrib={"name": name, "type": "fixed"})
    j.append(_origin(xyz, rpy))
    p = ET.SubElement(j, "parent")
    p.set("link", parent)
    c = ET.SubElement(j, "child")
    c.set("link", child)
    return j


def _empty_link(name: str) -> ET.Element:
    return ET.Element("link", attrib={"name": name})


def _indent(el: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(el):
        if not el.text or not el.text.strip():
            el.text = indent + "  "
        for child in el:
            _indent(child, level + 1)
        if not el.tail or not el.tail.strip():
            el.tail = indent
        if not child.tail or not child.tail.strip():
            child.tail = indent
    elif level and (not el.tail or not el.tail.strip()):
        el.tail = indent


def _move_children(src: ET.Element, dst: ET.Element) -> None:
    for child in list(src):
        dst.append(child)


def _build_srdf() -> str:
    """Stamp the FR3 SRDF twice (left/right side) into one bimanual SRDF.

    The original SRDF defines:
      * one chain group ``fr3_arm`` (base→tip),
      * named group_states (``ready``, ``extended``),
      * adjacent / never-collide pair filters.

    For the bimanual cell we duplicate everything per side and add a top
    ``bimanual_fr3`` group containing both chains.  We do not declare any
    cross-arm collision exclusions — the two arms must collision-check
    against each other.
    """
    src = SRC_SRDF.read_text()

    def _side(prefix: str, side_name: str) -> str:
        out = src
        # Replace fr3_armN/fr3_jointN/fr3_linkN names with the side prefix.
        # Use a regex with a word-boundary at the front and known suffixes
        # so we don't accidentally rename "fr3_arm" → "fr3_left_arm" inside
        # another token.
        out = re.sub(r"\bfr3_link", f"{prefix}link", out)
        out = re.sub(r"\bfr3_joint", f"{prefix}joint", out)
        out = re.sub(r"\bfr3_finger_joint", f"{prefix}finger_joint", out)
        out = re.sub(r'group name="fr3_arm"', f'group name="{prefix}arm"', out)
        out = re.sub(r'group="fr3_arm"', f'group="{prefix}arm"', out)
        # Drop SRDF wrapper + virtual_joint — we'll write our own.
        out = re.sub(r"^.*<\?xml[^?]*\?>\n", "", out, flags=re.MULTILINE)
        out = re.sub(r"<!--.*?-->", "", out, flags=re.DOTALL)
        out = re.sub(r'<robot name="fr3">', "", out)
        out = re.sub(r"</robot>\s*$", "", out)
        out = re.sub(r"<virtual_joint[^/]*/>\s*", "", out)
        return out.strip()

    left = _side("fr3_left_", "left")
    right = _side("fr3_right_", "right")

    return (
        '<?xml version="1.0" ?>\n'
        '<robot name="bimanual_fr3">\n'
        '  <group name="bimanual_fr3">\n'
        '    <group name="fr3_left_arm"/>\n'
        '    <group name="fr3_right_arm"/>\n'
        '  </group>\n'
        '  <virtual_joint child_link="world" name="virtual_joint" parent_frame="world" type="fixed"/>\n'
        + left
        + "\n"
        + right
        + "\n</robot>\n"
    )


def _strip_visuals(root: ET.Element) -> None:
    """Remove every ``<visual>`` element under ``root`` (used for the
    spherized URDF — cricket's FK generator only consumes collision
    primitives so the visual meshes aren't needed and only bloat the file)."""
    for link in root.iter("link"):
        for vis in list(link.findall("visual")):
            link.remove(vis)


def _strip_collisions(root: ET.Element) -> None:
    """Remove every ``<collision>`` element under ``root`` (used for the
    visualisation URDF — PyBullet's GUI client doesn't need collision
    geometry and hides .stl meshes anyway)."""
    for link in root.iter("link"):
        for col in list(link.findall("collision")):
            link.remove(col)


def _build_combined(src_urdf: Path) -> ET.Element:
    """Compose a full bimanual URDF document from ``src_urdf``."""
    bimanual = ET.Element("robot", attrib={"name": "bimanual_fr3"})

    bimanual.append(_empty_link("world"))
    bimanual.append(
        _fixed_joint(
            "world_to_fr3_left_base", "world", "fr3_left_link0",
            LEFT_BASE_XYZ, LEFT_BASE_RPY,
        )
    )
    bimanual.append(
        _fixed_joint(
            "world_to_fr3_right_base", "world", "fr3_right_link0",
            RIGHT_BASE_XYZ, RIGHT_BASE_RPY,
        )
    )
    bimanual.append(_empty_link("world_camera"))
    bimanual.append(
        _fixed_joint(
            "world_to_camera", "world", "world_camera",
            CAMERA_XYZ, CAMERA_RPY,
        )
    )

    left = _arm_copy("fr3_left_", src_urdf)
    right = _arm_copy("fr3_right_", src_urdf)
    _move_children(left, bimanual)
    _move_children(right, bimanual)

    return bimanual


def _serialize(root: ET.Element) -> str:
    _indent(root)
    xml = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" ?>\n' + xml + "\n"


def _write_urdf(path: Path, root: ET.Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_serialize(root))
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    # Assets URDF/SRDF (the source-of-truth that scripts/build_robot.sh consumes).
    main_urdf = _build_combined(SRC_URDF)
    _write_urdf(OUT_URDF, main_urdf)

    revolute = re.findall(r'<joint[^>]*type="revolute"', _serialize(main_urdf))
    print(f"  revolute joints: {len(revolute)} (expected 14)")
    assert len(revolute) == 14

    OUT_SRDF.write_text(_build_srdf())
    print(f"wrote {OUT_SRDF.relative_to(ROOT)}")

    # Resource copies inside the Python package: same URDF + spherized +
    # viz variants, all under the package://bimanual_fr3_description/ mesh
    # path so PyBullet/Pinocchio can find them.
    main_urdf_pkg = _build_combined(SRC_URDF)
    _write_urdf(OUT_URDF_RESOURCES, main_urdf_pkg)

    spherized = _build_combined(SRC_URDF_SPHERIZED)
    _strip_visuals(spherized)
    _write_urdf(OUT_URDF_SPHERIZED, spherized)

    viz = _build_combined(SRC_URDF)
    _strip_collisions(viz)
    _write_urdf(OUT_URDF_VIZ, viz)

    OUT_SRDF_RESOURCES.parent.mkdir(parents=True, exist_ok=True)
    OUT_SRDF_RESOURCES.write_text(_build_srdf())
    print(f"wrote {OUT_SRDF_RESOURCES.relative_to(ROOT)}")

    # Mirror collision (.stl) and visualisation (.dae) meshes into the
    # package resources folder so a pip install ships everything the
    # planner and PyBullet need without depending on the assets/ tree.
    import shutil
    meshes_dst = RESOURCES_OUT / "meshes"
    viz_dst = RESOURCES_OUT / "viz_meshes"

    for dst in (meshes_dst, viz_dst):
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

    src_meshes = ASSETS_OUT / "meshes"
    if src_meshes.is_dir():
        for f in src_meshes.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(src_meshes)
            # collision .stl → meshes/, visual .dae → viz_meshes/
            if f.suffix.lower() in {".stl"}:
                target = meshes_dst / rel
            else:
                target = viz_dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)

    print(
        "wrote "
        f"{meshes_dst.relative_to(ROOT)}/ ({sum(1 for _ in meshes_dst.rglob('*') if _.is_file())} files), "
        f"{viz_dst.relative_to(ROOT)}/ ({sum(1 for _ in viz_dst.rglob('*') if _.is_file())} files)"
    )


if __name__ == "__main__":
    main()
