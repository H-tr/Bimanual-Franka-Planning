"""Compose a bimanual FR3 URDF from cricket's single-arm FR3 source.

Reads ``third_party/cricket/resources/fr3/fr3.urdf`` (and its already-
spherized sibling) and stamps two prefix-renamed copies (``fr3_left_*``
and ``fr3_right_*``) into one ``<robot name="bimanual_fr3">`` document.

Frame design (chosen so deployment / extrinsic calibration is simple):

  ``world`` is co-located with ``fr3_left_link0`` — the LEFT arm's base
  is the calibration anchor.  In the field you place the left arm
  wherever you want the world frame to be; no extrinsic to measure on
  that side.

  The right arm's base is connected to ``world`` through a 3-DOF planar
  chain (``relative_base_x`` → ``relative_base_y`` → ``relative_base_yaw``)
  so the only extrinsic you ever need to set is the *relative* pose of
  the right arm w.r.t. the left arm.  This pose is part of the planning
  state vector — pin it via ``base_config`` to plan single-arm motion at
  a known cell layout, or leave it active to let the planner search over
  it (e.g. for cell layout optimisation).

Joint order in the planner state vector (from URDF tree DFS):

    [0:7]    fr3_left_joint1..7              left arm
    [7:10]   relative_base_{x, y, yaw}       right-arm base relative to left
    [10:17]  fr3_right_joint1..7             right arm

Total: 17 actuated DOFs.

The two prismatic finger joints per arm are converted to ``fixed`` so the
gripper geometry stays attached for visualisation/collision but adds no
DOFs.

Run:
    python tools/build_bimanual_urdf.py

Outputs:
    assets/bimanual_fr3_description/urdf/{bimanual_fr3.urdf,bimanual_fr3.srdf}
    bimanual_franka_planning/resources/robot/bimanual_fr3/
        bimanual_fr3.urdf            # planning URDF (mesh collision)
        bimanual_fr3_spherized.urdf  # FK-gen input (spheres + visual mesh)
        bimanual_fr3_viz.urdf        # PyBullet GUI (visual only)
        bimanual_fr3.srdf            # collision-pair filters
        meshes/ + viz_meshes/        # collision + visual mesh copies
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

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

# Default relative pose for the right arm w.r.t. the world (= left arm
# base): displaced by 0.8 m along -y, no rotation.  Used as the URDF
# joint origin and as the home value of the relative_base joints.
RELATIVE_BASE_HOME_XYZ = (0.0, -0.8, 0.0)
RELATIVE_BASE_HOME_RPY = (0.0, 0.0, 0.0)

# Limits for the relative-base planar joints — wide enough to cover any
# realistic bimanual layout but bounded so planners don't wander.
RELATIVE_BASE_XY_LIMIT = 1.5  # m
RELATIVE_BASE_YAW_LIMIT = 3.14159265  # rad

# Virtual head-mounted camera at a fixed offset from the world frame
# (= left arm base).  The Pink CoM/camera tasks point at this frame.
CAMERA_XYZ = (0.5, -0.4, 0.6)  # roughly above the centre between the arms
CAMERA_RPY = (0.0, 0.3, 0.0)


SOURCE_PREFIX = "fr3_"


def _retarget_name(value: str, prefix: str, known: set[str]) -> str:
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
    """Apply ``prefix`` to every link/joint name and parent/child reference."""
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
    """Rewrite ``package://franka_description/meshes/...`` → ``package://meshes/...``."""
    for mesh in arm_root.iter("mesh"):
        fn = mesh.attrib.get("filename", "")
        if fn.startswith("package://franka_description/meshes/"):
            mesh.attrib["filename"] = fn.replace(
                "package://franka_description/meshes/",
                "package://meshes/",
                1,
            )


def _drop_finger_dof(arm_root: ET.Element) -> None:
    """Convert the two prismatic finger joints to fixed."""
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
    becomes a free chain rooted at ``fr3_link0``."""
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
    xyz: tuple[float, float, float] = (0, 0, 0),
    rpy: tuple[float, float, float] = (0, 0, 0),
) -> ET.Element:
    j = ET.Element("joint", attrib={"name": name, "type": "fixed"})
    j.append(_origin(xyz, rpy))
    p = ET.SubElement(j, "parent")
    p.set("link", parent)
    c = ET.SubElement(j, "child")
    c.set("link", child)
    return j


def _planar_joint(
    name: str,
    parent: str,
    child: str,
    axis: tuple[float, float, float],
    joint_type: str,
    lower: float,
    upper: float,
    home: float = 0.0,
    velocity: float = 1.0,
    effort: float = 100.0,
) -> ET.Element:
    j = ET.Element("joint", attrib={"name": name, "type": joint_type})
    # Origin: zero by default; the home offset is captured by the joint's
    # value at the home configuration so calibration stays in one place.
    j.append(_origin((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    p = ET.SubElement(j, "parent")
    p.set("link", parent)
    c = ET.SubElement(j, "child")
    c.set("link", child)
    a = ET.SubElement(j, "axis")
    a.set("xyz", " ".join(str(v) for v in axis))
    lim = ET.SubElement(j, "limit")
    lim.set("lower", str(lower))
    lim.set("upper", str(upper))
    lim.set("effort", str(effort))
    lim.set("velocity", str(velocity))
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

    Inherits every per-arm exclusion from ``fr3.srdf``.  Adds NO
    cross-arm exclusions: the two arms must collision-check against
    each other so the planner can actually keep them apart.
    """
    src = SRC_SRDF.read_text()

    def _side(prefix: str) -> str:
        out = src
        out = re.sub(r"\bfr3_link", f"{prefix}link", out)
        out = re.sub(r"\bfr3_joint", f"{prefix}joint", out)
        out = re.sub(r"\bfr3_finger_joint", f"{prefix}finger_joint", out)
        out = re.sub(r'group name="fr3_arm"', f'group name="{prefix}arm"', out)
        out = re.sub(r'group="fr3_arm"', f'group="{prefix}arm"', out)
        out = re.sub(r"^.*<\?xml[^?]*\?>\n", "", out, flags=re.MULTILINE)
        out = re.sub(r"<!--.*?-->", "", out, flags=re.DOTALL)
        out = re.sub(r'<robot name="fr3">', "", out)
        out = re.sub(r"</robot>\s*$", "", out)
        out = re.sub(r"<virtual_joint[^/]*/>\s*", "", out)
        return out.strip()

    left = _side("fr3_left_")
    right = _side("fr3_right_")

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


def _strip_collisions(root: ET.Element) -> None:
    """Remove every ``<collision>`` element under ``root`` (visualisation URDF only)."""
    for link in root.iter("link"):
        for col in list(link.findall("collision")):
            link.remove(col)


def _build_combined(
    src_urdf: Path,
    *,
    visuals_from: Path | None = None,
) -> ET.Element:
    """Compose a full bimanual URDF document.

    ``src_urdf`` supplies the link/joint/collision skeleton.  When
    ``visuals_from`` is also given (the spherized URDF case), every
    ``<visual>`` element is taken from that file instead — so the
    spherized URDF can keep both <sphere> collision geometry and the
    high-poly mesh visuals side-by-side, making it easy to compare
    collision proxy vs. visual model in a viewer.
    """
    bimanual = ET.Element("robot", attrib={"name": "bimanual_fr3"})

    # ── World (= LEFT arm's base) ─────────────────────────────────────
    bimanual.append(_empty_link("world"))
    bimanual.append(
        _fixed_joint("world_to_fr3_left_base", "world", "fr3_left_link0")
    )

    # ── Relative base chain: world → ... → fr3_right_link0 ────────────
    # Three serial 1-DOF joints (x prismatic → y prismatic → yaw revolute).
    # The home values capture the "default" right-arm placement; pin them
    # via ``base_config`` to plan only the arms at a fixed cell layout.
    bimanual.append(_empty_link("relative_base_x_link"))
    bimanual.append(
        _planar_joint(
            "relative_base_x", "world", "relative_base_x_link",
            axis=(1, 0, 0), joint_type="prismatic",
            lower=-RELATIVE_BASE_XY_LIMIT, upper=RELATIVE_BASE_XY_LIMIT,
            velocity=0.5,
        )
    )
    bimanual.append(_empty_link("relative_base_y_link"))
    bimanual.append(
        _planar_joint(
            "relative_base_y", "relative_base_x_link", "relative_base_y_link",
            axis=(0, 1, 0), joint_type="prismatic",
            lower=-RELATIVE_BASE_XY_LIMIT, upper=RELATIVE_BASE_XY_LIMIT,
            velocity=0.5,
        )
    )
    bimanual.append(
        _planar_joint(
            "relative_base_yaw", "relative_base_y_link", "fr3_right_link0",
            axis=(0, 0, 1), joint_type="revolute",
            lower=-RELATIVE_BASE_YAW_LIMIT, upper=RELATIVE_BASE_YAW_LIMIT,
            velocity=1.0,
        )
    )

    # ── Camera (fixed, world-anchored) ────────────────────────────────
    bimanual.append(_empty_link("world_camera"))
    bimanual.append(
        _fixed_joint(
            "world_to_camera", "world", "world_camera",
            CAMERA_XYZ, CAMERA_RPY,
        )
    )

    # ── Stamp the two arm copies ──────────────────────────────────────
    left = _arm_copy("fr3_left_", src_urdf)
    right = _arm_copy("fr3_right_", src_urdf)

    # If a separate visuals_from URDF is provided, replace the <visual>
    # children of every link in the skeleton with the corresponding ones
    # from that source. This is how the spherized URDF gets its visual
    # meshes back: skeleton from the spherized file (sphere collisions)
    # + visuals from the regular URDF (mesh visuals).
    if visuals_from is not None:
        _graft_visuals(left, _arm_copy("fr3_left_", visuals_from))
        _graft_visuals(right, _arm_copy("fr3_right_", visuals_from))

    _move_children(left, bimanual)
    _move_children(right, bimanual)

    return bimanual


def _graft_visuals(target: ET.Element, source: ET.Element) -> None:
    """Replace ``<visual>`` children of every link in ``target`` with the
    matching link's visuals from ``source``.  No-op for links that don't
    appear in both."""
    src_visuals: dict[str, list[ET.Element]] = {}
    for link in source.iter("link"):
        name = link.attrib.get("name")
        if name:
            src_visuals[name] = list(link.findall("visual"))

    for link in target.iter("link"):
        name = link.attrib.get("name", "")
        if name not in src_visuals:
            continue
        for vis in list(link.findall("visual")):
            link.remove(vis)
        for vis in src_visuals[name]:
            link.append(vis)


def _serialize(root: ET.Element) -> str:
    _indent(root)
    xml = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" ?>\n' + xml + "\n"


def _write_urdf(path: Path, root: ET.Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_serialize(root))
    print(f"wrote {path.relative_to(ROOT)}")


def _check_dof(root: ET.Element, expected: int) -> None:
    xml = _serialize(root)
    n_revolute = len(re.findall(r'<joint[^>]*type="revolute"', xml))
    n_prismatic = len(re.findall(r'<joint[^>]*type="prismatic"', xml))
    total = n_revolute + n_prismatic
    print(f"  active joints: {n_revolute} revolute + {n_prismatic} prismatic = {total} (expected {expected})")
    assert total == expected, f"expected {expected} active DOF, got {total}"


def main() -> None:
    # Assets URDF/SRDF (the source-of-truth that scripts/build_robot.sh consumes).
    main_urdf = _build_combined(SRC_URDF)
    _write_urdf(OUT_URDF, main_urdf)
    _check_dof(main_urdf, expected=17)  # 7 left + 3 base + 7 right

    OUT_SRDF.write_text(_build_srdf())
    print(f"wrote {OUT_SRDF.relative_to(ROOT)}")

    # Resource copies inside the Python package.
    main_urdf_pkg = _build_combined(SRC_URDF)
    _write_urdf(OUT_URDF_RESOURCES, main_urdf_pkg)

    # Spherized URDF: sphere collisions + mesh visuals so a viewer can
    # render both side-by-side for inspection.
    spherized = _build_combined(SRC_URDF_SPHERIZED, visuals_from=SRC_URDF)
    _write_urdf(OUT_URDF_SPHERIZED, spherized)

    # Visualisation URDF: mesh visuals only.
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

    # Sphere density check — should be ~55 per arm (matching the cricket
    # FR3 source).  We don't hard-fail because cricket may regenerate
    # spheres later; we just report.
    sph = re.findall(r"<sphere", _serialize(spherized))
    print(f"  collision spheres: {len(sph)} ({len(sph)//2} per arm; expected ~55)")


if __name__ == "__main__":
    main()
