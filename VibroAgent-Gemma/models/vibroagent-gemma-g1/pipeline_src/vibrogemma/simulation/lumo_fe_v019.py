from __future__ import annotations

"""Pinned Abaqus-to-OpenSees translation for the official healthy LUMO model.

The source is the CC BY 3.0 ``LUMO_FEM_healthy`` resource published with DOI
10.25835/0027803.  This parser intentionally supports only the constructs used
by that exact file.  It is not a general Abaqus reader.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..data.lumo_geometry import (
    LUMO_DAMAGE_ELEVATIONS_M,
    LUMO_SENSOR_ELEVATIONS_M,
)


LUMO_FE_SCHEMA = "vibrogemma-lumo-fe-v0.19.0"
LUMO_ABAQUS_SOURCE_SHA256 = (
    "66cde0586d53ecc66c91c0c1de78b1b37cbae97b65a2b4039eb41973c58804d9"
)
LUMO_SOURCE_DOI = "10.25835/0027803"
_MERGE_TOLERANCE = 1.0e-7

# The official README locates the six reversible damage mechanisms inside
# diagonal-brace bays.  The Abaqus part repeats every three metres; values here
# are local part elevations bounding the corresponding three-brace bay.
_DAMAGE_BAYS = {
    1: ("Segment-3", 2.1, 2.5, LUMO_DAMAGE_ELEVATIONS_M["DAM1"]),
    2: ("Segment-3", 0.9, 1.3, LUMO_DAMAGE_ELEVATIONS_M["DAM2"]),
    3: ("Segment-2", 1.7, 2.1, LUMO_DAMAGE_ELEVATIONS_M["DAM3"]),
    4: ("Segment-2", 0.5, 0.9, LUMO_DAMAGE_ELEVATIONS_M["DAM4"]),
    5: ("Segment-1", 1.7, 2.1, LUMO_DAMAGE_ELEVATIONS_M["DAM5"]),
    6: ("Segment-1", 0.5, 0.9, LUMO_DAMAGE_ELEVATIONS_M["DAM6"]),
}

_SENSOR_HEIGHTS = LUMO_SENSOR_ELEVATIONS_M

_SUPPORTED_KEYWORDS = {
    "assembly",
    "beam section",
    "boundary",
    "density",
    "elastic",
    "element",
    "element output",
    "elset",
    "end assembly",
    "end instance",
    "end part",
    "end step",
    "frequency",
    "heading",
    "instance",
    "mass",
    "material",
    "node",
    "node output",
    "nset",
    "output",
    "part",
    "preprint",
    "restart",
    "step",
    "surface",
    "tie",
}


class LumoFEParseError(ValueError):
    """The pinned LUMO Abaqus source violates the supported contract."""


@dataclass(frozen=True)
class BeamSection:
    name: str
    abaqus_kind: str
    dimensions: tuple[float, ...]
    area: float
    iy: float
    iz: float
    torsion: float


@dataclass(frozen=True)
class LumoNode:
    tag: int
    coordinates: tuple[float, float, float]
    source_nodes: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class LumoBeam:
    tag: int
    node_i: int
    node_j: int
    section: str
    source_instance: str
    source_element: int
    orientation: tuple[float, float, float]


@dataclass(frozen=True)
class LumoDamageZone:
    index: int
    elevation_m: float
    members: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]

    @property
    def element_tags(self) -> tuple[int, ...]:
        return tuple(sorted(tag for member in self.members for tag in member))


@dataclass(frozen=True)
class LumoFEModel:
    schema: str
    source_sha256: str
    nodes: tuple[LumoNode, ...]
    beams: tuple[LumoBeam, ...]
    sections: tuple[BeamSection, ...]
    nodal_masses: tuple[tuple[int, float], ...]
    sensor_nodes: tuple[tuple[str, int], ...]
    base_nodes: tuple[int, ...]
    damage_zones: tuple[LumoDamageZone, ...]
    young_modulus: float
    poisson_ratio: float
    density: float
    unmerged_node_count: int

    def validate(self) -> None:
        if self.schema != LUMO_FE_SCHEMA or self.source_sha256 != LUMO_ABAQUS_SOURCE_SHA256:
            raise LumoFEParseError("LUMO FE model identity is invalid")
        if self.unmerged_node_count != 2541 or len(self.nodes) != 2535:
            raise LumoFEParseError("LUMO instance/tie node counts are invalid")
        if len(self.beams) != 2667:
            raise LumoFEParseError("LUMO beam count is invalid")
        section_counts = {
            name: sum(beam.section == name for beam in self.beams)
            for name in ("brace", "leg")
        }
        if section_counts != {"brace": 1791, "leg": 876}:
            raise LumoFEParseError(f"LUMO section coverage is invalid: {section_counts}")
        if [name for name, _ in self.sensor_nodes] != list(_SENSOR_HEIGHTS):
            raise LumoFEParseError("LUMO ML1..ML9 mapping is incomplete")
        if len(set(tag for _, tag in self.sensor_nodes)) != 9 or len(self.base_nodes) != 3:
            raise LumoFEParseError("LUMO sensor/base node identities are invalid")
        if [zone.index for zone in self.damage_zones] != list(range(1, 7)):
            raise LumoFEParseError("LUMO DAM1..DAM6 mapping is incomplete")
        for zone in self.damage_zones:
            if len(zone.members) != 3 or any(len(member) != 17 for member in zone.members):
                raise LumoFEParseError(f"DAM{zone.index} must contain three 17-element braces")
            if len(zone.element_tags) != 51:
                raise LumoFEParseError(f"DAM{zone.index} contains duplicate brace elements")
        if not all(math.isfinite(value) and value > 0.0 for value in (
            self.young_modulus,
            self.density,
        )) or not 0.0 < self.poisson_ratio < 0.5:
            raise LumoFEParseError("LUMO material properties are invalid")
        if not self.nodal_masses or any(
            not math.isfinite(value) or value <= 0.0 for _, value in self.nodal_masses
        ):
            raise LumoFEParseError("LUMO lumped masses are invalid")


@dataclass(frozen=True)
class LumoFEScales:
    leg_stiffness: float = 1.0
    brace_stiffness: float = 1.0
    distributed_mass: float = 1.0
    lumped_mass: float = 1.0

    def validate(self) -> None:
        values = (
            self.leg_stiffness,
            self.brace_stiffness,
            self.distributed_mass,
            self.lumped_mass,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("LUMO FE scale parameters must be finite and positive")


@dataclass(frozen=True)
class _Block:
    keyword: str
    options: tuple[tuple[str, str | None], ...]
    data: tuple[str, ...]
    line_number: int

    def option_map(self) -> dict[str, str | None]:
        return dict(self.options)


@dataclass(frozen=True)
class _Instance:
    name: str
    translation: tuple[float, float, float]
    axis_start: tuple[float, float, float]
    axis_end: tuple[float, float, float]
    angle_degrees: float


@dataclass(frozen=True)
class _SectionSource:
    element_set: str
    section: BeamSection
    orientation: tuple[float, float, float]


class _UnionFind:
    def __init__(self, values: Iterable[tuple[str, int]]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: tuple[str, int]) -> tuple[str, int]:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: tuple[str, int], second: tuple[str, int]) -> None:
        left, right = self.find(first), self.find(second)
        if left == right:
            return
        keep, replace = sorted((left, right), key=_source_node_key)
        self.parent[replace] = keep


def _source_node_key(value: tuple[str, int]) -> tuple[int, int]:
    match = re.fullmatch(r"Segment-([123])", value[0])
    if match is None:
        raise LumoFEParseError(f"unsupported LUMO instance name: {value[0]!r}")
    return int(match.group(1)), int(value[1])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokenize(text: str) -> list[_Block]:
    blocks: list[_Block] = []
    keyword: str | None = None
    options: tuple[tuple[str, str | None], ...] = ()
    data: list[str] = []
    start_line = 0

    def flush() -> None:
        if keyword is not None:
            blocks.append(_Block(keyword, options, tuple(data), start_line))

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            flush()
            pieces = [piece.strip() for piece in line[1:].split(",")]
            keyword = pieces[0].lower()
            if keyword not in _SUPPORTED_KEYWORDS:
                raise LumoFEParseError(
                    f"unsupported Abaqus keyword *{pieces[0]} at line {line_number}"
                )
            parsed: list[tuple[str, str | None]] = []
            for piece in pieces[1:]:
                if not piece:
                    continue
                if "=" in piece:
                    key, value = piece.split("=", 1)
                    parsed.append((key.strip().lower(), value.strip()))
                else:
                    parsed.append((piece.lower(), None))
            options = tuple(parsed)
            data = []
            start_line = line_number
        else:
            if keyword is None:
                raise LumoFEParseError(f"data before first Abaqus keyword at line {line_number}")
            data.append(line)
    flush()
    return blocks


def _csv(line: str) -> list[str]:
    return [value.strip() for value in line.split(",") if value.strip()]


def _floats(line: str, *, count: int | None = None) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in _csv(line))
    except ValueError as exc:
        raise LumoFEParseError(f"invalid numeric Abaqus data: {line!r}") from exc
    if count is not None and len(values) != count:
        raise LumoFEParseError(f"expected {count} numeric values, observed {line!r}")
    if not all(math.isfinite(value) for value in values):
        raise LumoFEParseError(f"non-finite Abaqus data: {line!r}")
    return values


def _integers(lines: Iterable[str]) -> set[int]:
    output: set[int] = set()
    for line in lines:
        try:
            output.update(int(value) for value in _csv(line))
        except ValueError as exc:
            raise LumoFEParseError(f"invalid integer Abaqus data: {line!r}") from exc
    return output


def _rotation(
    axis_start: tuple[float, float, float],
    axis_end: tuple[float, float, float],
    angle_degrees: float,
) -> tuple[tuple[float, float, float], ...]:
    axis = tuple(end - start for start, end in zip(axis_start, axis_end, strict=True))
    length = math.sqrt(sum(value * value for value in axis))
    if length <= 0.0:
        raise LumoFEParseError("Abaqus instance rotation axis has zero length")
    x, y, z = (value / length for value in axis)
    angle = math.radians(angle_degrees)
    c, s, one_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return (
        (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
        (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
        (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
    )


def _matvec(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(sum(row[index] * vector[index] for index in range(3)) for row in matrix)


def _transform_point(instance: _Instance, point: tuple[float, float, float]) -> tuple[float, float, float]:
    translated = tuple(
        value + offset for value, offset in zip(point, instance.translation, strict=True)
    )
    matrix = _rotation(instance.axis_start, instance.axis_end, instance.angle_degrees)
    relative = tuple(
        value - origin for value, origin in zip(translated, instance.axis_start, strict=True)
    )
    rotated = _matvec(matrix, relative)
    return tuple(
        value + origin for value, origin in zip(rotated, instance.axis_start, strict=True)
    )


def _transform_vector(instance: _Instance, vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return _matvec(_rotation(instance.axis_start, instance.axis_end, instance.angle_degrees), vector)


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second, strict=True)))


def _section(kind: str, dimensions: tuple[float, ...]) -> BeamSection:
    if kind == "CIRC" and len(dimensions) == 1:
        radius = dimensions[0]
        if radius <= 0.0:
            raise LumoFEParseError("brace radius must be positive")
        area = math.pi * radius**2
        inertia = math.pi * radius**4 / 4.0
        return BeamSection("brace", kind, dimensions, area, inertia, inertia, 2.0 * inertia)
    if kind == "PIPE" and len(dimensions) == 2:
        outer, thickness = dimensions
        inner = outer - thickness
        if inner <= 0.0 or thickness <= 0.0:
            raise LumoFEParseError("leg PIPE radii are invalid")
        area = math.pi * (outer**2 - inner**2)
        inertia = math.pi * (outer**4 - inner**4) / 4.0
        return BeamSection("leg", kind, dimensions, area, inertia, inertia, 2.0 * inertia)
    raise LumoFEParseError(f"unsupported LUMO beam section {kind} {dimensions}")


def load_lumo_fe_model(path: Path) -> LumoFEModel:
    """Parse and expand the exact published healthy LUMO Abaqus input file."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_sha256 = _sha256(path)
    if observed_sha256 != LUMO_ABAQUS_SOURCE_SHA256:
        raise LumoFEParseError(
            "LUMO Abaqus source hash mismatch: "
            f"expected={LUMO_ABAQUS_SOURCE_SHA256}, observed={observed_sha256}"
        )
    blocks = _tokenize(path.read_text(encoding="utf-8-sig"))

    context: str | None = None
    part_name: str | None = None
    part_nodes: dict[int, tuple[float, float, float]] = {}
    part_beams: dict[int, tuple[int, int]] = {}
    part_nsets: dict[str, set[int]] = {}
    part_elsets: dict[str, set[int]] = {}
    part_mass_nodes: dict[str, list[int]] = {}
    part_mass_values: dict[str, float] = {}
    section_sources: list[_SectionSource] = []
    instances: dict[str, _Instance] = {}
    assembly_nsets: dict[str, tuple[tuple[str, int], ...]] = {}
    surfaces: dict[str, str] = {}
    ties: list[tuple[str, str]] = []
    assembly_mass_nodes: dict[str, list[tuple[str, int]]] = {}
    assembly_mass_values: dict[str, float] = {}
    boundary_rows: list[list[str]] = []
    material_name: str | None = None
    density: float | None = None
    elastic: tuple[float, float] | None = None

    for block in blocks:
        options = block.option_map()
        keyword = block.keyword
        if keyword == "part":
            if context is not None or part_name is not None:
                raise LumoFEParseError("only one top-level Abaqus part is supported")
            part_name = str(options.get("name") or "")
            if part_name != "Segment":
                raise LumoFEParseError(f"unsupported LUMO part: {part_name!r}")
            context = "part"
            continue
        if keyword == "end part":
            if context != "part":
                raise LumoFEParseError("*End Part without active part")
            context = None
            continue
        if keyword == "assembly":
            if context is not None or str(options.get("name") or "") != "Assembly":
                raise LumoFEParseError("unsupported LUMO assembly declaration")
            context = "assembly"
            continue
        if keyword == "end assembly":
            if context != "assembly":
                raise LumoFEParseError("*End Assembly without active assembly")
            context = None
            continue

        if keyword == "node":
            if context != "part" or options:
                raise LumoFEParseError("only plain part-level *Node is supported")
            for line in block.data:
                values = _csv(line)
                if len(values) != 4:
                    raise LumoFEParseError(f"invalid LUMO node row: {line!r}")
                node = int(values[0])
                if node in part_nodes:
                    raise LumoFEParseError(f"duplicate LUMO part node {node}")
                coordinates = tuple(float(value) for value in values[1:])
                if not all(math.isfinite(value) for value in coordinates):
                    raise LumoFEParseError(f"non-finite LUMO node {node}")
                part_nodes[node] = coordinates  # type: ignore[assignment]
            continue

        if keyword == "element":
            element_type = str(options.get("type") or "").upper()
            element_set = str(options.get("elset") or "")
            if context == "part" and element_type == "B31" and not element_set:
                for line in block.data:
                    values = _csv(line)
                    if len(values) != 3:
                        raise LumoFEParseError(f"invalid B31 element row: {line!r}")
                    tag, node_i, node_j = map(int, values)
                    if tag in part_beams:
                        raise LumoFEParseError(f"duplicate LUMO B31 element {tag}")
                    part_beams[tag] = (node_i, node_j)
                continue
            if element_type != "MASS" or not element_set:
                raise LumoFEParseError(
                    f"unsupported *Element at line {block.line_number}: type={element_type!r}"
                )
            target = part_mass_nodes if context == "part" else assembly_mass_nodes
            if context not in {"part", "assembly"}:
                raise LumoFEParseError("MASS element outside part/assembly")
            target.setdefault(element_set.lower(), [])
            for line in block.data:
                values = _csv(line)
                if len(values) != 2:
                    raise LumoFEParseError(f"invalid MASS element row: {line!r}")
                if context == "part":
                    target[element_set.lower()].append(int(values[1]))  # type: ignore[arg-type]
                else:
                    match = re.fullmatch(r"(Segment-[123])\.(\d+)", values[1])
                    if match is None:
                        raise LumoFEParseError(f"invalid assembly MASS node: {values[1]!r}")
                    target[element_set.lower()].append((match.group(1), int(match.group(2))))  # type: ignore[arg-type]
            continue

        if keyword in {"nset", "elset"}:
            if "generate" in options:
                raise LumoFEParseError("generated Abaqus sets are not supported")
            set_key = keyword
            name = str(options.get(set_key) or "").lower()
            if not name:
                raise LumoFEParseError(f"*{keyword} lacks a name")
            if keyword == "elset":
                if context != "part" or "instance" in options:
                    raise LumoFEParseError("only part element sets are supported")
                part_elsets[name] = _integers(block.data)
            elif context == "part":
                if "instance" in options:
                    raise LumoFEParseError("part node set cannot name an instance")
                part_nsets[name] = _integers(block.data)
            elif context == "assembly":
                instance = str(options.get("instance") or "")
                if instance not in {"Segment-1", "Segment-2", "Segment-3"}:
                    raise LumoFEParseError(f"unsupported assembly node-set instance: {instance!r}")
                assembly_nsets[name] = tuple(
                    (instance, node) for node in sorted(_integers(block.data))
                )
            else:
                raise LumoFEParseError(f"*{keyword} outside part/assembly")
            continue

        if keyword == "beam section":
            if context != "part":
                raise LumoFEParseError("beam section outside the LUMO part")
            allowed = {"elset", "material", "poisson", "temperature", "section"}
            if set(options) - allowed:
                raise LumoFEParseError(f"unsupported beam-section options: {sorted(set(options) - allowed)}")
            if str(options.get("material") or "") != "steel" or len(block.data) != 2:
                raise LumoFEParseError("unsupported LUMO beam-section material/data")
            section_kind = str(options.get("section") or "").upper()
            section = _section(section_kind, _floats(block.data[0]))
            orientation = _floats(block.data[1], count=3)
            section_sources.append(
                _SectionSource(str(options.get("elset") or "").lower(), section, orientation)  # type: ignore[arg-type]
            )
            continue

        if keyword == "mass":
            element_set = str(options.get("elset") or "").lower()
            if not element_set or len(block.data) != 1:
                raise LumoFEParseError("invalid Abaqus *Mass declaration")
            value = _floats(block.data[0], count=1)[0]
            if value <= 0.0:
                raise LumoFEParseError("Abaqus lumped mass must be positive")
            if context == "part":
                part_mass_values[element_set] = value
            elif context == "assembly":
                assembly_mass_values[element_set] = value
            else:
                raise LumoFEParseError("*Mass outside part/assembly")
            continue

        if keyword == "instance":
            if context != "assembly" or str(options.get("part") or "") != "Segment":
                raise LumoFEParseError("unsupported LUMO instance")
            name = str(options.get("name") or "")
            if name in instances or len(block.data) != 2:
                raise LumoFEParseError(f"invalid or duplicate LUMO instance: {name!r}")
            translation = _floats(block.data[0], count=3)
            rotation_data = _floats(block.data[1], count=7)
            instances[name] = _Instance(
                name,
                translation,  # type: ignore[arg-type]
                rotation_data[:3],  # type: ignore[arg-type]
                rotation_data[3:6],  # type: ignore[arg-type]
                rotation_data[6],
            )
            continue

        if keyword == "surface":
            if context != "assembly" or str(options.get("type") or "").upper() != "NODE":
                raise LumoFEParseError("only assembly node surfaces are supported")
            if len(block.data) != 1 or len(_csv(block.data[0])) != 2:
                raise LumoFEParseError("invalid LUMO node surface")
            surfaces[str(options.get("name") or "").lower()] = _csv(block.data[0])[0].lower()
            continue

        if keyword == "tie":
            if context != "assembly" or str(options.get("adjust") or "").lower() != "yes":
                raise LumoFEParseError("only adjusted LUMO assembly ties are supported")
            if len(block.data) != 1 or len(_csv(block.data[0])) != 2:
                raise LumoFEParseError("invalid LUMO tie declaration")
            slave, master = _csv(block.data[0])
            ties.append((slave.lower(), master.lower()))
            continue

        if keyword == "material":
            if context is not None or material_name is not None:
                raise LumoFEParseError("unsupported or duplicate LUMO material")
            material_name = str(options.get("name") or "")
            if material_name != "steel":
                raise LumoFEParseError(f"unsupported LUMO material: {material_name!r}")
            continue
        if keyword == "density":
            if context is not None or len(block.data) != 1:
                raise LumoFEParseError("invalid LUMO density")
            density = _floats(block.data[0], count=1)[0]
            continue
        if keyword == "elastic":
            if context is not None or len(block.data) != 1:
                raise LumoFEParseError("invalid LUMO elastic material")
            values = _floats(block.data[0], count=2)
            elastic = values[0], values[1]
            continue
        if keyword == "boundary":
            if context is not None or options:
                raise LumoFEParseError("only plain top-level LUMO boundary rows are supported")
            boundary_rows.extend(_csv(line) for line in block.data)
            continue

        # These source-analysis controls are recognized but intentionally not
        # translated; the emitted Tcl defines its own deterministic eigen step.
        if keyword in {
            "heading", "preprint", "end instance", "step", "frequency", "restart",
            "output", "node output", "element output", "end step",
        }:
            continue
        raise LumoFEParseError(f"unhandled Abaqus keyword *{keyword}")

    if context is not None or part_name != "Segment":
        raise LumoFEParseError("unterminated or missing LUMO part/assembly")
    if len(part_nodes) != 847 or len(part_beams) != 889:
        raise LumoFEParseError(
            f"unexpected LUMO part topology: nodes={len(part_nodes)}, beams={len(part_beams)}"
        )
    if set(instances) != {"Segment-1", "Segment-2", "Segment-3"}:
        raise LumoFEParseError(f"unexpected LUMO instances: {sorted(instances)}")
    if density is None or elastic is None:
        raise LumoFEParseError("LUMO steel density/elastic properties are missing")
    if set(part_mass_nodes) != set(part_mass_values):
        raise LumoFEParseError("LUMO part MASS sets/values disagree")
    if set(assembly_mass_nodes) != set(assembly_mass_values):
        raise LumoFEParseError("LUMO assembly MASS sets/values disagree")

    section_by_element: dict[int, _SectionSource] = {}
    for source in section_sources:
        elements = part_elsets.get(source.element_set)
        if not elements:
            raise LumoFEParseError(f"beam section references empty set {source.element_set!r}")
        for element in elements:
            if element in section_by_element:
                raise LumoFEParseError(f"B31 element {element} has multiple sections")
            section_by_element[element] = source
    if set(section_by_element) != set(part_beams):
        raise LumoFEParseError("LUMO B31 section assignment is incomplete")

    raw_coordinates: dict[tuple[str, int], tuple[float, float, float]] = {}
    instance_orientation: dict[str, tuple[float, float, float]] = {}
    for name in sorted(instances, key=lambda value: int(value.rsplit("-", 1)[-1])):
        instance = instances[name]
        for node, coordinates in part_nodes.items():
            raw_coordinates[(name, node)] = _transform_point(instance, coordinates)
        orientation = _transform_vector(instance, section_sources[0].orientation)
        length = math.sqrt(sum(value * value for value in orientation))
        instance_orientation[name] = tuple(value / length for value in orientation)

    union = _UnionFind(raw_coordinates)
    for slave_surface, master_surface in ties:
        if slave_surface not in surfaces or master_surface not in surfaces:
            raise LumoFEParseError("LUMO tie references an unknown node surface")
        slave_nodes = list(assembly_nsets.get(surfaces[slave_surface], ()))
        master_nodes = list(assembly_nsets.get(surfaces[master_surface], ()))
        if len(slave_nodes) != len(master_nodes) or not slave_nodes:
            raise LumoFEParseError("LUMO tie node-set cardinalities disagree")
        available = set(master_nodes)
        for slave in slave_nodes:
            distances = sorted(
                (_distance(raw_coordinates[slave], raw_coordinates[master]), master)
                for master in available
            )
            if not distances or distances[0][0] > _MERGE_TOLERANCE:
                raise LumoFEParseError("LUMO tie nodes are not coincident")
            master = distances[0][1]
            available.remove(master)
            union.union(slave, master)

    buckets: dict[tuple[int, int, int], list[tuple[str, int]]] = {}
    for source, coordinates in raw_coordinates.items():
        key = tuple(round(value / _MERGE_TOLERANCE) for value in coordinates)
        buckets.setdefault(key, []).append(source)  # type: ignore[arg-type]
    for candidates in buckets.values():
        for index, first in enumerate(candidates):
            for second in candidates[index + 1 :]:
                if first[0] == second[0]:
                    continue
                if _distance(raw_coordinates[first], raw_coordinates[second]) <= _MERGE_TOLERANCE:
                    union.union(first, second)

    groups: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for source in raw_coordinates:
        groups.setdefault(union.find(source), []).append(source)
    ordered_groups = sorted(
        (tuple(sorted(values, key=_source_node_key)) for values in groups.values()),
        key=lambda values: _source_node_key(values[0]),
    )
    if len(raw_coordinates) - len(ordered_groups) != 6:
        raise LumoFEParseError("LUMO interfaces must merge exactly six node pairs")
    global_node_for_source: dict[tuple[str, int], int] = {}
    nodes: list[LumoNode] = []
    for tag, source_nodes in enumerate(ordered_groups, start=1):
        coordinates = raw_coordinates[source_nodes[0]]
        nodes.append(LumoNode(tag, coordinates, source_nodes))
        for source in source_nodes:
            global_node_for_source[source] = tag

    beams: list[LumoBeam] = []
    beam_tag_for_source: dict[tuple[str, int], int] = {}
    for instance_name in sorted(instances, key=lambda value: int(value.rsplit("-", 1)[-1])):
        for source_element, (local_i, local_j) in sorted(part_beams.items()):
            node_i = global_node_for_source[(instance_name, local_i)]
            node_j = global_node_for_source[(instance_name, local_j)]
            if node_i == node_j:
                raise LumoFEParseError(f"merged B31 element became zero-length: {instance_name}.{source_element}")
            source = section_by_element[source_element]
            tag = len(beams) + 1
            beam_tag_for_source[(instance_name, source_element)] = tag
            beams.append(
                LumoBeam(
                    tag,
                    node_i,
                    node_j,
                    source.section.name,
                    instance_name,
                    source_element,
                    instance_orientation[instance_name],
                )
            )

    mass_by_node: dict[int, float] = {}
    for instance_name in instances:
        for element_set, local_nodes in part_mass_nodes.items():
            value = part_mass_values[element_set]
            for local_node in local_nodes:  # type: ignore[assignment]
                tag = global_node_for_source[(instance_name, int(local_node))]
                mass_by_node[tag] = mass_by_node.get(tag, 0.0) + value
    for element_set, source_nodes in assembly_mass_nodes.items():
        value = assembly_mass_values[element_set]
        for source_node in source_nodes:  # type: ignore[assignment]
            tag = global_node_for_source[source_node]
            mass_by_node[tag] = mass_by_node.get(tag, 0.0) + value

    sensor_nodes: list[tuple[str, int]] = []
    node_by_tag = {node.tag: node for node in nodes}
    for sensor, expected_height in _SENSOR_HEIGHTS.items():
        sources = assembly_nsets.get(sensor.lower(), ())
        if len(sources) != 1:
            raise LumoFEParseError(f"{sensor} must resolve to exactly one Abaqus node")
        tag = global_node_for_source[sources[0]]
        if abs(node_by_tag[tag].coordinates[2] - expected_height) > 1.0e-6:
            raise LumoFEParseError(f"{sensor} height is inconsistent with the official README")
        sensor_nodes.append((sensor, tag))

    fixed_dofs: dict[tuple[str, int], set[int]] = {}
    for values in boundary_rows:
        if len(values) != 3:
            raise LumoFEParseError(f"unsupported LUMO boundary row: {values}")
        set_name, first, last = values
        source_nodes = assembly_nsets.get(set_name.lower())
        if not source_nodes:
            raise LumoFEParseError(f"boundary references unknown node set {set_name!r}")
        start, stop = int(first), int(last)
        if start < 1 or stop > 6 or start > stop:
            raise LumoFEParseError(f"invalid LUMO boundary DOFs: {values}")
        for source_node in source_nodes:
            fixed_dofs.setdefault(source_node, set()).update(range(start, stop + 1))
    fixed_sources = sorted(
        (source for source, dofs in fixed_dofs.items() if dofs == set(range(1, 7))),
        key=_source_node_key,
    )
    if len(fixed_sources) != 3 or any(dofs != set(range(1, 7)) for dofs in fixed_dofs.values()):
        raise LumoFEParseError("LUMO base constraints must fix three nodes in all six DOFs")
    base_nodes = tuple(sorted(global_node_for_source[source] for source in fixed_sources))

    damage_zones: list[LumoDamageZone] = []
    for zone, (instance_name, lower, upper, elevation) in _DAMAGE_BAYS.items():
        selected: list[int] = []
        for local_element, (local_i, local_j) in part_beams.items():
            if section_by_element[local_element].section.name != "brace":
                continue
            y_i, y_j = part_nodes[local_i][1], part_nodes[local_j][1]
            if (
                min(y_i, y_j) >= lower - 1.0e-6
                and max(y_i, y_j) <= upper + 1.0e-6
                and abs(y_i - y_j) > 1.0e-9
            ):
                selected.append(beam_tag_for_source[(instance_name, local_element)])
        adjacency: dict[int, list[tuple[int, int]]] = {}
        beam_by_tag = {beam.tag: beam for beam in beams}
        for tag in selected:
            beam = beam_by_tag[tag]
            adjacency.setdefault(beam.node_i, []).append((beam.node_j, tag))
            adjacency.setdefault(beam.node_j, []).append((beam.node_i, tag))
        unseen = set(adjacency)
        members: list[tuple[int, ...]] = []
        while unseen:
            stack = [min(unseen)]
            component_nodes: set[int] = set()
            component_elements: set[int] = set()
            while stack:
                node = stack.pop()
                if node in component_nodes:
                    continue
                component_nodes.add(node)
                unseen.discard(node)
                for neighbor, element in adjacency[node]:
                    component_elements.add(element)
                    stack.append(neighbor)
            members.append(tuple(sorted(component_elements)))
        def member_key(values: tuple[int, ...]) -> tuple[float, int]:
            degrees: dict[int, int] = {}
            for element in values:
                beam = beam_by_tag[element]
                degrees[beam.node_i] = degrees.get(beam.node_i, 0) + 1
                degrees[beam.node_j] = degrees.get(beam.node_j, 0) + 1
            endpoints = [node_by_tag[node].coordinates for node, degree in degrees.items() if degree == 1]
            if len(endpoints) != 2:
                raise LumoFEParseError(f"DAM{zone} brace is not one continuous member")
            midpoint_x = 0.5 * (endpoints[0][0] + endpoints[1][0])
            midpoint_y = 0.5 * (endpoints[0][1] + endpoints[1][1])
            angle = math.atan2(midpoint_y, midpoint_x)
            if midpoint_x < 0.0 and abs(midpoint_y) < 1.0e-6:
                angle = -math.pi
            return angle, values[0]

        # The publisher's 010/111 code indexes the three physical brace faces,
        # not Abaqus element numbers.  Fixed azimuth ordering keeps the middle
        # 010 member on one physical face at every DAM elevation.
        members.sort(key=member_key)
        if len(members) != 3 or any(len(member) != 17 for member in members):
            raise LumoFEParseError(
                f"DAM{zone} does not resolve to three 17-element diagonal braces"
            )
        damage_zones.append(LumoDamageZone(zone, elevation, tuple(members)))  # type: ignore[arg-type]

    unique_sections = {source.section.name: source.section for source in section_sources}
    model = LumoFEModel(
        schema=LUMO_FE_SCHEMA,
        source_sha256=observed_sha256,
        nodes=tuple(nodes),
        beams=tuple(beams),
        sections=tuple(unique_sections[name] for name in ("brace", "leg")),
        nodal_masses=tuple(sorted(mass_by_node.items())),
        sensor_nodes=tuple(sensor_nodes),
        base_nodes=base_nodes,
        damage_zones=tuple(damage_zones),
        young_modulus=elastic[0],
        poisson_ratio=elastic[1],
        density=density,
        unmerged_node_count=len(raw_coordinates),
    )
    model.validate()
    return model


def _number(value: float) -> str:
    if abs(value) < 5.0e-18:
        value = 0.0
    return f"{float(value):.17g}"


def build_lumo_eigen_tcl(
    model: LumoFEModel,
    *,
    scales: LumoFEScales = LumoFEScales(),
    damage_zone: int | None = None,
    damage_pattern: str | None = None,
    damage_stiffness_scale: float = 1.0,
    mode_count: int = 30,
    modal_report_path: Path | None = None,
) -> str:
    """Emit a deterministic OpenSees 3D eigen-analysis Tcl script.

    Pattern ``111`` scales all three diagonal braces at the selected DAM level;
    official pattern ``010`` scales only the deterministic middle member.  The
    stiffness scale must stay positive so the retained mesh does not introduce
    orphaned zero-mass nodes.
    """

    model.validate()
    scales.validate()
    if damage_zone is not None and damage_zone not in set(range(1, 7)):
        raise ValueError("damage_zone must be None or one of 1..6")
    if damage_zone is None and damage_pattern is not None:
        raise ValueError("healthy LUMO FE state cannot declare a damage_pattern")
    if damage_zone is not None and damage_pattern not in {"010", "111"}:
        raise ValueError("damaged LUMO FE state requires damage_pattern='010' or '111'")
    if not math.isfinite(damage_stiffness_scale) or not 0.0 < damage_stiffness_scale <= 1.0:
        raise ValueError("damage_stiffness_scale must be finite and in (0, 1]")
    if damage_zone is None and damage_stiffness_scale != 1.0:
        raise ValueError("damage_stiffness_scale requires a selected damage_zone")
    if not isinstance(mode_count, int) or isinstance(mode_count, bool) or not 1 <= mode_count <= 100:
        raise ValueError("mode_count must be an integer in [1, 100]")
    if modal_report_path is not None and '"' in str(modal_report_path):
        raise ValueError("modal_report_path cannot contain a quote")

    section_by_name = {section.name: section for section in model.sections}
    if damage_zone is None:
        damaged_elements: set[int] = set()
    else:
        zone = model.damage_zones[damage_zone - 1]
        damaged_elements = (
            set(zone.members[1]) if damage_pattern == "010" else set(zone.element_tags)
        )
    orientation = model.beams[0].orientation
    if any(_distance(beam.orientation, orientation) > 1.0e-12 for beam in model.beams):
        raise ValueError("the pinned LUMO instances must share one beam orientation")
    orientation = tuple(0.0 if abs(value) < 1.0e-12 else value for value in orientation)
    node_coordinates = {node.tag: node.coordinates for node in model.nodes}
    for beam in model.beams:
        first, second = node_coordinates[beam.node_i], node_coordinates[beam.node_j]
        direction = tuple(right - left for left, right in zip(first, second, strict=True))
        length = math.sqrt(sum(value * value for value in direction))
        cosine = abs(sum(
            direction[index] * orientation[index] for index in range(3)
        )) / length
        if cosine >= 1.0 - 1.0e-10:
            raise ValueError(f"beam {beam.tag} is parallel to the Abaqus section orientation")

    lines = [
        f"# {LUMO_FE_SCHEMA}",
        f"# source_sha256 {model.source_sha256}",
        f"# source_doi {LUMO_SOURCE_DOI}",
        "# generated deterministically from the pinned healthy Abaqus model",
        f"# scales {json.dumps(scales.__dict__, sort_keys=True, separators=(',', ':'))}",
        f"# damage_zone {damage_zone if damage_zone is not None else 'healthy'}",
        f"# damage_pattern {damage_pattern if damage_pattern is not None else 'healthy'}",
        f"# damage_stiffness_scale {_number(damage_stiffness_scale)}",
        "wipe",
        "model BasicBuilder -ndm 3 -ndf 6",
    ]
    for node in model.nodes:
        x, y, z = node.coordinates
        lines.append(f"node {node.tag} {_number(x)} {_number(y)} {_number(z)}")
    for tag, mass in model.nodal_masses:
        value = mass * scales.lumped_mass
        lines.append(
            f"mass {tag} {_number(value)} {_number(value)} {_number(value)} 0 0 0"
        )
    for tag in model.base_nodes:
        lines.append(f"fix {tag} 1 1 1 1 1 1")
    lines.append(
        "geomTransf Linear 1 " + " ".join(_number(value) for value in orientation)
    )
    for beam in model.beams:
        section = section_by_name[beam.section]
        stiffness = (
            scales.brace_stiffness if beam.section == "brace" else scales.leg_stiffness
        )
        if beam.tag in damaged_elements:
            stiffness *= damage_stiffness_scale
        young = model.young_modulus * stiffness
        shear = young / (2.0 * (1.0 + model.poisson_ratio))
        mass_per_length = model.density * section.area * scales.distributed_mass
        lines.append(
            "element elasticBeamColumn "
            f"{beam.tag} {beam.node_i} {beam.node_j} "
            f"{_number(section.area)} {_number(young)} {_number(shear)} "
            f"{_number(section.torsion)} {_number(section.iy)} {_number(section.iz)} 1 "
            f"-mass {_number(mass_per_length)}"
        )
    lines.extend(
        [
            "constraints Plain",
            "numberer RCM",
            "system BandGeneral",
            f"set lambda [eigen -genBandArpack {mode_count}]",
            *(
                [f'modalProperties -file "{Path(modal_report_path).as_posix()}"']
                if modal_report_path is not None
                else []
            ),
            "set pi [expr {acos(-1.0)}]",
            "set mode 1",
            "foreach eigenvalue $lambda {",
            "    if {$eigenvalue <= 0.0} {",
            '        puts stderr "VIBROGEMMA_EIGEN_ERROR=nonpositive"',
            "        exit 2",
            "    }",
            "    set omega [expr {sqrt($eigenvalue)}]",
            "    set frequency [expr {$omega / (2.0 * $pi)}]",
            '    puts [format "VIBROGEMMA_EIGEN=%d %.17g %.17g" $mode $eigenvalue $frequency]',
        ]
    )
    for sensor, tag in model.sensor_nodes:
        lines.append(
            f'    puts [format "VIBROGEMMA_MODE=%d {sensor} %.17g %.17g %.17g" '
            f'$mode [nodeEigenvector {tag} $mode 1] [nodeEigenvector {tag} $mode 2] '
            f'[nodeEigenvector {tag} $mode 3]]'
        )
    lines.extend(
        [
            "    incr mode",
            "}",
            'puts stdout "VIBROGEMMA_EIGEN_COMPLETE=[expr {$mode - 1}]"',
            "wipe",
            "exit",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "BeamSection",
    "LUMO_ABAQUS_SOURCE_SHA256",
    "LUMO_FE_SCHEMA",
    "LUMO_SOURCE_DOI",
    "LumoBeam",
    "LumoDamageZone",
    "LumoFEModel",
    "LumoFEParseError",
    "LumoFEScales",
    "LumoNode",
    "build_lumo_eigen_tcl",
    "load_lumo_fe_model",
]
