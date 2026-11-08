"""Position-only adapter from Domino N:M bitmasks to OpenEye zero-run offsets.

This module deliberately does not quantize FP32 checkpoint values and does not
claim to produce a complete OpenEye DMA stream.  It implements the metadata
conversion needed inside one OpenEye weight-address region.  Tensor reordering
and PE/address-region partitioning must happen before calling ``encode_region``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PackedGroup:
    """One Domino group: retained values in ascending original position."""

    values: tuple[int, ...]
    bitmask: int


@dataclass(frozen=True)
class OpenEyeOffsetPair:
    """One OpenEye sparse item before transport-word packing."""

    value: int
    zero_run: int
    original_position: int
    synthetic_bridge: bool = False


@dataclass(frozen=True)
class EncodedRegion:
    pairs: tuple[OpenEyeOffsetPair, ...]
    total_positions: int
    trailing_zeros: int


def validate_nm(n: int, m: int) -> tuple[int, int]:
    n, m = int(n), int(m)
    if m != 4 or n not in {2, 3, 4}:
        raise ValueError("The first FPGA milestone accepts only 4:4, 3:4, or 2:4.")
    return n, m


def retained_positions(bitmask: int, n: int, m: int) -> tuple[int, ...]:
    """Decode one Domino bitmask and validate that it retains exactly N slots."""

    n, m = validate_nm(n, m)
    bitmask = int(bitmask)
    if bitmask < 0 or bitmask >> m:
        raise ValueError(f"Bitmask 0x{bitmask:x} contains bits outside M={m}.")
    positions = tuple(position for position in range(m) if bitmask & (1 << position))
    if len(positions) != n:
        raise ValueError(
            f"Bitmask 0x{bitmask:x} retains {len(positions)} positions; expected N={n}."
        )
    return positions


def encode_region(
    groups: Sequence[PackedGroup],
    n: int,
    m: int,
    *,
    offset_bits: int = 4,
    bridge_long_runs: bool = False,
) -> EncodedRegion:
    """Convert consecutive N:M groups in one OpenEye address region.

    OpenEye stores the count of zeros immediately preceding each retained
    weight.  The count therefore carries across N:M group boundaries, but is
    reset by the caller at every OpenEye weight-address region boundary.

    Values must already be signed hardware payloads.  In particular, this
    function never converts the FP32 packed artifact to INT8.
    """

    n, m = validate_nm(n, m)
    if offset_bits <= 0:
        raise ValueError("offset_bits must be positive.")
    retained_items: list[tuple[int, int]] = []

    for group_index, group in enumerate(groups):
        positions = retained_positions(group.bitmask, n, m)
        if len(group.values) != n:
            raise ValueError(
                f"Group {group_index} contains {len(group.values)} values; expected {n}."
            )
        retained = dict(zip(positions, group.values, strict=True))
        for local_position in range(m):
            if local_position not in retained:
                continue
            value = int(retained[local_position])
            if value == 0:
                raise ValueError(
                    "OpenEye zero-run encoding cannot represent a retained zero; "
                    "the INT8 conversion policy must resolve it explicitly."
                )
            retained_items.append((group_index * m + local_position, value))

    return encode_sparse_positions(
        retained_items,
        len(groups) * m,
        offset_bits=offset_bits,
        bridge_long_runs=bridge_long_runs,
    )


def encode_sparse_positions(
    items: Sequence[tuple[int, int]],
    total_positions: int,
    *,
    offset_bits: int = 4,
    bridge_long_runs: bool = False,
) -> EncodedRegion:
    """Encode an already-reordered sparse region.

    ``items`` must contain strictly increasing ``(position, value)`` pairs.
    When enabled, long runs are split by synthetic zero-valued bridge tokens.
    A bridge consumes one logical zero position, allowing the following offset
    to restart without changing the decoded dense region. A zero-valued bridge
    never uses the all-ones offset because that transport code is reserved as
    the escape for a logical zero/offset-zero token. Hardware acceptance of
    these tokens still requires a separate RTL correctness test.
    """

    if offset_bits <= 0:
        raise ValueError("offset_bits must be positive.")
    total_positions = int(total_positions)
    if total_positions < 0:
        raise ValueError("total_positions cannot be negative.")
    max_offset = (1 << int(offset_bits)) - 1
    bridge_offset = max_offset - 1
    previous_position = -1
    pairs: list[OpenEyeOffsetPair] = []
    for raw_position, raw_value in items:
        position, value = int(raw_position), int(raw_value)
        if position <= previous_position or position >= total_positions:
            raise ValueError("Sparse positions must be strictly increasing and in range.")
        if value == 0:
            raise ValueError("A retained sparse value cannot be zero.")
        pending_zeros = position - previous_position - 1
        while pending_zeros > max_offset:
            if not bridge_long_runs:
                raise ValueError(
                    f"Zero run {pending_zeros} exceeds {offset_bits}-bit OpenEye metadata."
                )
            if bridge_offset < 0:
                raise ValueError("At least one non-sentinel bridge offset is required.")
            # {value=0, offset=max} is reserved as the transport escape for
            # a logical zero/offset-zero entry. Long-run bridges therefore
            # stop at max-1 and leave the reserved code unambiguous.
            bridge_position = previous_position + bridge_offset + 1
            pairs.append(
                OpenEyeOffsetPair(
                    value=0,
                    zero_run=bridge_offset,
                    original_position=bridge_position,
                    synthetic_bridge=True,
                )
            )
            previous_position = bridge_position
            pending_zeros = position - previous_position - 1
        pairs.append(
            OpenEyeOffsetPair(
                value=value,
                zero_run=pending_zeros,
                original_position=position,
            )
        )
        previous_position = position

    return EncodedRegion(
        pairs=tuple(pairs),
        total_positions=total_positions,
        trailing_zeros=total_positions - previous_position - 1,
    )


def encode_sparse_positions_paired(
    items: Sequence[tuple[int, int]],
    total_positions: int,
    *,
    offset_bits: int = 4,
    emit_region_end_bridge: bool = False,
    pad_incomplete_word: bool = False,
) -> EncodedRegion:
    """Encode a region while respecting OpenEye's two-offset adder width.

    OpenEye transports two sparse entries per word and accumulates their two
    offsets in an ``offset_bits``-wide signal. Extra zero-valued bridge tokens
    are therefore inserted not only for an individual long run, but also when
    two otherwise-valid neighboring offsets would overflow that shared adder.
    The all-ones offset with a zero payload remains reserved for transporting
    a logical zero/offset-zero bridge. ``emit_region_end_bridge`` additionally
    materializes the final logical position when it is zero, so OpenEye can
    observe an otherwise implicit hardware-region boundary.
    ``pad_incomplete_word`` inserts a bridge at an
    unused logical zero when needed, because OpenEye consumes two entries from
    every accepted transport word.
    """

    if offset_bits <= 0:
        raise ValueError("offset_bits must be positive.")
    total_positions = int(total_positions)
    if total_positions < 0:
        raise ValueError("total_positions cannot be negative.")
    maximum = (1 << int(offset_bits)) - 1
    normalized_items = [(int(position), int(value), False) for position, value in items]
    previous_input_position = -1
    for position, value, _ in normalized_items:
        if position <= previous_input_position or position >= total_positions:
            raise ValueError("Sparse positions must be strictly increasing and in range.")
        if value == 0:
            raise ValueError("A retained sparse value cannot be zero.")
        previous_input_position = position
    if (
        emit_region_end_bridge
        and total_positions
        and (not normalized_items or normalized_items[-1][0] != total_positions - 1)
    ):
        normalized_items.append((total_positions - 1, 0, True))

    while True:
        previous_position = -1
        pairs: list[OpenEyeOffsetPair] = []
        for position, value, forced_bridge in normalized_items:
            pending = position - previous_position - 1
            while True:
                first_offset = pairs[-1].zero_run if len(pairs) % 2 else 0
                allowance = maximum - first_offset if len(pairs) % 2 else maximum
                reserved_zero_bridge = (
                    forced_bridge and value == 0 and pending == maximum
                )
                if pending <= allowance and not reserved_zero_bridge:
                    break
                bridge_offset = min(allowance, maximum - 1)
                if bridge_offset < 0:
                    raise ValueError(
                        "At least one non-sentinel bridge offset is required."
                    )
                bridge_position = previous_position + bridge_offset + 1
                if bridge_position >= position:
                    raise AssertionError(
                        "Pair-aware bridge did not advance inside a zero run."
                    )
                pairs.append(
                    OpenEyeOffsetPair(
                        value=0,
                        zero_run=bridge_offset,
                        original_position=bridge_position,
                        synthetic_bridge=True,
                    )
                )
                previous_position = bridge_position
                pending = position - previous_position - 1
            pairs.append(
                OpenEyeOffsetPair(
                    value=value,
                    zero_run=pending,
                    original_position=position,
                    synthetic_bridge=forced_bridge,
                )
            )
            previous_position = position

        if not pad_incomplete_word or len(pairs) % 2 == 0:
            break
        occupied = {pair.original_position for pair in pairs}
        normalized_positions = {position for position, _, _ in normalized_items}
        padding_position = next(
            (
                position
                for position in range(total_positions)
                if position not in occupied and position not in normalized_positions
            ),
            None,
        )
        if padding_position is None:
            raise ValueError(
                "Cannot fill the final two-entry OpenEye word inside this region."
            )
        normalized_items.append((padding_position, 0, True))
        normalized_items.sort(key=lambda item: item[0])

    for index in range(0, len(pairs), 2):
        pair_sum = pairs[index].zero_run
        if index + 1 < len(pairs):
            pair_sum += pairs[index + 1].zero_run
        if pair_sum > maximum:
            raise AssertionError("Pair-aware encoder produced an overflowing offset sum.")
    return EncodedRegion(
        pairs=tuple(pairs),
        total_positions=total_positions,
        trailing_zeros=total_positions - previous_position - 1,
    )


def decode_region(encoded: EncodedRegion) -> tuple[int, ...]:
    """Reconstruct a region, used for exhaustive position round-trip tests."""

    result: list[int] = []
    for pair in encoded.pairs:
        result.extend([0] * pair.zero_run)
        if len(result) != pair.original_position:
            raise ValueError("Offset pairs are inconsistent with their recorded positions.")
        result.append(pair.value)
    result.extend([0] * encoded.trailing_zeros)
    if len(result) != encoded.total_positions:
        raise ValueError(
            f"Decoded {len(result)} positions; expected {encoded.total_positions}."
        )
    return tuple(result)
