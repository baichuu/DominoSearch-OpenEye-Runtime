import itertools
import unittest

from fpga.openeye.nm_offset_adapter import (
    PackedGroup,
    decode_region,
    encode_region,
    encode_sparse_positions,
    encode_sparse_positions_paired,
    retained_positions,
)


class OpenEyeNmOffsetAdapterTest(unittest.TestCase):
    def test_every_legal_first_milestone_mask_round_trips(self):
        for n in (2, 3, 4):
            for positions in itertools.combinations(range(4), n):
                mask = sum(1 << position for position in positions)
                values = tuple(position + 1 for position in positions)
                encoded = encode_region((PackedGroup(values, mask),), n, 4)
                expected = tuple(
                    position + 1 if position in positions else 0 for position in range(4)
                )
                self.assertEqual(decode_region(encoded), expected)

    def test_zero_run_carries_across_group_boundary(self):
        encoded = encode_region(
            (
                PackedGroup((11, 12), 0b0011),
                PackedGroup((21, 22), 0b1100),
            ),
            2,
            4,
        )
        self.assertEqual([pair.zero_run for pair in encoded.pairs], [0, 0, 4, 0])
        self.assertEqual(decode_region(encoded), (11, 12, 0, 0, 0, 0, 21, 22))

    def test_rejects_invalid_mask_and_retained_zero(self):
        with self.assertRaisesRegex(ValueError, "retains 1 positions"):
            retained_positions(0b0001, 2, 4)
        with self.assertRaisesRegex(ValueError, "outside M"):
            retained_positions(0b10011, 2, 4)
        with self.assertRaisesRegex(ValueError, "retained zero"):
            encode_region((PackedGroup((0, 5), 0b0011),), 2, 4)

    def test_rejects_non_milestone_scheme(self):
        with self.assertRaisesRegex(ValueError, "first FPGA milestone"):
            retained_positions(0b0001, 1, 4)

    def test_long_run_bridge_round_trips_without_changing_dense_positions(self):
        encoded = encode_sparse_positions(
            ((0, 11), (33, 22)),
            34,
            offset_bits=4,
            bridge_long_runs=True,
        )
        self.assertEqual(
            [
                (pair.original_position, pair.zero_run, pair.synthetic_bridge)
                for pair in encoded.pairs
            ],
            [(0, 0, False), (15, 14, True), (30, 14, True), (33, 2, False)],
        )
        self.assertEqual(decode_region(encoded), (11,) + (0,) * 32 + (22,))

    def test_long_run_is_rejected_without_explicit_bridge_mode(self):
        with self.assertRaisesRegex(ValueError, "exceeds 4-bit"):
            encode_sparse_positions(((16, 7),), 17)

    def test_pair_aware_bridge_limits_shared_offset_sum(self):
        encoded = encode_sparse_positions_paired(
            ((5, 11), (17, 22)),
            18,
            offset_bits=4,
        )
        self.assertEqual(
            [(pair.original_position, pair.zero_run) for pair in encoded.pairs],
            [(5, 5), (16, 10), (17, 0)],
        )
        self.assertEqual(decode_region(encoded), (0,) * 5 + (11,) + (0,) * 11 + (22,))
        self.assertLessEqual(
            encoded.pairs[0].zero_run + encoded.pairs[1].zero_run,
            15,
        )

    def test_region_end_bridge_keeps_empty_region_visible_to_hardware(self):
        encoded = encode_sparse_positions_paired(
            (),
            4,
            offset_bits=4,
            emit_region_end_bridge=True,
        )
        self.assertEqual(decode_region(encoded), (0, 0, 0, 0))
        self.assertEqual(
            [pair.original_position for pair in encoded.pairs if pair.synthetic_bridge],
            [3],
        )

    def test_reserved_sentinel_is_not_emitted_as_a_logical_long_run_bridge(self):
        encoded = encode_sparse_positions_paired(
            (),
            16,
            offset_bits=4,
            emit_region_end_bridge=True,
            pad_incomplete_word=True,
        )
        self.assertEqual(decode_region(encoded), (0,) * 16)
        self.assertTrue(all(pair.zero_run != 15 for pair in encoded.pairs))

    def test_region_end_bridge_materializes_trailing_zeros(self):
        encoded = encode_sparse_positions_paired(
            ((0, 11), (2, 22)),
            6,
            offset_bits=4,
            emit_region_end_bridge=True,
        )
        self.assertEqual(decode_region(encoded), (11, 0, 22, 0, 0, 0))
        self.assertEqual(
            [
                (pair.original_position, pair.zero_run, pair.synthetic_bridge)
                for pair in encoded.pairs
            ],
            [(0, 0, False), (2, 1, False), (5, 2, True)],
        )
        self.assertEqual(encoded.trailing_zeros, 0)

    def test_incomplete_word_is_filled_at_a_real_zero_position(self):
        encoded = encode_sparse_positions_paired(
            ((0, 11), (2, 22)),
            6,
            offset_bits=4,
            emit_region_end_bridge=True,
            pad_incomplete_word=True,
        )
        self.assertEqual(decode_region(encoded), (11, 0, 22, 0, 0, 0))
        self.assertEqual(len(encoded.pairs), 4)
        self.assertEqual(
            [pair.original_position for pair in encoded.pairs if pair.synthetic_bridge],
            [1, 5],
        )


if __name__ == "__main__":
    unittest.main()
