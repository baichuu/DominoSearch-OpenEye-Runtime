# Public runtime scope

Work only on this exported OpenEye hybrid network runner. Keep the immutable
MAC20 source, masks, INT8 arithmetic and all raw/logical golden checks. Apply
maintained patches only to temporary copies. No training, new calibration,
Vivado, board programming or new compression method in this repository.

OpenEye commit: 36395ddc530e106f06c50025a4069dd097688d8d

The workflow must remain manual-only, standard Ubuntu, exact public checkout,
same-dispatch gates and read-only repository permissions. Never import old
private-repository checkpoints across execution identities. Keep source tensors,
credentials, learned affine coefficients, private simulator logs and encrypted
checkpoints out of Git and public Actions artifacts. Run the public-evidence
negative controls and validate generated public copies before publication.

Checkpoints use v2, segments0..5 are completed blocks,6..10 are checked layer4
stages with original skip retained,11 is final GAP/FC audit. Failed/skipped or
incomplete work cannot be certified as PASS. Read README.md before changes.
