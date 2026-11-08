# Public runtime scope

## Separate clocked experiment authorized 2026-11-09

The user explicitly authorizes full-network INT8/fixed-point comparisons for
dense, MAC20 and MAC20 optimized, after updating to verified upstream pin
9a742fbd42a55ebfa8fcd39aa41dfa36a367572f. The runtime is isolated under
clocked_runtime/ and openeye-clocked-network.yml; preserve the historical
hybrid runtime and pin below. Do not inherit hybrid accuracy or cycle results.

Measure one uninterrupted real RTL timeline, from the first signed image byte
through actual stem, OpenEye backbone, residual/post-ops/GAP and final Q40
logit. No host tensor arithmetic or summed disconnected services may become
network cycle evidence. Same input and arithmetic contract, unchanged source
weights/masks, same-dispatch native core gate and independent full output audit
are required. No synthesis, training, new accuracy claim or board run here.

Use only the existing owner-only private Drive root/grant for authenticated
immutable model inputs. Keep all model tensors, coefficients, ROMs and raw logs
out of Git and public artifacts. Publish only the explicit aggregate schemas,
cycle CSVs, hashes and real JUnit failure status. Manual-only standard Ubuntu,
exact public commit and read-only repository permissions remain mandatory.

The sections below govern the historical hybrid runner.

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
