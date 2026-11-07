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

## Dense baseline extension — 2026-11-07

User explicitly authorizes original dense checkpoint comparison against audited
MAC20 run42238422605/public79544bf. On branch baseline-dense-r3 use
OPENEYE_MODEL_PROFILE=dense-baseline-v1 and verified original dense4:4 weights.
Keep source input sample0 from immutable MAC20 verification vectors; dense has
its own BN/scales/oracle/checkpoint chain. No reconstruction of pruned weights.
Keep exact pin363/r3/72PE/buffers/mapping/RTL/patches and all numeric gates.
This overrides only the earlier MAC20-only source restriction for this explicit
baseline experiment. Never substitute either model or import cross-model state.


## Dense fixture environment fix — 2026-11-07

Run42374896479/ed17bc5 fails checkpoint unit controls before any segment0 RTL:
legacy fabricated MAC20 metadata inherits OPENEYE_MODEL_PROFILE=dense-baseline-v1.
Both own gates independently audited PASS:downsample64RTL+8focused/196checksums,
FC1000RTL+8focused/2086checksums. Preserve gate evidence and failed run; no dense
network PASS or completed checkpoint boundary. Local exact CI-environment
reproducer confirms14fixture failures before DUT. Scope model=mac20 ONLY to
checkpoint fixture-test step, include explicit dense source admission tests;
all12inherited segment steps covered. Actual workflow/DUT remain dense. No
runtime/arithmetic/golden/checkpoint/ACL/grant/source/RTL changes. Workflow
commit changes require a fresh same-commit gated chain, no old gate import.
