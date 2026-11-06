# DominoSearch OpenEye runtime

Standalone public runner for the FPGA correctness evaluation of layer-wise
mixed N:M pruning. The main research repository and its history remain private.
This repository contains selected runtime code, maintained compatibility patches,
execution geometry and synthetic tests. It contains no model weights, datasets,
activation tensors, checkpoints, learned affine coefficients or credentials.

OpenEye is checked out separately from
[Learning-Chips-Lab/OpenEye](https://github.com/Learning-Chips-Lab/OpenEye), pinned
to `36395dd`. Maintained patches apply only to temporary/generated copies.
Original upstream copyright notices are retained. Public visibility alone does
not grant a new license for code without an explicit license.

## Running

The only workflow entry point is manual `hybrid-network-resumable`, r3,
original 56x56 stem geometry. GitHub standard Ubuntu runners are used. Both
same-dispatch, same-commit downsample/FC gates must pass before the network jobs.
There are six block jobs followed by five layer4 stage jobs and final GAP/FC.
This avoids combining multiple long layer4 stages in one six-hour hosted job.
Whole-stage checkpoints retain actual DUT, independent oracle, scales and the
original residual skip. No incomplete tile or reference substitution is reused.

Configure these Actions secrets using existing authorized credentials:

- `DOMINO_RCLONE_CONFIG_B64`: only the existing `domino-drive` remote, with
  unchanged grant/scopes; never committed or printed.
- `OPENEYE_NETWORK_RESUME_KEY_B64`: a dedicated random 32-byte key for this
  public runner, stored only as an Actions secret and private local recovery.

Source bundles stay in private Drive and are checked by manifest/size/SHA-256,
checkpoint and scheme identity. Encrypted checkpoint objects stay in the
approved owner-only private root. Identity binds the public execution repository,
commit, chain ID, OpenEye pin, helper hashes, immutable source and inputs.
The old private repository's checkpoints cannot be imported into a public run.

Start a fresh chain with `resume_run_id` empty. For infrastructure failures
on unchanged code use **Re-run failed jobs** on the same run. Code, pin or
checkpoint-format changes require fresh gates/chain. The six-hour job limit
still applies; splitting stages is not a guaranteed runtime bound.

## Public artifacts and logs

Raw simulator stdout is redirected to runner-private logs. A separate
`publish_public_evidence.py` copy exports only allowlisted metadata, hashes,
numeric handshake counters and redacted real JUnit. It strips simulator payload
lines and JUnit free-text/property paths, preserves failure/error/skip status,
and regenerates checksums after redaction. The unchanged counter and golden
verifiers must still accept the published copy. Private sealing happens before
publication; no activation/checkpoint files or ciphertext are uploaded as public
Actions artifacts. Raw private logs are ephemeral on the runner and are not
available after its disposal.

The workflow is dispatch-only, checks out exact public source, has read-only
repository permissions and does not execute fork/PR code with secrets. Source
provenance is recorded in `PUBLIC_SOURCE_MANIFEST.json`; exported code has a new
Git history rather than publishing the research repository's old commits.

## Evidence limits

This is hybrid INT8 correctness debugging with host stem, BN/requantization,
channel accumulation, residual/ReLU, GAP and FC bias. It is not full-RTL
inference, ImageNet accuracy, a measured board speedup, power/energy result or
post-route evidence. FP32 pruning accuracy remains a separate experiment.
Synthetic unit tests are not network DUT evidence. A network PASS requires
all 20 weighted RTL stages, eight residual blocks, 3,720 partial JUnit and final
logit checks in the maintained final audit.

## Export validation

Local Artix Linux (reference-environment deviation): 308 tests PASS; six
opt-in RTL cases are skipped locally and remain required by the workflow's
focused RTL gates. Actionlint 1.7.12 validates the dispatch-only workflow.
Publication redaction was also checked against preserved historical evidence:
672 prefix real JUnit plus 256/512 layer4 convolution JUnit and all counter
totals remain accepted by the unchanged evidence verifiers. Those historical
checks validate publication behavior, not a new public network execution.

The initial public run `41623326842` failed its focused gates before accepted
hardware-counter records. The export omitted `nm_offset_adapter.py`, imported
from the generated pair-safe patch rather than a normal source AST. The helper
and its exhaustive adapter controls are now included; a new export check scans
generated patch imports. Updated unit suite: **320 PASS, 6 opt-in RTL skipped**.
The failed run remains failed evidence; only a fresh same-commit gated run can
establish new public RTL/network correctness.

## Current baseline (2026-11-06)

The new full-network baseline uses pin36395dd and maintained correctness patches,
including activation packing and the updated 32/24/18/16-bit register layout.
Keep original r3 hardware, 64-channel/8-filter commands, MAC20 source/masks and
INT8 arithmetic fixed. Experimental activation reuse, compact staging, round
overlap, native-stride projection tiling and filter parallelism are not enabled.
This baseline isolates later OpenEye changes; no new DominoSearch run is included.
The historical pin93 run41625678985 remains separate evidence, not this baseline.
Start a fresh chain; never import its checkpoints or assign its counters to pin363.

Validation of the pin363 baseline export on Artix Linux: 331 tests PASS,
six opt-in RTL cases skipped locally and still required in same-dispatch CI
gates; PE smoke 3/3 PASS. Actionlint1.7.12 and shell/Python syntax PASS.
Local AEAD tests use a separate Python3.11 environment with the pinned
cryptography dependency; simulation environment and private credentials unchanged.
These checks do not establish a new full-network PASS.

Host BatchNorm/activation instrumentation records calls, elements and
perf_counter_ns spans in host_operations.csv and JSON. Stem FP64 BN/ReLU,
fixed-point BN affine/requantization, residual alignment/add and fused
ReLU/INT8 clamp/cast are distinct operations. Forward/oracle/coefficient-setup
phases remain separate. These are single-call host timings, not FPGA cycles,
standalone INT8 ReLU latency, accuracy or board measurements. Resumed prefix
timings are counted once; rebuilding the host stem for checkpoint validation
is excluded from model execution totals. No tensor/coefficient values are exported.

## Pin363 projection count fix (2026-11-06)

Baseline run42235216530 failed both focused gates; all network segments were
skipped. The retained-zero kernel1 path inherited an unconditional address-count
clear in the new upstream activation pipeline. Its generated-only patch now
clears metadata only at compute_i, matching the maintained kernel3 policy.
Weight sparsity, model/masks, INT8 arithmetic, pin, r3 geometry and all raw/logical
goldens remain fixed. A module-level Icarus regression fails before the fix and
passes after it; projection8/16 full-wrapper fixtures pass two real DUT JUnit.
Private selected and public redacted copies both pass the unchanged verifier.
Unit/privacy/checkpoint controls:332 PASS, six required opt-in CI cases skipped
locally. Start a fresh same-dispatch gated chain, never reuse old boundaries.
