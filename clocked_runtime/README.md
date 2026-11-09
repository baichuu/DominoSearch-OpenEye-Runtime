# Clocked whole-network comparison

Separate experiment: `whole-network-int8-fixedpoint-v1`, OpenEye pin `9a742fb`.
Manual profiles are dense, MAC20 and MAC20 optimized. Existing hybrid workflows
and their historical evidence retain their original pin and scope.

One real RTL timeline starts at signed image loading and ends at the last Q40
logit. A common scalar stem feeds actual OpenEye convolutions and FC; clocked
scratch, channel accumulation, fixed-point BN/requantization/ReLU, residual,
GAP and FC scale/bias are included. No host inference tensor operation is timed
as FPGA work. Model initialization precedes the observation interval.

The independent audit checks original model weights/masks, every raw DMA slot,
live activation packets, every forwarded intermediate value, all logits and
native controller counters. Fused BN/requantization/ReLU is reported together.
Stem pipeline spans overlap and must not be summed. The10ns clock is simulation
only; this experiment does not establish accuracy, Fmax, resources or board speed.

All profiles share far-padding32 for the original28-square stride-2 stages.
The optimized core retains zero values with sparse address tags on the bounded
K3/stride-2/height32 path after sparse-anchor controls failed128raw outputs.
Padding alone did not fix that failure. Other verified geometry paths retain
the sparse-anchor policy; every raw slot remains checked. Corrected mixed
geometry controls pass24,576raw words and33,712post-op values, not full network.

Private immutable inputs use the existing owner-only Drive grant. Raw logs,
coefficients and tensors stay runner-private. Public artifacts contain audited
cycle tables, aggregate hashes and real audit JUnit only. Each Actions run must
pass its own native-core gate and full graph audit before totals certify.
Local controls and dispatch alone do not establish full-network PASS.

`SOURCE_SHA256.json` records the explicit runtime-source export. No private Git
history, model tensors or credentials are exported here.


## Resumable clocked execution

Manual workflow `openeye-clocked-resumable.yml` provides all three profiles.
It has 21 numeric-checked stages: stem, 19 convolutions with post-processing,
and GAP/FC. Each stage allows three automatic 3-hour inference chunks; each
active job has a 5-hour cap with checkpoint/audit/upload reserve. A stage can
still exceed that budget: Re-run failed jobs continues its latest same-run,
same-commit snapshot. Completed stages do not rerun.

The snapshot contains the entire Verilated model, memories, cache/FSM, observer,
counters and simulation timestamp. The C++ driver pauses on a settled low edge;
restoring does not reset the inference clock or inject reference tensors. Exact
binary/source/trace identity and authenticated chunk ancestry are required.
Independent prefix checks gate progress; only the complete final audit can
certify network cycles and numeric PASS.

Use the existing private Drive grant and `OPENEYE_NETWORK_RESUME_KEY_B64`.
Model snapshots and cumulative traces stay AES-256-GCM encrypted in the approved
owner-only private root. Public artifacts contain only allowlisted progress or
final audited aggregates/JUnit. Numeric failure preserves private state and
blocks the next stage. Code/pin/format changes require a fresh gated chain.
The monolithic workflow remains available; running old processes cannot acquire
snapshots retroactively. No completed network PASS follows from these controls.

Save/restore local controls preserve all output/accepted/raw/forwarded bytes:
50,135 cycles for a small convolution; 119,533,168 for image→stem→8 filters;
and 3,326,501 across seven geometry sites / four processes. An authenticated
whole-model stem snapshot restored into a new directory also passes the full
first convolution (200,704 values). These are bounded controls, not network PASS.


## Corrected zero-rich convolution path

The original optimized run 42515583049 fails the independent numeric audit.
Local connected-prefix diagnosis identifies layer1.1.conv1 after a verified
residual. An isolated original command fails 32 of 32,256 raw words; preserving
zero activation values with sparse position/address metadata passes the same
unchanged input, weights, mask, mapping and golden. Runtime K3 convolutions now
use that correctness fallback; native FC and other optimizations are preserved.
Sparse-anchor skipping is not admitted for this connected-network experiment.

User-requested cancellation of old MAC20 run 42515574607 is preserved as cancelled.
Old dense run 42515554294 retains its independently audited 415,099,249 cycles;
it is not reassigned to new execution commits. New chains require fresh gates.
All profiles use the same signed image, fixed-point arithmetic and effective
numeric architecture parameters. Results cover one verification vector and
modeled ROM/SRAM interfaces; they are not external DDR/AXI or board latency.
Numeric failures expose fixed codes/instruction offsets, never operands.


## Explicit three-channel stem candidate

Manual `openeye-clocked-resumable.yml` input `stem_channel_lanes` defaults to
`1`. User authorizes a full-network test with `profile=optimized` and lanes `3`.
This hardware candidate banks the unchanged signed RGB image and weights by
channel and uses three MAC lanes. Local full stems for dense/MAC20 pass every
802,816 pre-activation and 200,704 pooled value; stem service cycles reduce
118,164,487 to39,488,519. Image→stem→first complete convolution also passes the
independent prefix check. These gates are not a new complete-network result.

The requested lane count is authenticated in private checkpoint context and
checked against the actual binary parameter before execute/final audit. Each
job receives the same option. Native convolution→FC gate stays unchanged;
only the graph frontend selects parallel lanes. New code uses a fresh chain,
never old scalar/local snapshots. Inputs, masks, fixed-point contract and raw
checkers stay unchanged. Additional multipliers/banking have no measured area
or Fmax yet; no board speedup or accuracy claim follows from this dispatch.


## Passive per-function cycle tables

New manual staged runs default to `function_metrics=1` for all three profiles.
The option binds every stage and final audit to the actual binary and AES-GCM
context; counter-free checkpoints cannot be imported. Existing runs keep their
exact execution source and cannot acquire counters retroactively.

`functions.csv` observes actual register handshakes for folded BN affine,
residual scale/add, requantization, ReLU/bypass, INT8 saturation, FIFO, GAP,
FC Q40 scale/bias, stem convolution, maxpool and image ingress. Per-value
latency and whole-tensor span are distinct. Function rows overlap and must
not be added to network cycles. Fused arithmetic has no invented sub-step
latency. Disabled functions are explicitly `NOT_EXECUTED`.

`exclusive-steps.csv` partitions every observed edge among ingress, stem,
DMA, PSUM, SRAM, postops and controller. Each instruction closes against its
actual timeline; the final sum must equal the independent whole-network cycle
count. `summarize_function_metrics.py` makes a per-site CSV and readable
Markdown table from a completed independent audit only. Public egress checks
CSV equality with the certified JSON. No tensors or raw logs are exported.

Actual controls preserve output and total cycles, including a save/restore
inside the pipeline. A connected stem3/first-convolution prefix preserves
45,143,129 cycles and every actual trace. This is prefix validation, not a
completed full-network function result or a new accuracy/Fmax/board claim.


## Opt-in shared-port postops controller

Manual `postops_pipelined=1` overlaps affine/residual inputs with earlier
outputs in the existing pipeline. Default remains0. Arithmetic, model/masks,
FIFO depth8 and shared scratch address are unchanged. Writes occupy non-read
slots; no additional memory port is assumed. Exact in-place and disjoint
buffers are supported; shifted overlap fails before writes.

All stages/final bind the option to authenticated checkpoint identity and
actual ELF parameters. The same-dispatch native convolution/FC gate uses the
same controller policy. Never import a legacy checkpoint under a new policy.
Existing Actions executions keep their original source and measurements.

Local independent native gates check50,176conv outputs,8FC logits,32,264raw
words and37,980accepted words:1,968,407→1,667,356cycles, all five actual traces
identical. Candidate with passive counters and three save/restore chunks has
the same total/output and exact exclusive ledger. Focused200,704-value postops
controls reduce affine66.67% and residual54.55%. These are gate/control results,
not candidate full-network, accuracy, Fmax or board evidence. No candidate
network has been dispatched. The~9% additional network reduction is a projection.

When overlap is enabled, zero exclusive STORE-state cycles do not mean zero
SRAM writes: writes happen during other state slots and all actual outputs
remain checked. Public reports label the controller policy explicitly.
