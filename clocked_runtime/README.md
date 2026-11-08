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
