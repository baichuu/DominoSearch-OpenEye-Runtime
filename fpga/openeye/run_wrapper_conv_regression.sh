#!/usr/bin/env bash
# Run convolution-wrapper correctness checks without modifying pinned OpenEye.

set -Eeuo pipefail

readonly EXPECTED_OPENEYE_COMMIT='93f0e82bd6aa0cddee73c9db115884e8ebdd7329'
readonly FIXED_IACT_RAM_CELLS=8
readonly FIXED_IACT_BUFFER_WIDTH=12
readonly FIXED_WGHT_BUFFER_WIDTH=13
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPOSITORY_ROOT="$(realpath -- "$SCRIPT_DIR/../..")"
readonly REPOSITORY_ROOT
readonly IACT_COUNTER_PATCH="$SCRIPT_DIR/patches/openeye-iact-counter-ownership.patch"
readonly PE_WGHT_COUNT_WIDTH_PATCH="$SCRIPT_DIR/patches/openeye-pe-weight-spad-count-width.patch"
readonly ODD_CHANNEL_PATCH="$SCRIPT_DIR/patches/openeye-serial-odd-channel-padding.patch"
readonly SHARED_STATE_PATCH="$SCRIPT_DIR/patches/openeye-layer-mapper-shared-state.patch"
readonly CONV_BIAS_ORDER_PATCH="$SCRIPT_DIR/patches/openeye-conv-bias-kernel-order.patch"
readonly TALL_CONV_SYNC_PATCH="$SCRIPT_DIR/patches/openeye-tall-conv-sync.patch"
readonly OUTPUT_ORDER_PATCH="$SCRIPT_DIR/patches/openeye-wrapper-conv-output-order.patch"
readonly PAIR_SAFE_WGHT_PATCH="$SCRIPT_DIR/patches/openeye-wrapper-pair-safe-weights.patch"
readonly ZERO_BRIDGE_PATCH="$SCRIPT_DIR/patches/openeye-zero-bridge-sentinel.patch"
readonly HARDWARE_METRICS_PATCH="$SCRIPT_DIR/patches/openeye-wrapper-hardware-metrics.patch"
readonly PROGRESS_WATCHDOG_PATCH="$SCRIPT_DIR/patches/openeye-wrapper-progress-watchdog.patch"
readonly ACTUAL_CANDIDATE_BIAS_PATCH="$SCRIPT_DIR/patches/openeye-wrapper-actual-candidate-bias.patch"
readonly WIDE_PSUM_OUTPUT_PATCH="$SCRIPT_DIR/patches/openeye-wrapper-wide-psum-output.patch"
readonly WIDE_IACT_BUFFER_COUNT_PATCH="$SCRIPT_DIR/patches/openeye-wrapper-wide-iact-buffer-count.patch"
readonly WEIGHT_STAGING_URAM_PATCH="$SCRIPT_DIR/patches/openeye-weight-staging-uram.patch"
readonly SELECT_SIMULATOR_PATCH="$SCRIPT_DIR/patches/openeye-wrapper-select-simulator.patch"
readonly GEMM_FIXTURE_PATCH="$SCRIPT_DIR/patches/openeye-wrapper-gemm-fixture.patch"
readonly STRIDE_OUTPUT_PATCH="$SCRIPT_DIR/patches/openeye-wrapper-stride-output-bounds.patch"
readonly STRIDED_IACT_PATCH="$SCRIPT_DIR/patches/openeye-wrapper-strided-iact-pitch.patch"
readonly SPATIAL_CHECK_PATCH="$SCRIPT_DIR/patches/openeye-wrapper-spatial-check.patch"
readonly SPATIAL_TILE_TOOL="$SCRIPT_DIR/spatial_layer_tiles.py"
readonly RUNTIME_ACTIVATION_TOOL="$SCRIPT_DIR/runtime_activations.py"
readonly RUNTIME_ACTIVATION_PATCH="$SCRIPT_DIR/patches/openeye-wrapper-runtime-activations.patch"
readonly NETWORK_TILE_TOOL="$SCRIPT_DIR/network_layer_tiles.py"
readonly RUNTIME_PROJECTION_IACT_PATCH="$SCRIPT_DIR/patches/openeye-runtime-projection-dense-iact.patch"
readonly RUNTIME_KERNEL3_IACT_PATCH="$SCRIPT_DIR/patches/openeye-runtime-kernel3-retained-iact.patch"
readonly METRICS_SUMMARIZER="$SCRIPT_DIR/summarize_wrapper_metrics.py"
readonly FIXTURE_TOOL="$SCRIPT_DIR/prepare_wrapper_candidate_fixtures.py"
readonly PACKED_LAYER_FIXTURE_TOOL="$SCRIPT_DIR/prepare_packed_layer_fixture.py"

die() {
  printf '[openeye-wrapper-regression] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  fpga/openeye/run_wrapper_conv_regression.sh OPENEYE_ROOT OUTPUT_DIR [MODE] [FIXTURE_ROOT]

MODE is one of:
  focused              Run the upstream NUM_GLB_IACT=1, four-channel tall test.
  representative       Run a 56x56, four-channel geometry probe.
  dense-64-smoke       Run dense constants at 8x4, 64 channels, 64 filters.
  dense-scale          Run dense constants with OPENEYE_DENSE_{CHANNELS,FILTERS}.
  candidates           Run candidate masks at 8x4, 4 channels, 4 filters.
  candidates-scale     Run candidates with OPENEYE_CANDIDATE_{CHANNELS,FILTERS}.
  candidates-64-smoke  Run candidate masks at 8x4, 64 channels, 64 filters.
  candidates-layer     Run candidate masks at 56x56, 64 channels, 64 filters.
  packed-layer         Run one selected packed MAC20 convolution or FC fixture.
  kernel7-smoke        Run a small asymmetric 7x7 stride-2 probe on full-layer-r3.
  full                 Run all upstream test_conv_const.py cases.
  gemm                 Run all upstream test_gemm_layer.py cases.
  attention            Run all upstream test_attention.py cases.

The default is focused. FIXTURE_ROOT is required for every candidates* mode.
FIXTURE_ROOT is also required for packed-layer and must contain one fixture
created by prepare_packed_layer_fixture.py.
OPENEYE_ARCHITECTURE is v6 (default) or full-layer-r3. The latter is allowed
only for packed-layer and kernel7-smoke; it does not alter historical gates.
OPENEYE_CANDIDATE_SCHEMES may select a whitespace-separated subset of 4x4,
3x4, and 2x4. OPENEYE_SIMULATOR may be icarus (default) or verilator.
OPENEYE_WRAPPER_TIMEOUT_SECONDS defaults to 3600 for candidates-layer and 1800
otherwise. The cocotb simulation timeout defaults to 60 ms for
candidates-layer and 15 ms otherwise; override it with OPENEYE_SIM_TIMEOUT_NS.
For targeted diagnosis in full mode, OPENEYE_CONV_TEST_NODEID may select one
test_conv_const.py node ID; the default remains the complete file.
For gemm mode, OPENEYE_GEMM_TEST_NODEID may select one test_gemm_layer.py node
ID; the default remains the complete file. The target is recorded in evidence.
The no-progress watchdog defaults to 250000 cycles for candidates-layer and
100000 otherwise; override it with OPENEYE_FAIL_ON_STALL. OUTPUT_DIR must not
exist. OpenEye is copied to a temporary directory and remains clean.
EOF
}

[[ ${1:-} != -h && ${1:-} != --help ]] || {
  usage
  exit 0
}

(($# >= 2 && $# <= 4)) || {
  usage >&2
  exit 2
}

openeye_root=$(realpath -- "$1")
output_dir=$(realpath -m -- "$2")
mode=${3:-focused}
fixture_root=${4:-}
candidate_mode=0
packed_layer_mode=0
case "$mode" in
  candidates|candidates-scale|candidates-64-smoke|candidates-layer)
    candidate_mode=1
    ;;
  packed-layer)
    packed_layer_mode=1
    ;;
esac
fixture_mode=$((candidate_mode || packed_layer_mode))

[[ $mode == focused || $mode == representative || $mode == dense-64-smoke || \
   $mode == dense-scale || \
   $mode == candidates || $mode == candidates-scale || \
   $mode == candidates-64-smoke || $mode == candidates-layer || \
   $mode == packed-layer || $mode == kernel7-smoke || $mode == full || \
   $mode == gemm || $mode == attention ]] ||
  die "Unknown mode: $mode"
architecture=${OPENEYE_ARCHITECTURE:-v6}
case "$architecture" in
  v6)
    cluster_rows=2
    architecture_id=openeye-fpga-first-layer-v6-staging-depths
    ;;
  full-layer-r3)
    export STRIDE_ENABLED=1
    # Freeze generated RTL depth; upstream simulator otherwise defaults to 14.
    export BUFFER_WIDTH_PSUM=12
    [[ $mode == packed-layer || $mode == kernel7-smoke ]] ||
      die 'full-layer-r3 is only supported by packed-layer and kernel7-smoke.'
    cluster_rows=3
    architecture_id=openeye-fpga-full-layer-r3-staging-depths
    ;;
  *) die "Unknown OPENEYE_ARCHITECTURE: $architecture" ;;
esac
if [[ $mode == kernel7-smoke && $architecture != full-layer-r3 ]]; then
  die 'kernel7-smoke requires OPENEYE_ARCHITECTURE=full-layer-r3.'
fi
kernel7_stride=${OPENEYE_KERNEL7_STRIDE:-2}
runtime_mode=0
runtime_channel_tiles=${OPENEYE_RUNTIME_CHANNEL_TILES:-0}
[[ $runtime_channel_tiles == 0 || $runtime_channel_tiles == 1 ]] || die 'Invalid runtime channel tiling flag.'
if [[ -n ${OPENEYE_RUNTIME_ACTIVATIONS:-} ]]; then
  [[ $architecture == full-layer-r3 && $mode == packed-layer ]] ||
    die 'Runtime activations require an explicitly selected r3 packed-layer pilot.'
  [[ -s $OPENEYE_RUNTIME_ACTIVATIONS && -n ${OPENEYE_RUNTIME_OUTPUT:-} ]] ||
    die 'Runtime activation input and private output path are required.'
  runtime_output=$(realpath -m -- "$OPENEYE_RUNTIME_OUTPUT")
  [[ ! -e $runtime_output && $runtime_output != "$output_dir" && $runtime_output != "$output_dir"/* ]] ||
    die 'Runtime tensor output must be new and outside uploaded evidence.'
  runtime_mode=1
fi
if [[ $runtime_channel_tiles == 1 && $runtime_mode == 0 ]]; then die 'Network tiles require runtime activations.'; fi
[[ $kernel7_stride == 1 || $kernel7_stride == 2 ]] || die 'OPENEYE_KERNEL7_STRIDE must be 1 or 2.'
if ((fixture_mode)); then
  [[ -n $fixture_root ]] || die 'FIXTURE_ROOT is required for fixture modes.'
  fixture_root=$(realpath -- "$fixture_root")
  [[ -s $fixture_root/manifest.json ]] || die "Fixture manifest not found: $fixture_root"
elif [[ -n $fixture_root ]]; then
  die 'FIXTURE_ROOT is valid only for candidates* or packed-layer modes.'
fi
default_timeout_seconds=1800
if [[ $mode == packed-layer ]]; then
  default_timeout_seconds=19800
elif [[ $mode == candidates-layer ]]; then
  # Verilator needs about 30 minutes including reference generation for the
  # full 4:4 layer on the development host. Keep the process guard above that
  # measured runtime; this is independent of the simulated-cycle timeout.
  default_timeout_seconds=3600
elif [[ $mode == gemm ]]; then
  default_timeout_seconds=7200
fi
timeout_seconds=${OPENEYE_WRAPPER_TIMEOUT_SECONDS:-$default_timeout_seconds}
[[ $timeout_seconds =~ ^[1-9][0-9]*$ ]] ||
  die "OPENEYE_WRAPPER_TIMEOUT_SECONDS must be a positive integer: $timeout_seconds"
default_sim_timeout_ns=15000000
if [[ $mode == packed-layer ]]; then
  default_sim_timeout_ns=180000000
elif [[ $mode == candidates-layer ]]; then
  # The exact 4:4 layer completes at 33.453060 ms. A 30 ms guard therefore
  # rejects a progressing, numerically correct execution before output starts.
  default_sim_timeout_ns=60000000
fi
sim_timeout_ns=${OPENEYE_SIM_TIMEOUT_NS:-$default_sim_timeout_ns}
[[ $sim_timeout_ns =~ ^[1-9][0-9]*$ ]] ||
  die "OPENEYE_SIM_TIMEOUT_NS must be a positive integer: $sim_timeout_ns"
simulator=${OPENEYE_SIMULATOR:-icarus}
[[ $simulator == icarus || $simulator == verilator ]] ||
  die "OPENEYE_SIMULATOR must be icarus or verilator: $simulator"
default_fail_on_stall=100000
if [[ $mode == packed-layer ]]; then
  default_fail_on_stall=500000
elif [[ $mode == candidates-layer ]]; then
  # A correct 56x56x64 conversion holds the main/PSUM/compute progress tuple
  # unchanged for 122497 cycles while fsm_cycle advances in CONVERT_IACT.
  default_fail_on_stall=250000
fi
fail_on_stall=${OPENEYE_FAIL_ON_STALL:-$default_fail_on_stall}
[[ $fail_on_stall =~ ^[1-9][0-9]*$ ]] ||
  die "OPENEYE_FAIL_ON_STALL must be a positive integer: $fail_on_stall"
if [[ $mode == dense-64-smoke || $mode == dense-scale ]]; then
  if [[ $mode == dense-64-smoke ]]; then
    dense_channels=64
    dense_filters=64
  else
    dense_channels=${OPENEYE_DENSE_CHANNELS:-64}
    dense_filters=${OPENEYE_DENSE_FILTERS:-4}
  fi
  dense_width=${OPENEYE_DENSE_WIDTH:-8}
  dense_height=${OPENEYE_DENSE_HEIGHT:-4}
  for dimension in "$dense_width" "$dense_height" "$dense_channels" "$dense_filters"; do
    [[ $dimension =~ ^[1-9][0-9]*$ ]] || die "Dense geometry must contain positive integers."
  done
fi
[[ -d $openeye_root/hdl ]] || die "OpenEye checkout not found: $openeye_root"
[[ -x $openeye_root/.venv/bin/python ]] ||
  die "OpenEye virtual environment not found: $openeye_root/.venv"
[[ ! -e $output_dir ]] || die "OUTPUT_DIR already exists: $output_dir"
for patch_file in "$ODD_CHANNEL_PATCH" "$SHARED_STATE_PATCH" "$IACT_COUNTER_PATCH" "$CONV_BIAS_ORDER_PATCH" "$TALL_CONV_SYNC_PATCH" "$OUTPUT_ORDER_PATCH" "$PAIR_SAFE_WGHT_PATCH" "$ZERO_BRIDGE_PATCH" "$HARDWARE_METRICS_PATCH" "$PROGRESS_WATCHDOG_PATCH" "$ACTUAL_CANDIDATE_BIAS_PATCH" "$WIDE_PSUM_OUTPUT_PATCH" "$WIDE_IACT_BUFFER_COUNT_PATCH" "$WEIGHT_STAGING_URAM_PATCH" "$SELECT_SIMULATOR_PATCH" "$GEMM_FIXTURE_PATCH"; do
  [[ -s $patch_file ]] || die "Patch not found: $patch_file"
done
[[ -s $METRICS_SUMMARIZER ]] || die "Metrics summarizer not found: $METRICS_SUMMARIZER"
[[ -s $PE_WGHT_COUNT_WIDTH_PATCH ]] || die "Patch not found: $PE_WGHT_COUNT_WIDTH_PATCH"
[[ -s $FIXTURE_TOOL ]] || die "Fixture tool not found: $FIXTURE_TOOL"
[[ -s $PACKED_LAYER_FIXTURE_TOOL ]] ||
  die "Packed-layer fixture tool not found: $PACKED_LAYER_FIXTURE_TOOL"

actual_commit=$(git -C "$openeye_root" rev-parse HEAD)
[[ $actual_commit == "$EXPECTED_OPENEYE_COMMIT" ]] ||
  die "OpenEye commit mismatch: expected $EXPECTED_OPENEYE_COMMIT, got $actual_commit"
[[ -z $(git -C "$openeye_root" status --short) ]] ||
  die 'OpenEye has tracked/unignored changes; preserve a clean pinned checkout.'

if ((packed_layer_mode)); then
  "$openeye_root/.venv/bin/python" "$PACKED_LAYER_FIXTURE_TOOL" \
    --verify-root "$fixture_root" --check-wrapper-geometry --architecture "$architecture"
fi

mkdir -p -- "$output_dir"
work_dir=$(mktemp -d -t openeye-wrapper-regression-XXXXXX)
if [[ ${OPENEYE_KEEP_WORK_DIR:-0} == 1 ]]; then
  printf '[openeye-wrapper-regression] Preserving diagnostic work copy: %s\n' "$work_dir"
else
  trap 'rm -rf -- "$work_dir"' EXIT
fi

# Run only from immutable-per-run copies.  Hashing the live source files at the
# beginning is insufficient: a concurrent edit between hashing and applying
# could otherwise make the manifest describe different inputs.
input_snapshot_dir="$work_dir/.domino-inputs"
mkdir -p -- "$input_snapshot_dir/patches"
cp -- "$IACT_COUNTER_PATCH" "$input_snapshot_dir/patches/iact-counter.patch"
cp -- "$PE_WGHT_COUNT_WIDTH_PATCH" "$input_snapshot_dir/patches/pe-weight-count-width.patch"
cp -- "$ODD_CHANNEL_PATCH" "$input_snapshot_dir/patches/serial-odd-channel-padding.patch"
cp -- "$SHARED_STATE_PATCH" "$input_snapshot_dir/patches/layer-mapper-shared-state.patch"
cp -- "$CONV_BIAS_ORDER_PATCH" "$input_snapshot_dir/patches/conv-bias-order.patch"
cp -- "$TALL_CONV_SYNC_PATCH" "$input_snapshot_dir/patches/tall-conv-sync.patch"
cp -- "$OUTPUT_ORDER_PATCH" "$input_snapshot_dir/patches/output-order.patch"
cp -- "$PAIR_SAFE_WGHT_PATCH" "$input_snapshot_dir/patches/pair-safe-weight.patch"
cp -- "$ZERO_BRIDGE_PATCH" "$input_snapshot_dir/patches/zero-bridge.patch"
cp -- "$HARDWARE_METRICS_PATCH" "$input_snapshot_dir/patches/hardware-metrics.patch"
cp -- "$PROGRESS_WATCHDOG_PATCH" "$input_snapshot_dir/patches/progress-watchdog.patch"
cp -- "$ACTUAL_CANDIDATE_BIAS_PATCH" "$input_snapshot_dir/patches/actual-candidate-bias.patch"
cp -- "$WIDE_PSUM_OUTPUT_PATCH" "$input_snapshot_dir/patches/wide-psum-output.patch"
cp -- "$WIDE_IACT_BUFFER_COUNT_PATCH" "$input_snapshot_dir/patches/wide-iact-buffer-count.patch"
cp -- "$WEIGHT_STAGING_URAM_PATCH" "$input_snapshot_dir/patches/weight-staging-uram.patch"
cp -- "$SELECT_SIMULATOR_PATCH" "$input_snapshot_dir/patches/select-simulator.patch"
cp -- "$GEMM_FIXTURE_PATCH" "$input_snapshot_dir/patches/gemm-fixture.patch"
if [[ $architecture == full-layer-r3 ]]; then
  cp -- "$STRIDE_OUTPUT_PATCH" "$input_snapshot_dir/patches/stride-output.patch"
  cp -- "$STRIDED_IACT_PATCH" "$input_snapshot_dir/patches/strided-iact-pitch.patch"
  cp -- "$SPATIAL_CHECK_PATCH" "$input_snapshot_dir/patches/spatial-check.patch"
  cp -- "$SPATIAL_TILE_TOOL" "$input_snapshot_dir/domino_spatial_tiles.py"
fi
if ((runtime_mode)); then
  cp -- "$RUNTIME_ACTIVATION_TOOL" "$input_snapshot_dir/domino_runtime_activations.py"
  cp -- "$NETWORK_TILE_TOOL" "$input_snapshot_dir/domino_network_tiles.py"
  cp -- "$RUNTIME_ACTIVATION_PATCH" "$input_snapshot_dir/patches/runtime-activations.patch"
  cp -- "$RUNTIME_PROJECTION_IACT_PATCH" "$input_snapshot_dir/patches/runtime-projection-dense-iact.patch"
  cp -- "$RUNTIME_KERNEL3_IACT_PATCH" "$input_snapshot_dir/patches/runtime-kernel3-retained-iact.patch"
  cp -- "$OPENEYE_RUNTIME_ACTIVATIONS" "$input_snapshot_dir/runtime-activations.json"
  export OPENEYE_RUNTIME_ACTIVATIONS="$input_snapshot_dir/runtime-activations.json"
  export OPENEYE_RUNTIME_OUTPUT="$runtime_output"
fi
cp -- "$METRICS_SUMMARIZER" "$input_snapshot_dir/summarize-wrapper-metrics.py"
cp -- "$FIXTURE_TOOL" "$input_snapshot_dir/prepare-wrapper-fixtures.py"
cp -- "$PACKED_LAYER_FIXTURE_TOOL" "$input_snapshot_dir/prepare-packed-layer-fixture.py"

iact_counter_patch_snapshot="$input_snapshot_dir/patches/iact-counter.patch"
pe_wght_count_width_patch_snapshot="$input_snapshot_dir/patches/pe-weight-count-width.patch"
conv_bias_order_patch_snapshot="$input_snapshot_dir/patches/conv-bias-order.patch"
tall_conv_sync_patch_snapshot="$input_snapshot_dir/patches/tall-conv-sync.patch"
output_order_patch_snapshot="$input_snapshot_dir/patches/output-order.patch"
pair_safe_weight_patch_snapshot="$input_snapshot_dir/patches/pair-safe-weight.patch"
zero_bridge_patch_snapshot="$input_snapshot_dir/patches/zero-bridge.patch"
hardware_metrics_patch_snapshot="$input_snapshot_dir/patches/hardware-metrics.patch"
progress_watchdog_patch_snapshot="$input_snapshot_dir/patches/progress-watchdog.patch"
actual_candidate_bias_patch_snapshot="$input_snapshot_dir/patches/actual-candidate-bias.patch"
wide_psum_output_patch_snapshot="$input_snapshot_dir/patches/wide-psum-output.patch"
wide_iact_buffer_count_patch_snapshot="$input_snapshot_dir/patches/wide-iact-buffer-count.patch"
weight_staging_uram_patch_snapshot="$input_snapshot_dir/patches/weight-staging-uram.patch"
select_simulator_patch_snapshot="$input_snapshot_dir/patches/select-simulator.patch"
gemm_fixture_patch_snapshot="$input_snapshot_dir/patches/gemm-fixture.patch"
metrics_summarizer_snapshot="$input_snapshot_dir/summarize-wrapper-metrics.py"
fixture_tool_snapshot="$input_snapshot_dir/prepare-wrapper-fixtures.py"
packed_layer_fixture_tool_snapshot="$input_snapshot_dir/prepare-packed-layer-fixture.py"

iact_counter_patch_sha256=$(sha256sum "$iact_counter_patch_snapshot" | awk '{print $1}')
pe_wght_count_width_patch_sha256=$(sha256sum "$pe_wght_count_width_patch_snapshot" | awk '{print $1}')
conv_bias_order_patch_sha256=$(sha256sum "$conv_bias_order_patch_snapshot" | awk '{print $1}')
tall_conv_sync_patch_sha256=$(sha256sum "$tall_conv_sync_patch_snapshot" | awk '{print $1}')
output_order_patch_sha256=$(sha256sum "$output_order_patch_snapshot" | awk '{print $1}')
pair_safe_weight_patch_sha256=$(sha256sum "$pair_safe_weight_patch_snapshot" | awk '{print $1}')
zero_bridge_patch_sha256=$(sha256sum "$zero_bridge_patch_snapshot" | awk '{print $1}')
hardware_metrics_patch_sha256=$(sha256sum "$hardware_metrics_patch_snapshot" | awk '{print $1}')
progress_watchdog_patch_sha256=$(sha256sum "$progress_watchdog_patch_snapshot" | awk '{print $1}')
actual_candidate_bias_patch_sha256=$(sha256sum "$actual_candidate_bias_patch_snapshot" | awk '{print $1}')
wide_psum_output_patch_sha256=$(sha256sum "$wide_psum_output_patch_snapshot" | awk '{print $1}')
wide_iact_buffer_count_patch_sha256=$(sha256sum "$wide_iact_buffer_count_patch_snapshot" | awk '{print $1}')
weight_staging_uram_patch_sha256=$(sha256sum "$weight_staging_uram_patch_snapshot" | awk '{print $1}')
select_simulator_patch_sha256=$(sha256sum "$select_simulator_patch_snapshot" | awk '{print $1}')
gemm_fixture_patch_sha256=$(sha256sum "$gemm_fixture_patch_snapshot" | awk '{print $1}')
metrics_summarizer_sha256=$(sha256sum "$metrics_summarizer_snapshot" | awk '{print $1}')
fixture_tool_sha256=$(sha256sum "$fixture_tool_snapshot" | awk '{print $1}')
packed_layer_fixture_tool_sha256=$(sha256sum "$packed_layer_fixture_tool_snapshot" | awk '{print $1}')

fixture_run_root=$fixture_root
fixture_payload_mode=none
candidate_zero_bias=
zero_bias_label=false
packed_layer_name=
packed_layer_type=
packed_layer_scheme=
packed_layer_geometry=
if ((candidate_mode)); then
  "$openeye_root/.venv/bin/python" "$fixture_tool_snapshot" \
    --verify-root "$fixture_root"
  fixture_run_root="$input_snapshot_dir/fixtures"
  mkdir -p -- "$fixture_run_root"
  cp -- "$fixture_root/manifest.json" \
    "$fixture_root/candidate-4x4-wrapper.json" \
    "$fixture_root/candidate-3x4-wrapper.json" \
    "$fixture_root/candidate-2x4-wrapper.json" \
    "$fixture_run_root/"
  "$openeye_root/.venv/bin/python" "$fixture_tool_snapshot" \
    --verify-root "$fixture_run_root"
  fixture_manifest_sha256=$(sha256sum "$fixture_run_root/manifest.json" | awk '{print $1}')
  fixture_payload_mode=$("$openeye_root/.venv/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("payload_mode", "legacy-token"))' \
    "$fixture_run_root/manifest.json")
  if [[ $fixture_payload_mode == int8-symmetric ]]; then
    candidate_zero_bias=1
    zero_bias_label=true
  fi
elif ((packed_layer_mode)); then
  "$openeye_root/.venv/bin/python" "$packed_layer_fixture_tool_snapshot" \
    --verify-root "$fixture_root"
  fixture_run_root="$input_snapshot_dir/fixtures"
  mkdir -p -- "$fixture_run_root"
  cp -- "$fixture_root/manifest.json" \
    "$fixture_root/packed-layer-wrapper.json" \
    "$fixture_run_root/"
  "$openeye_root/.venv/bin/python" "$packed_layer_fixture_tool_snapshot" \
    --verify-root "$fixture_run_root" --check-wrapper-geometry --architecture "$architecture"
  fixture_manifest_sha256=$(sha256sum "$fixture_run_root/manifest.json" | awk '{print $1}')
  fixture_payload_mode=selected-packed-layer-int8
  candidate_zero_bias=1
  zero_bias_label=true
  IFS=$'\t' read -r packed_layer_name packed_layer_type \
    packed_layer_scheme packed_layer_geometry < <(
      "$openeye_root/.venv/bin/python" -c \
        'import json,sys
m=json.load(open(sys.argv[1], encoding="utf-8"))
e=m["execution"]
if m["layer_type"] == "conv2d":
    geometry="{}x{}-ch{}-f{}-k{}x{}-s{}".format(
        e["input_width"], e["input_height"], e["input_channels"],
        e["output_channels"], e["kernel_width"], e["kernel_height"], e["stride"])
else:
    geometry="k{}-n{}".format(e["input_features"], e["output_features"])
print("\t".join((m["source_layer"], m["layer_type"], m["scheme"], geometry)))' \
        "$fixture_run_root/manifest.json"
    )
fi

tar --exclude=.venv --exclude=test/.temp -cf - \
  -C "$openeye_root" . | tar -xf - -C "$work_dir"
ln -sfn "$openeye_root/.venv" "$work_dir/.venv"

patch --batch --forward --fuzz=0 -p1 -d "$work_dir" <"$tall_conv_sync_patch_snapshot"
git -C "$work_dir" apply "$zero_bridge_patch_snapshot"
git -C "$work_dir" apply "$iact_counter_patch_snapshot"
git -C "$work_dir" apply "$conv_bias_order_patch_snapshot"
git -C "$work_dir" apply "$output_order_patch_snapshot"
git -C "$work_dir" apply "$pair_safe_weight_patch_snapshot"
git -C "$work_dir" apply "$input_snapshot_dir/patches/layer-mapper-shared-state.patch"
git -C "$work_dir" apply "$input_snapshot_dir/patches/serial-odd-channel-padding.patch"
git -C "$work_dir" apply "$hardware_metrics_patch_snapshot"
git -C "$work_dir" apply "$progress_watchdog_patch_snapshot"
git -C "$work_dir" apply "$actual_candidate_bias_patch_snapshot"
git -C "$work_dir" apply "$wide_psum_output_patch_snapshot"
git -C "$work_dir" apply "$wide_iact_buffer_count_patch_snapshot"
iact_write_alignment_patch_sha256=not-applied-upstream
git -C "$work_dir" apply "$pe_wght_count_width_patch_snapshot"
git -C "$work_dir" apply "$weight_staging_uram_patch_snapshot"
git -C "$work_dir" apply "$select_simulator_patch_snapshot"
git -C "$work_dir" apply "$gemm_fixture_patch_snapshot"
stride_output_patch_sha256=not-applied
strided_iact_patch_sha256=not-applied
r3_weight_send_tail_patch_sha256=not-applied
spatial_check_patch_sha256=not-applied
spatial_tile_tool_sha256=not-applied
runtime_activation_patch_sha256=not-applied
runtime_activation_tool_sha256=not-applied
runtime_activation_input_sha256=not-applied
network_tile_tool_sha256=not-applied
runtime_projection_iact_patch_sha256=not-applied
runtime_kernel3_iact_patch_sha256=not-applied
if [[ $architecture == full-layer-r3 ]]; then
  git -C "$work_dir" apply "$input_snapshot_dir/patches/stride-output.patch"
  git -C "$work_dir" apply "$input_snapshot_dir/patches/strided-iact-pitch.patch"
  git -C "$work_dir" apply "$input_snapshot_dir/patches/spatial-check.patch"
  cp -- "$input_snapshot_dir/domino_spatial_tiles.py" "$work_dir/src/domino_spatial_tiles.py"
  stride_output_patch_sha256=$(sha256sum "$input_snapshot_dir/patches/stride-output.patch" | awk '{print $1}')
  strided_iact_patch_sha256=$(sha256sum "$input_snapshot_dir/patches/strided-iact-pitch.patch" | awk '{print $1}')
  r3_weight_send_tail_patch_sha256=not-applied-upstream
  spatial_check_patch_sha256=$(sha256sum "$input_snapshot_dir/patches/spatial-check.patch" | awk '{print $1}')
  spatial_tile_tool_sha256=$(sha256sum "$input_snapshot_dir/domino_spatial_tiles.py" | awk '{print $1}')
fi
if ((runtime_mode)); then
  git -C "$work_dir" apply "$input_snapshot_dir/patches/runtime-activations.patch"
  cp -- "$input_snapshot_dir/domino_runtime_activations.py" "$work_dir/src/domino_runtime_activations.py"
  cp -- "$input_snapshot_dir/domino_network_tiles.py" "$work_dir/src/domino_network_tiles.py"
  network_tile_tool_sha256=$(sha256sum "$input_snapshot_dir/domino_network_tiles.py" | awk '{print $1}')
  runtime_activation_patch_sha256=$(sha256sum "$input_snapshot_dir/patches/runtime-activations.patch" | awk '{print $1}')
  runtime_activation_tool_sha256=$(sha256sum "$input_snapshot_dir/domino_runtime_activations.py" | awk '{print $1}')
  runtime_activation_input_sha256=$(sha256sum "$OPENEYE_RUNTIME_ACTIVATIONS" | awk '{print $1}')
  if [[ $runtime_channel_tiles == 1 && $packed_layer_type == conv2d && $packed_layer_geometry == *-k1x1-s* ]]; then
    # Native bounded projection only: never change v6, FC, kernel3 or the
    # independent-layer/pilot paths. This generated variant is not cost evidence.
    git -C "$work_dir" apply "$input_snapshot_dir/patches/runtime-projection-dense-iact.patch"
    runtime_projection_iact_patch_sha256=$(sha256sum "$input_snapshot_dir/patches/runtime-projection-dense-iact.patch" | awk '{print $1}')
  fi
  if [[ $runtime_channel_tiles == 1 && $packed_layer_type == conv2d && $packed_layer_geometry == *-k3x3-s* ]]; then
    # Retain zeros only in the opt-in runtime kernel3 generated variant.
    # Weight decoder/MAC sparsity and all non-runtime paths remain unchanged.
    git -C "$work_dir" apply "$input_snapshot_dir/patches/runtime-kernel3-retained-iact.patch"
    runtime_kernel3_iact_patch_sha256=$(sha256sum "$input_snapshot_dir/patches/runtime-kernel3-retained-iact.patch" | awk '{print $1}')
  fi
fi

log_path="$output_dir/wrapper-conv-$mode.log"
python_path="$REPOSITORY_ROOT:$SCRIPT_DIR/compat:$work_dir/src:$work_dir/test/cocotb_fpga"
pytest_shard_count=${OPENEYE_PYTEST_SHARD_COUNT:-1}
pytest_shard_index=${OPENEYE_PYTEST_SHARD_INDEX:-0}
[[ $pytest_shard_count =~ ^[1-9][0-9]*$ ]] ||
  die "OPENEYE_PYTEST_SHARD_COUNT must be a positive integer: $pytest_shard_count"
[[ $pytest_shard_index =~ ^[0-9]+$ ]] ||
  die "OPENEYE_PYTEST_SHARD_INDEX must be a non-negative integer: $pytest_shard_index"
((pytest_shard_index < pytest_shard_count)) ||
  die "OPENEYE_PYTEST_SHARD_INDEX must be smaller than OPENEYE_PYTEST_SHARD_COUNT"

build_pytest_command() {
  local target=$1
  local line normalized
  local -a collected=() selected=()

  if ((pytest_shard_count == 1)); then
    selected=("$target")
  else
    while IFS= read -r line; do
      [[ $line == *::* ]] || continue
      normalized=${line#test/cocotb_fpga/}
      collected+=("$normalized")
    done < <(
      cd -- "$work_dir/test/cocotb_fpga"
      PYTHONPATH="$python_path" "$work_dir/.venv/bin/python" -m pytest \
        --collect-only -q "$target"
    )
    ((${#collected[@]} > 0)) || die "No pytest cases collected for sharding: $target"
    for ((index = pytest_shard_index; index < ${#collected[@]}; index += pytest_shard_count)); do
      selected+=("${collected[index]}")
    done
    ((${#selected[@]} > 0)) ||
      die "Pytest shard $pytest_shard_index/$pytest_shard_count is empty for $target"
  fi

  test_command=(
    "$work_dir/.venv/bin/python" -m pytest -q
    --junitxml="$output_dir/pytest-results.xml" "${selected[@]}"
  )
}

case "$mode" in
  kernel7-smoke)
    test_command=(
      "$work_dir/.venv/bin/python" -c
      'import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import test_conv_const as t
t.INPUT_SIZE_X = 16
t.NUM_FILTERS = 4
t.KERNEL_SIZE_X = t.KERNEL_SIZE_Y = 7
t.STRIDE = int(sys.argv[1])
constant_probe = os.environ.get("OPENEYE_KERNEL7_CONSTANTS") == "1"
pattern_path = Path.cwd() / "kernel7-smoke-weights.json"
weights = [[[[((1 if constant_probe else 1 + x + 7*y + c + 3*f) if c < 3 else 0)
              for x in range(7)] for y in range(7)]
            for f in range(4)] for c in range(4)]
probe = os.environ.get("OPENEYE_KERNEL7_PROBE", "none")
if probe not in ("none", "rows", "columns", "channels"):
    raise SystemExit("Invalid OPENEYE_KERNEL7_PROBE")
if probe != "none":
    # Diagnostic operands only; the reference still checks every output.
    weights = [[[[int(c < 3 and (
        (x == (0, 3, 6, 3)[f] if probe == "rows" else
         y == (0, 3, 6, 3)[f] if probe == "columns" else c == f)))
        for x in range(7)] for y in range(7)]
        for f in range(4)] for c in range(4)]
pattern_path.write_text(json.dumps({"scheme": "synthetic-dense", "weights": weights}))
os.environ["OPENEYE_WGHT_PATTERN_JSON"] = str(pattern_path)
request = SimpleNamespace(node=SimpleNamespace(nodeid="domino_kernel7_r3_stride2_smoke"))
t._run_conv_const("Convolution_Single", 1, 3, 1, 4, request,
                  ramp_iacts=not constant_probe, psum_width=32, trans_words=4,
                  bias_step=7, input_height=16, asymmetric_kernel=True)'
      "$kernel7_stride"
    )
    ;;
  focused)
    test_command=(
      "$work_dir/.venv/bin/python" -m pytest -q
      --junitxml="$output_dir/pytest-results.xml"
      'test_conv_const.py::test_conv_tall_input[1-4]'
    )
    ;;
  full)
    conv_test_nodeid=${OPENEYE_CONV_TEST_NODEID:-test_conv_const.py}
    [[ $conv_test_nodeid == test_conv_const.py || \
       $conv_test_nodeid == test_conv_const.py::* ]] ||
      die "OPENEYE_CONV_TEST_NODEID must select test_conv_const.py: $conv_test_nodeid"
    build_pytest_command "$conv_test_nodeid"
    ;;
  gemm)
    gemm_test_nodeid=${OPENEYE_GEMM_TEST_NODEID:-test_gemm_layer.py}
    [[ $gemm_test_nodeid == test_gemm_layer.py || \
       $gemm_test_nodeid == test_gemm_layer.py::* ]] ||
      die "OPENEYE_GEMM_TEST_NODEID must select test_gemm_layer.py: $gemm_test_nodeid"
    build_pytest_command "$gemm_test_nodeid"
    ;;
  attention)
    build_pytest_command test_attention.py
    ;;
  representative)
    test_command=(
      "$work_dir/.venv/bin/python" -c
      'from types import SimpleNamespace; import test_conv_const as t; t.INPUT_SIZE_X=56; request=SimpleNamespace(node=SimpleNamespace(nodeid="domino_layer1_0_conv1_geometry")); t._run_conv_const("Convolution_Single", 8, 2, 1, 4, request, ramp_iacts=True, psum_width=32, trans_words=4, bias_step=7, input_height=56, asymmetric_kernel=True)'
    )
    ;;
  dense-64-smoke|dense-scale)
    test_command=(
      "$work_dir/.venv/bin/python" -c
      'import sys
from types import SimpleNamespace
import test_conv_const as t
width, height, channels, filters = map(int, sys.argv[1:])
t.INPUT_SIZE_X = width
t.NUM_FILTERS = filters
request = SimpleNamespace(
    node=SimpleNamespace(nodeid=f"domino_dense_{width}x{height}_ch{channels}_f{filters}")
)
t._run_conv_const(
    "Convolution_Single", 1, 2, 1, channels, request,
    ramp_iacts=False, psum_width=32, trans_words=4,
    bias_step=0, input_height=height, asymmetric_kernel=False,
)'
      "$dense_width" "$dense_height" "$dense_channels" "$dense_filters"
    )
    ;;
  candidates|candidates-scale|candidates-64-smoke|candidates-layer)
    case "$mode" in
      candidates)
        candidate_width=8
        candidate_height=4
        candidate_channels=4
        candidate_filters=4
        ;;
      candidates-scale)
        candidate_width=${OPENEYE_CANDIDATE_WIDTH:-8}
        candidate_height=${OPENEYE_CANDIDATE_HEIGHT:-4}
        candidate_channels=${OPENEYE_CANDIDATE_CHANNELS:-4}
        candidate_filters=${OPENEYE_CANDIDATE_FILTERS:-4}
        for dimension in "$candidate_width" "$candidate_height" \
                         "$candidate_channels" "$candidate_filters"; do
          [[ $dimension =~ ^[1-9][0-9]*$ ]] ||
            die "Candidate geometry must contain positive integers."
        done
        ;;
      candidates-64-smoke)
        candidate_width=8
        candidate_height=4
        candidate_channels=64
        candidate_filters=64
        ;;
      candidates-layer)
        candidate_width=56
        candidate_height=56
        candidate_channels=64
        candidate_filters=64
        ;;
    esac
    test_command=(
      "$work_dir/.venv/bin/python" -c
      'import os
import sys
from pathlib import Path
from types import SimpleNamespace
import test_conv_const as t

root = Path(sys.argv[1])
width, height, channels, filters = map(int, sys.argv[2:])
allowed = ("4x4", "3x4", "2x4")
schemes = tuple(os.environ.get("OPENEYE_CANDIDATE_SCHEMES", " ".join(allowed)).split())
if not schemes or any(scheme not in allowed for scheme in schemes):
    raise SystemExit(f"Invalid OPENEYE_CANDIDATE_SCHEMES: {schemes}")
t.INPUT_SIZE_X = width
t.NUM_FILTERS = filters
for scheme in schemes:
    os.environ["OPENEYE_WGHT_PATTERN_JSON"] = str(root / f"candidate-{scheme}-wrapper.json")
    request = SimpleNamespace(
        node=SimpleNamespace(
            nodeid=f"domino_candidate_{scheme}_{width}x{height}_ch{channels}_f{filters}"
        )
    )
    t._run_conv_const(
        "Convolution_Single", 1, 2, 1, channels, request,
        ramp_iacts=True, psum_width=32, trans_words=4,
        bias_step=7, input_height=height, asymmetric_kernel=False,
    )'
      "$fixture_run_root"
      "$candidate_width" "$candidate_height"
      "$candidate_channels" "$candidate_filters"
    )
    ;;
  packed-layer)
    test_command=(
      "$work_dir/.venv/bin/python" -c
      'import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])
cluster_rows = int(sys.argv[2])
num_glb_iact = 1 if cluster_rows == 3 else 3
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
fixture = json.loads((root / "packed-layer-wrapper.json").read_text(encoding="utf-8"))
if fixture.get("source_layer") != manifest.get("source_layer"):
    raise SystemExit("Packed-layer fixture source identity mismatch")
os.environ["OPENEYE_WGHT_PATTERN_JSON"] = str(root / "packed-layer-wrapper.json")
os.environ["OPENEYE_CONST_IACTS"] = "1"
os.environ["OPENEYE_BIAS_STEP"] = "7"
execution = fixture["execution"]
layer_name = str(manifest["source_layer"])
request = SimpleNamespace(
    node=SimpleNamespace(nodeid="domino_packed_layer_" + layer_name)
)

if os.environ.get("OPENEYE_RUNTIME_CHANNEL_TILES") == "1":
    from domino_network_tiles import run_tiles
    if fixture["layer_type"] == "conv2d":
        import test_conv_const as test_module
    elif fixture["layer_type"] == "linear":
        import test_gemm_layer as test_module
    else:
        raise SystemExit("Unsupported runtime layer type")
    run_tiles(test_module, fixture, root / "packed-layer-wrapper.json", Path(sys.argv[3]) / "spatial-tiles")
    raise SystemExit(0)

if fixture["layer_type"] == "conv2d":
    import test_conv_const as test_module

    if cluster_rows == 3 and execution["kernel_width"] in (1, 3, 7):
        from domino_spatial_tiles import run_tiles
        run_tiles(test_module, fixture, root / "packed-layer-wrapper.json", Path(sys.argv[3]) / "spatial-tiles")
        raise SystemExit(0)
    test_module.INPUT_SIZE_X = int(execution["input_width"])
    test_module.INPUT_SIZE_Y = int(execution["input_height"])
    test_module.NUM_FILTERS = int(execution["output_channels"])
    test_module.KERNEL_SIZE_X = int(execution["kernel_width"])
    test_module.KERNEL_SIZE_Y = int(execution["kernel_height"])
    test_module.STRIDE = int(execution["stride"])
    test_module._run_conv_const(
        "Convolution_Single", 1, cluster_rows, 1, int(execution["input_channels"]), request,
        ramp_iacts=False, psum_width=32, trans_words=4,
        bias_step=7, input_height=int(execution["input_height"]),
        asymmetric_kernel=False,
    )
elif fixture["layer_type"] == "linear":
    import test_gemm_layer as test_module

    test_module.test_gemm_layer(
        INPUT_SIZE=int(execution["input_features"]),
        OUTPUT_SIZE=int(execution["output_features"]),
        CLUSTER_ROWS=cluster_rows,
        NUM_GLB_IACT=num_glb_iact,
        NUM_GLB_PSUM=4,
        NUM_GLB_WGHT=3,
        DATAFLOW=str(execution["dataflow"]),
        PARALLEL_MACS=2,
        CLUSTER_COLUMNS=2,
        request=request,
    )
else:
    raise SystemExit("Unsupported packed layer type: " + str(fixture["layer_type"]))'
      "$fixture_run_root"
      "$cluster_rows"
      "$output_dir"
    )
    ;;
esac

set +e
(
  cd -- "$work_dir/test/cocotb_fpga"
  # Match prepare_openeye_rtl.sh's fixed first-layer architecture.  Four
  # activation cells need 6272 addresses for 56x56x64 and wrap a 12-bit RAM;
  # eight cells need 3136 addresses and preserve the complete input tensor.
  IACT_RAM_CELLS="$FIXED_IACT_RAM_CELLS" \
    BUFFER_WIDTH_IACT="$FIXED_IACT_BUFFER_WIDTH" \
    BUFFER_WIDTH_IACT_STREAM_CONSTRUCTOR=16 \
    BUFFER_WIDTH_WGHT="$FIXED_WGHT_BUFFER_WIDTH" \
    OPENEYE_HW_METRICS=1 OPENEYE_ZERO_BIAS="$candidate_zero_bias" \
    OPENEYE_SIMULATOR="$simulator" \
    OPENEYE_SIM_TIMEOUT_NS="$sim_timeout_ns" \
    OPENEYE_FAIL_ON_STALL="$fail_on_stall" \
    PYTHONPATH="$python_path" \
    timeout "$timeout_seconds" "${test_command[@]}"
) 2>&1 | tee "$log_path"
test_status=${PIPESTATUS[0]}
set -e

metrics_path="$output_dir/hardware_metrics.json"
metrics_scope_args=()
if ((runtime_mode)); then metrics_scope_args+=(--correctness-only); fi
metrics_schemes=()
if ((candidate_mode)); then
  read -r -a metrics_schemes <<<"${OPENEYE_CANDIDATE_SCHEMES:-4x4 3x4 2x4}"
elif ((packed_layer_mode)); then
  metrics_schemes=("$packed_layer_scheme")
  if [[ -s $output_dir/spatial_layer_verification.json ]]; then
    spatial_count=$("$openeye_root/.venv/bin/python" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["tiles"]))' "$output_dir/spatial_layer_verification.json")
    spatial_execution=$("$openeye_root/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["execution"])' "$output_dir/spatial_layer_verification.json")
    metrics_schemes=()
    for ((tile_index=0; tile_index<spatial_count; tile_index++)); do
      metrics_schemes+=("$packed_layer_scheme")
    done
  elif [[ $architecture == full-layer-r3 && -d $output_dir/spatial-tiles ]]; then
    # An incomplete tile run is never a complete layer cost table entry.
    metrics_schemes=()
  fi
fi
metrics_status=0
metrics_validation=NOT_APPLICABLE
rtl_result=FAIL
if ((test_status == 0)); then rtl_result=PASS; fi
if [[ $mode == gemm || $mode == attention ]]; then
  {
    printf '{\n'
    printf '  "schema_version": 1,\n'
    printf '  "measurement_level": "pytest-correctness-regression",\n'
    printf '  "mode": "%s",\n' "$mode"
    printf '  "validation": {"result": "NOT_APPLICABLE", "errors": []}\n'
    printf '}\n'
  } >"$metrics_path"
else
  set +e
  "$openeye_root/.venv/bin/python" "$metrics_summarizer_snapshot" \
    --log "$log_path" --output "$metrics_path" --mode "$mode" \
    --architecture-id "$architecture_id" \
    --rtl-result "$rtl_result" \
    "${metrics_scope_args[@]}" \
    --schemes "${metrics_schemes[@]}"
  metrics_status=$?
  set -e
  if ((test_status == 0 && metrics_status != 0)); then
    test_status=97
  fi
  metrics_validation=FAIL
  if ((metrics_status == 0)); then
    metrics_validation=PASS
  fi
fi

if ((test_status == 0)); then
  result=PASS
else
  result=FAIL
fi

{
  printf 'measurement_level=rtl-wrapper-simulation\n'
  printf 'openeye_commit=%s\n' "$actual_commit"
  printf 'mode=%s\n' "$mode"
  if [[ $mode == full ]]; then
    printf 'conv_test_nodeid=%s\n' "$conv_test_nodeid"
  fi
  printf 'architecture_id=%s\n' "$architecture_id"
  printf 'cluster_rows=%s\n' "$cluster_rows"
  printf 'cluster_columns=2\npe_rows=3\nparallel_macs=2\n'
  printf 'iact_ram_cells=%s\n' "$FIXED_IACT_RAM_CELLS"
  printf 'iact_buffer_address_bits=%s\n' "$FIXED_IACT_BUFFER_WIDTH"
  printf 'wght_buffer_address_bits=%s\n' "$FIXED_WGHT_BUFFER_WIDTH"
  if [[ $mode == kernel7-smoke ]]; then
    printf 'geometry=16x16-ch4-f4-k7x7-s%s\n' "$kernel7_stride"
    printf 'source_input_channels=3\ninput_channel_padding=1\n'
    if [[ ${OPENEYE_KERNEL7_CONSTANTS:-0} == 1 ]]; then
      printf 'activation_payload=constant-one\n'
    else
      printf 'activation_payload=flat-position-ramp-mod-127-in-range-1-to-127\n'
    fi
    printf 'claim_level=synthetic-kernel7-correctness-not-mac20-full-layer\n'
  fi
  if ((fixture_mode)); then
    printf 'fixture_root=%s\n' "$fixture_root"
    printf 'fixture_manifest_sha256=%s\n' "$fixture_manifest_sha256"
    printf 'fixture_payload_mode=%s\n' "$fixture_payload_mode"
    printf 'zero_bias=%s\n' "$zero_bias_label"
  fi
  if ((candidate_mode)); then
    printf 'activation_payload=flat-position-ramp-mod-127-in-range-1-to-127\n'
    printf 'candidate_schemes=%s\n' "${OPENEYE_CANDIDATE_SCHEMES:-4x4 3x4 2x4}"
    printf 'geometry=%sx%s-ch%s-f%s\n' \
      "$candidate_width" "$candidate_height" "$candidate_channels" "$candidate_filters"
  fi
  if ((packed_layer_mode)); then
    if [[ $packed_layer_type == linear && $architecture == v6 ]]; then
      printf 'num_glb_iact=3\n'
    else
      printf 'num_glb_iact=1\n'
    fi
    printf 'num_glb_wght=3\nnum_glb_psum=4\n'
    printf 'source_layer=%s\n' "$packed_layer_name"
    printf 'layer_type=%s\n' "$packed_layer_type"
    printf 'selected_scheme=%s\n' "$packed_layer_scheme"
    if ((runtime_mode)); then
      printf 'activation_payload=verified-runtime-int8-cxy\n'
    else
      printf 'activation_payload=constant-one\n'
    fi
    printf 'geometry=%s\n' "$packed_layer_geometry"
    if ((runtime_mode)); then
      printf 'claim_level=runtime-convolution-pilot-not-network-latency-or-cost\n'
    else
      printf 'claim_level=independent-weighted-layer-rtl-correctness-not-end-to-end-inference\n'
    fi
  fi
  if [[ $mode == dense-64-smoke || $mode == dense-scale ]]; then
    printf 'geometry=%sx%s-ch%s-f%s\n' \
      "$dense_width" "$dense_height" "$dense_channels" "$dense_filters"
  fi
  printf 'timeout_seconds=%s\n' "$timeout_seconds"
  printf 'pytest_shard_index=%s\n' "$pytest_shard_index"
  printf 'pytest_shard_count=%s\n' "$pytest_shard_count"
  printf 'pytest_target=%s\n' "${conv_test_nodeid:-${gemm_test_nodeid:-$mode}}"
  printf 'sim_timeout_ns=%s\n' "$sim_timeout_ns"
  printf 'simulator=%s\n' "$simulator"
  printf 'fail_on_stall_cycles=%s\n' "$fail_on_stall"
  printf 'result=%s\n' "$result"
  printf 'exit_status=%s\n' "$test_status"
  printf 'iact_counter_patch_sha256=%s\n' "$iact_counter_patch_sha256"
  printf 'iact_write_alignment_patch_sha256=%s\n' "$iact_write_alignment_patch_sha256"
  printf 'pe_weight_count_width_patch_sha256=%s\n' "$pe_wght_count_width_patch_sha256"
  printf 'conv_bias_order_patch_sha256=%s\n' "$conv_bias_order_patch_sha256"
  printf 'serial_odd_channel_padding_patch_sha256=%s\n' "$(sha256sum "$input_snapshot_dir/patches/serial-odd-channel-padding.patch" | awk '{print $1}')"
  printf 'layer_mapper_shared_state_patch_sha256=%s\n' "$(sha256sum "$input_snapshot_dir/patches/layer-mapper-shared-state.patch" | awk '{print $1}')"
  printf 'packed_weight_count_patch_sha256=%s\n' "not-applied-upstream"
  printf 'tall_conv_sync_patch_sha256=%s\n' "$tall_conv_sync_patch_sha256"
  printf 'output_order_patch_sha256=%s\n' "$output_order_patch_sha256"
  printf 'pair_safe_weight_patch_sha256=%s\n' "$pair_safe_weight_patch_sha256"
  printf 'zero_bridge_patch_sha256=%s\n' "$zero_bridge_patch_sha256"
  printf 'hardware_metrics_patch_sha256=%s\n' "$hardware_metrics_patch_sha256"
  printf 'progress_watchdog_patch_sha256=%s\n' "$progress_watchdog_patch_sha256"
  printf 'actual_candidate_bias_patch_sha256=%s\n' "$actual_candidate_bias_patch_sha256"
  printf 'wide_psum_output_patch_sha256=%s\n' "$wide_psum_output_patch_sha256"
  printf 'wide_iact_buffer_count_patch_sha256=%s\n' "$wide_iact_buffer_count_patch_sha256"
  printf 'weight_staging_uram_patch_sha256=%s\n' "$weight_staging_uram_patch_sha256"
  printf 'select_simulator_patch_sha256=%s\n' "$select_simulator_patch_sha256"
  printf 'gemm_fixture_patch_sha256=%s\n' "$gemm_fixture_patch_sha256"
  printf 'stride_output_patch_sha256=%s\n' "$stride_output_patch_sha256"
  printf 'strided_iact_patch_sha256=%s\n' "$strided_iact_patch_sha256"
  printf 'r3_weight_send_tail_patch_sha256=%s\n' "$r3_weight_send_tail_patch_sha256"
  printf 'spatial_check_patch_sha256=%s\n' "$spatial_check_patch_sha256"
  printf 'spatial_tile_tool_sha256=%s\n' "$spatial_tile_tool_sha256"
  printf 'runtime_activation_patch_sha256=%s\n' "$runtime_activation_patch_sha256"
  printf 'runtime_activation_tool_sha256=%s\n' "$runtime_activation_tool_sha256"
  printf 'runtime_activation_input_sha256=%s\n' "$runtime_activation_input_sha256"
  printf 'runtime_projection_iact_patch_sha256=%s\n' "$runtime_projection_iact_patch_sha256"
  printf 'runtime_kernel3_iact_patch_sha256=%s\n' "$runtime_kernel3_iact_patch_sha256"
  printf 'network_tile_tool_sha256=%s\n' "$network_tile_tool_sha256"
  printf 'runtime_channel_tiles=%s\n' "$runtime_channel_tiles"
  if [[ $architecture == full-layer-r3 ]]; then
    printf 'stride_enabled=1\n'
    printf 'psum_buffer_address_bits=12\n'
    if [[ -s $output_dir/spatial_layer_verification.json ]]; then
      printf 'execution=%s\n' "$spatial_execution"
      printf 'execution_tiles=%s\n' "$spatial_count"
    fi
  fi
  printf 'metrics_summarizer_sha256=%s\n' "$metrics_summarizer_sha256"
  printf 'fixture_tool_sha256=%s\n' "$fixture_tool_sha256"
  printf 'packed_layer_fixture_tool_sha256=%s\n' "$packed_layer_fixture_tool_sha256"
  printf 'metrics_validation=%s\n' "$metrics_validation"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$output_dir/regression_manifest.txt"

(
  cd -- "$output_dir"
  checksum_files=(
    wrapper-conv-"$mode".log
    hardware_metrics.json
    regression_manifest.txt
  )
  if [[ -s pytest-results.xml ]]; then
    checksum_files+=(pytest-results.xml)
  fi
  if [[ -s spatial_layer_verification.json ]]; then
    checksum_files+=(spatial_layer_verification.json)
  fi
  if [[ -d spatial-tiles ]]; then
    while IFS= read -r spatial_report; do checksum_files+=("$spatial_report"); done \
      < <(find spatial-tiles -type f \( -name 'tile-*.json' -o -name 'junit-*.xml' \) | sort)
  fi
  sha256sum "${checksum_files[@]}" >SHA256SUMS
)

[[ -z $(git -C "$openeye_root" status --short) ]] ||
  die 'Pinned OpenEye checkout changed during regression.'
if ((test_status != 0)); then
  printf '[openeye-wrapper-regression] FAIL: mode=%s output=%s\n' \
    "$mode" "$output_dir" >&2
  exit "$test_status"
fi
printf '[openeye-wrapper-regression] PASS: mode=%s output=%s\n' "$mode" "$output_dir"
