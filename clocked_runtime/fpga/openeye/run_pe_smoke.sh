#!/usr/bin/env bash
# Run the mandatory PE smoke test against the pinned OpenEye checkout.
#
# Upstream removed the TRANS_BITWIDTH_WGHT parameter from hdl/PE.v in the
# e2eab3c lint cleanup but hdl/debug/PE_IO_debug.v still binds it, so
# test/cocotb_PE fails to elaborate on a pristine pin. The pinned checkout must
# stay clean (prepare_openeye_rtl.sh checks that), so apply the checked-in
# compatibility patch to a temporary copy and run the suite there, as AGENTS.md
# requires.
#
# Usage:
#   fpga/openeye/run_pe_smoke.sh OPENEYE_ROOT [LOG_PATH]
#
# LOG_PATH defaults to OPENEYE_ROOT/openeye-pe-smoke.log, which is what
# check-fpga-environment.sh looks for.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly EXPECTED_OPENEYE_COMMIT='9a742fbd42a55ebfa8fcd39aa41dfa36a367572f'
readonly DEBUG_PATCH="$SCRIPT_DIR/patches/openeye-pe-debug-wrapper.patch"
readonly PE_WGHT_COUNT_WIDTH_PATCH="$SCRIPT_DIR/patches/openeye-pe-weight-spad-count-width.patch"

die() {
  printf '[openeye-pe-smoke] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  fpga/openeye/run_pe_smoke.sh OPENEYE_ROOT [LOG_PATH]

Copy OPENEYE_ROOT to a temporary directory, apply the checked-in PE debug
wrapper compatibility patch, and run test/cocotb_PE there. The pinned checkout
itself is never modified.
EOF
}

[[ ${1:-} != -h && ${1:-} != --help ]] || {
  usage
  exit 0
}

(($# >= 1 && $# <= 2)) || {
  usage >&2
  exit 2
}

openeye_root=$(realpath -- "$1")
[[ -d $openeye_root/hdl ]] || die "OpenEye checkout not found: $openeye_root"
[[ -x $openeye_root/.venv/bin/python ]] ||
  die "OpenEye virtual environment not found: $openeye_root/.venv"
[[ -s $DEBUG_PATCH ]] || die "Debug wrapper patch not found: $DEBUG_PATCH"
[[ -s $PE_WGHT_COUNT_WIDTH_PATCH ]] || die "PE count-width patch not found: $PE_WGHT_COUNT_WIDTH_PATCH"
command -v patch >/dev/null 2>&1 || die 'patch is required.'

[[ -z $(git -C "$openeye_root" status --short) ]] ||
  die 'OpenEye has tracked/unignored changes; preserve a clean pinned checkout.'
actual_commit=$(git -C "$openeye_root" rev-parse HEAD)
[[ $actual_commit == "$EXPECTED_OPENEYE_COMMIT" ]] ||
  die "OpenEye commit mismatch: expected $EXPECTED_OPENEYE_COMMIT, got $actual_commit"

log_path=${2:-$openeye_root/openeye-pe-smoke.log}

work_dir=$(mktemp -d -t openeye-pe-smoke-XXXXXX)
trap 'rm -rf -- "$work_dir"' EXIT

# Copy the checkout without the venv (symlinked) and without simulation output.
# Archive the verified commit, excluding mutable Git metadata and ignored run files.
git -C "$openeye_root" archive "$actual_commit" | tar -xf - -C "$work_dir"
ln -sfn "$openeye_root/.venv" "$work_dir/.venv"

patch --batch --forward --fuzz=0 -p1 -d "$work_dir" <"$DEBUG_PATCH"
git -C "$work_dir" apply "$PE_WGHT_COUNT_WIDTH_PATCH"

{
  printf '# Measured on a temporary copy of %s\n' "$openeye_root"
  printf '# commit: %s\n' "$actual_commit"
  printf '# compatibility patch: %s\n' \
    "$(sha256sum "$DEBUG_PATCH" | awk '{print $1}')"
  printf '# The pinned checkout itself is unmodified.\n'
  printf '# PE count-width patch: %s\n' \
    "$(sha256sum "$PE_WGHT_COUNT_WIDTH_PATCH" | awk '{print $1}')"
} >"$log_path"

(
  cd -- "$work_dir/test/cocotb_PE"
  rm -rf .temp
  PYTHONPATH="$SCRIPT_DIR/compat:$work_dir/src" \
    timeout 900 "$work_dir/.venv/bin/python" -m pytest -q 2>&1
) >>"$log_path" || true

tail -n 3 "$log_path"

summary=$(grep -E '^=+ .*(passed|failed)' "$log_path" | tail -1)
if grep -qE '^=+ [0-9]+ passed' <<<"$summary" && ! grep -qE 'failed' <<<"$summary"; then
  printf '[openeye-pe-smoke] PASS: %s\n' "$summary"
else
  printf '[openeye-pe-smoke] FAIL: %s\n' "${summary:-no summary line}" >&2
  exit 1
fi
