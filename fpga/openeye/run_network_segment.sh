#!/usr/bin/env bash
# One checked block/stage per hosted job; authenticated DUT boundaries only.
set -euo pipefail
if [[ $# != 7 ]]; then
  echo "Usage: run_network_segment.sh OPEN_EYE BUNDLE WORK INDEX CHAIN_ID CONFIG FOCUSED_GATE" >&2
  exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
openeye=$(realpath "$1")
bundle=$(realpath "$2")
work=$3
index=$4
chain_id=$5
config=$(realpath "$6")
focused_gate=$(realpath "$7")
[[ $index =~ ^([0-9]|10|11)$ && $chain_id =~ ^[0-9]+$ ]]
test ! -e "$work"
mkdir -m 700 -- "$work"
work=$(realpath "$work")
python="$openeye/.venv/bin/python"
private_remote=${OPENEYE_NETWORK_PRIVATE_REMOTE:-domino-drive:DominoSearch-runtime-private}
sharing_policy=${OPENEYE_NETWORK_PRIVATE_POLICY:-owner-only}
"$python" "$script_dir/network_checkpoint.py" preflight --config "$config" \
  --verify-drive --remote "$private_remote" --sharing-policy "$sharing_policy"
commit=$(git -C "$script_dir" rev-parse HEAD)
remote="$private_remote/$chain_id/$commit"
context="$work/context.json"
"$python" "$script_dir/run_network_inference.py" \
  --bundle-root "$bundle" --openeye-root "$openeye" \
  --private-root "$work/context-private-unused" --evidence "$work/context-evidence-unused" \
  --mode full-network --spatial-size 56 --chain-id "$chain_id" --resume-context-only "$context"

# The workflow preflight requires an existing write grant; no OAuth/ACL edits.
rclone mkdir --config "$config" "$remote"
rclone lsf --config "$config" --files-only "$remote" > "$work/objects.txt"
cached=0
found=0
while IFS= read -r object; do
  if [[ $object == "segment-$index.aesgcm" ]]; then found=1; fi
done < "$work/objects.txt"
if [[ $index != 11 && $found == 1 ]]; then
  rclone copyto --immutable --config "$config" "$remote/segment-$index.aesgcm" "$work/cached.aesgcm"
  "$python" "$script_dir/network_checkpoint.py" unseal --source "$work/cached.aesgcm" \
    --output "$work/cached" --context "$context" --expected-segment "$index"
  cp -a -- "$work/cached/evidence" "$work/evidence"
  cached=1
else
  resume_args=()
  if ((index > 0)); then
    previous=$((index - 1))
    rclone copyto --immutable --config "$config" "$remote/segment-$previous.aesgcm" "$work/predecessor.aesgcm"
    resume_args=(--resume-checkpoint "$work/predecessor.aesgcm")
  fi
  "$python" "$script_dir/run_network_inference.py" \
    --bundle-root "$bundle" --openeye-root "$openeye" \
    --private-root "$work/private" --evidence "$work/evidence" \
    --mode full-network --spatial-size 56 --chain-id "$chain_id" \
    --segment-index "$index" "${resume_args[@]}"
  if [[ $index != 11 ]]; then
    "$python" "$script_dir/network_checkpoint.py" seal --state "$work/private/network-state.json" \
      --evidence "$work/evidence" --output "$work/completed.aesgcm" \
      --context "$context" --expected-segment "$index"
    rclone copyto --immutable --config "$config" "$work/completed.aesgcm" "$remote/segment-$index.aesgcm"
  fi
fi
if [[ $index == 11 ]]; then
  while IFS= read -r path; do
    cp -a -- "$path" "$work/evidence/"
  done < <(find "$focused_gate" -mindepth 1 -maxdepth 1 -type d -name 'focused-*')
fi
(
  cd "$work/evidence"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum
) > "$work/final-SHA256SUMS"
mv -- "$work/final-SHA256SUMS" "$work/evidence/SHA256SUMS"
if [[ $index == 11 ]]; then
  "$python" "$script_dir/verify_network_evidence.py" \
    --evidence "$work/evidence" --mode full-network --domino-commit "$commit"
fi
printf 'Network segment %s verified; reused authenticated checkpoint=%s; not board/accuracy/latency evidence\n' "$index" "$cached"
