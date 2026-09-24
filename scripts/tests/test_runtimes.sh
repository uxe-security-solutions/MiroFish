#!/usr/bin/env bash
#
# test_runtimes.sh — offline checks for the runtime profiles in provision_local.sh.
#
# Runs the real script against a throwaway copy of scripts/ with docker,
# nvidia-smi, curl, sudo and friends replaced by stubs that only record what
# they were asked to do. Nothing touches Docker, the GPU or the network, so
# this is safe on a live host and runs on a laptop (bash 3.2 or later).
#
# The first test is the important one: it pins, word for word, the containers
# the DGX Spark runtime starts. Adding or tuning another runtime must not move
# it; if it fails, the DGX build changed.
#
#   bash scripts/tests/test_runtimes.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Fail closed. The cleanup trap kills and deletes everything under $WORK, so
# $WORK must be a directory this run just created — never an empty string, and
# never the directory we were started from (which, via `provision_local.sh test`,
# is the live checkout with its weights and .env).
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sosim-runtimes.XXXXXX")" || { echo "mktemp failed; refusing to run" >&2; exit 1; }
# Physical path: on macOS $TMPDIR sits behind the /var -> /private/var symlink,
# and the script under test resolves its own ROOT physically.
WORK="$(cd "$WORK" && pwd -P)" || exit 1
case "$WORK" in
  */sosim-runtimes.??????) ;;
  *) echo "refusing to run: unexpected work directory '$WORK'" >&2; exit 1 ;;
esac
cleanup() {
  [[ -d "$WORK" && "$WORK" == */sosim-runtimes.?????? ]] || return 0
  pkill -f "$WORK/" >/dev/null 2>&1
  if [[ -n "${KEEP:-}" ]]; then echo "kept $WORK"; else rm -rf "$WORK"; fi
}
trap cleanup EXIT

PASS=0
FAILED=()
pass() { PASS=$((PASS + 1)); printf '  ok   %s\n' "$1"; }
flunk() { FAILED+=("$1"); printf '  FAIL %s\n' "$1"; [[ -n "${2:-}" ]] && printf '%s\n' "$2" | sed 's/^/         /'; }

# --- stubs --------------------------------------------------------------------

STUBS="$WORK/bin"
mkdir -p "$STUBS"

stub() { printf '#!/bin/bash\n%s\n' "$2" >"$STUBS/$1"; chmod +x "$STUBS/$1"; }

# Every stub appends its argv to $STUB_LOG. Containers are simulated with one
# state file each, so `ps`, `rm` and `run` agree with each other.
stub docker '
echo "docker $*" >>"$STUB_LOG"
# The same call with every argument bracketed, so a changed argument split
# (two words fused into one, or one split in two) cannot hide.
{ printf docker; printf " [%s]" "$@"; echo; } >>"$STUB_LOG.argv"
S="$STUB_STATE/containers"; mkdir -p "$S"
case "$1" in
  info) [[ "$*" == *Runtimes* ]] && echo "{\"nvidia\":{}}" ;;
  --version) echo "Docker version 27.0.0, build stub" ;;
  ps) name=$(printf "%s\n" "$@" | sed -n "s/^name=\^\(.*\)\$\$/\1/p")
      if [[ -f "$S/$name" ]] && { [[ " $* " == *" -aq "* ]] || [[ $(cat "$S/$name") == up ]]; }; then
        echo "id_$name"; fi ;;
  rm) for a in "${@:2}"; do [[ "$a" == -* ]] || rm -f "$S/$a"; done ;;
  stop) [[ -f "$S/$2" ]] && echo down >"$S/$2" ;;
  run) prev=""; for a in "$@"; do [[ "$prev" == --name ]] && echo up >"$S/$a"; prev="$a"; done
       [[ "$*" == *get_arch_list* ]] && echo "${STUB_ARCHES:-sm_80_sm_90}" | sed "s/_sm/ sm/g" ;;
  inspect) [[ "$*" == *.Args* && -n "${STUB_INSPECT_ARGS:-}" ]] && cat "$STUB_INSPECT_ARGS" ;;
  volume) exit 1 ;;
  manifest) echo "{\"manifests\":[{\"platform\":{\"architecture\": \"arm64\"}},{\"platform\":{\"architecture\": \"amd64\"}}]}" ;;
esac
exit 0'
stub curl '
echo "curl $*" >>"$STUB_LOG"
[[ "$*" == */v1/embeddings* ]] && printf "{\"data\":[{\"embedding\":[0.1]}]}"
exit 0'
stub nvidia-smi '
echo "nvidia-smi $*" >>"$STUB_LOG"
case "$STUB_GPU" in
  gb10) name="NVIDIA GB10"; cc=12.1; total="[N/A]"; used="[N/A]"; free="[N/A]" ;;
  l40s) name="NVIDIA L40S-48C"; cc=8.9; total=49152; used=178; free=${STUB_FREE:-48974} ;;
  none) echo "NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver." >&2; exit 9 ;;
  hang) exit 124 ;;   # what run_bounded reports when the query timed out
esac
for a in "$@"; do case "$a" in
  --query-gpu=*) out=""
    IFS=, read -ra fields <<<"${a#--query-gpu=}"
    for f in "${fields[@]}"; do
      case "$f" in
        name) v="$name" ;; compute_cap) v="$cc" ;; memory.total) v="$total" ;;
        memory.used) v="$used" ;; memory.free) v="$free" ;; *) v="[N/A]" ;;
      esac
      out="${out:+$out, }$v"
    done
    echo "$out"; exit 0 ;;
  --query-compute-apps=*) exit 0 ;;
  -q) [[ "$STUB_GPU" == l40s ]] && printf "    GPU Virtualization Mode\n        Virtualization Mode : VGPU\n    vGPU Software Licensed Product\n        License Status : %s\n" "${STUB_LICENSE:-Licensed (Expiry: 2027-1-1 0:0:0 GMT)}"
      exit 0 ;;
esac; done
echo "$name"'
stub uname 'case "$1" in -s) echo Linux ;; *) echo "$STUB_ARCH" ;; esac'
stub sudo 'echo "sudo $*" >>"$STUB_LOG"; [[ "$1" == -n ]] && exit 1; exit 0'
stub sleep 'exit 0'
# Deterministic stand-ins: the real ones would race the no-op sleep above
# (run_bounded's fallback kills its child after "N seconds" of instant sleeps).
stub timeout 'shift; exec "$@"'
stub setsid 'exec "$@"'
stub hostname 'echo 10.0.0.5'
stub df 'echo "Filesystem 1024-blocks Used Available Capacity Mounted"; echo "/dev/stub 999999999 1 524288000 1% /"'
stub dpkg 'exit 0'
stub node 'echo v22.0.0'
for t in uv npm git hf apt-get; do stub "$t" "echo \"$t \$*\" >>\"\$STUB_LOG\"; exit 0"; done

# Refuse to run at all unless every command the script could use to change
# something resolves to a stub.
for t in docker sudo nvidia-smi curl; do
  if [[ "$(PATH="$STUBS:/usr/bin:/bin" command -v "$t")" != "$STUBS/$t" ]]; then
    echo "refusing to run: '$t' does not resolve to the stub" >&2
    exit 1
  fi
done

# --- sandbox ------------------------------------------------------------------

# run <name> <gpu:gb10|l40s> <arch> <entry script> <args...>
# Leaves $WORK/<name>/{out,rc,calls} behind. A seed .env can be supplied in
# $SEED_ENV. A sandbox keeps its state (for chained commands) if it exists.
run() {
  local name="$1" gpu="$2" arch="$3" entry="$4"; shift 4
  local box="$WORK/$name"
  if [[ ! -d "$box/sb" ]]; then
    mkdir -p "$box/sb" "$box/state"
    cp -R "$REPO/scripts" "$box/sb/scripts"
    cp "$REPO/.env.example" "$box/sb/.env.example"
    mkdir -p "$box/sb/backend/.venv/bin" "$box/sb/third_party/graphiti/server/.venv/bin" \
             "$box/sb/third_party/graphiti/server/graph_service/zep_compat" "$box/sb/frontend/node_modules"
    touch "$box/sb/third_party/graphiti/server/graph_service/zep_compat/router.py"
    local py
    for py in "$box/sb/backend/.venv/bin/python" "$box/sb/third_party/graphiti/server/.venv/bin/python"; do
      printf '#!/bin/bash\necho "python[cuda=${CUDA_VISIBLE_DEVICES-unset}][hf=${HF_HOME-unset}] $*" >>"$STUB_LOG"\n[[ " $* " == *" --preflight-only "* || "$1 $2" == "-m pytest" ]] && exit 0\nexec /bin/sleep 30\n' >"$py"
      chmod +x "$py"
    done
  fi
  [[ -n "${SEED_ENV:-}" ]] && cp "$SEED_ENV" "$box/sb/.env"
  : >"$box/calls"; : >"$box/calls.argv"
  ( cd "$box/sb" && env -i PATH="$STUBS:/usr/bin:/bin" HOME="$box" TERM=dumb \
      STUB_LOG="$box/calls" STUB_STATE="$box/state" STUB_GPU="$gpu" STUB_ARCH="$arch" \
      ${EXTRA_ENV:-} bash "$entry" "$@" ) >"$box/out" 2>&1
  echo $? >"$box/rc"
  pkill -f "$box/sb/" >/dev/null 2>&1 || true
}

# The `docker run` line that created one container, with the sandbox path
# replaced so it can be compared across sandboxes. run_argv is the same line
# with argument boundaries kept.
run_line() { grep -E "^docker run .*--name $2( |$)" "$WORK/$1/calls" | sed "s#$WORK/$1/sb#<ROOT>#g"; }
run_argv() { grep -F "docker [run] [-d] [--name] [$2] " "$WORK/$1/calls.argv" | sed "s#$WORK/$1/sb#<ROOT>#g"; }
# "a b c" -> "[a] [b] [c]"; for pins, which contain no brackets or spaces inside arguments.
bracket() { local w out=""; for w in $1; do out="${out:+$out }[$w]"; done; printf '%s' "$out"; }

expect_eq() { # <label> <expected> <actual>
  if [[ "$2" == "$3" ]]; then pass "$1"; else flunk "$1" "expected: $2
actual:   $3"; fi
}
expect_has() { # <label> <haystack> <needle>
  if [[ "$2" == *"$3"* ]]; then pass "$1"; else flunk "$1" "missing: $3
in: $2"; fi
}
expect_lacks() {
  if [[ "$2" != *"$3"* ]]; then pass "$1"; else flunk "$1" "unexpected: $3"; fi
}

# --- 1. the DGX Spark containers, word for word --------------------------------

echo "dgx-spark"
DGX_EMBED='docker run -d --name sosim-embed --restart unless-stopped --gpus all --ipc=host -p 127.0.0.1:8081:8000 -v <ROOT>/data/hf-cache:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 nvcr.io/nvidia/vllm:26.05.post1-py3 vllm serve BAAI/bge-m3 --runner pooling --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.08 --max-model-len 8192'
DGX_LLM='docker run -d --name sosim-llm --restart unless-stopped --gpus all --ipc=host -p 127.0.0.1:8000:8000 -v <ROOT>/data/hf-cache:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 nvcr.io/nvidia/vllm:26.05.post1-py3 vllm serve RedHatAI/Qwen3.6-35B-A3B-NVFP4 --served-model-name local-llm --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.70 --max-model-len 32768 --max-num-seqs 16 --enable-auto-tool-choice --tool-call-parser qwen3_xml'

# A host that predates runtime profiles: an .env with no SOSIM_RUNTIME, and the
# bare script. It must get exactly the DGX containers.
grep -v '^SOSIM_RUNTIME=' "$REPO/.env.example" >"$WORK/legacy.env"
SEED_ENV="$WORK/legacy.env" run legacy gb10 aarch64 scripts/provision_local.sh start
expect_eq "legacy host, bare script: start exits 0" 0 "$(cat "$WORK/legacy/rc")"
expect_eq "legacy host: embeddings container unchanged" "$DGX_EMBED" "$(run_line legacy sosim-embed)"
expect_eq "legacy host: LLM container unchanged" "$DGX_LLM" "$(run_line legacy sosim-llm)"
expect_eq "legacy host: embeddings argv unchanged, argument for argument" \
  "docker $(bracket "${DGX_EMBED#docker }")" "$(run_argv legacy sosim-embed)"
expect_eq "legacy host: LLM argv unchanged, argument for argument" \
  "docker $(bracket "${DGX_LLM#docker }")" "$(run_argv legacy sosim-llm)"
expect_has "legacy host: page cache is still dropped" "$(cat "$WORK/legacy/calls")" "sudo -n sh -c echo 3 > /proc/sys/vm/drop_caches"
expect_lacks "legacy host: backend started as before, GPU not hidden" \
  "$(grep 'started backend' "$WORK/legacy/sb/data/logs/backend.log" | tail -1)" "CUDA_VISIBLE_DEVICES"
expect_lacks "legacy host: shim started as before, GPU not hidden" \
  "$(grep 'started zep-shim' "$WORK/legacy/sb/data/logs/zep-shim.log" | tail -1)" "CUDA_VISIBLE_DEVICES"
# Parallel start: both servers are created before either is waited on.
order=$(grep -nE '^docker run .*sosim-llm|^curl .*:8081/v1/models' "$WORK/legacy/calls" | head -1)
expect_has "legacy host: LLM launched before the embeddings wait" "$order" "sosim-llm"

expect_has "legacy host: GB10 is detected as dgx-spark" "$(cat "$WORK/legacy/out")" "Runtime: dgx-spark (detected"

# No GPU answer at all: the documented default, still the same containers.
SEED_ENV="$WORK/legacy.env" run nogpu none aarch64 scripts/provision_local.sh start
expect_has "unqueryable GPU falls back to dgx-spark" "$(cat "$WORK/nogpu/out")" "Runtime: dgx-spark (default)"
expect_eq "unqueryable GPU: LLM container unchanged" "$DGX_LLM" "$(run_line nogpu sosim-llm)"

run dgx-setup gb10 aarch64 scripts/provision_dgx_spark.sh setup
expect_eq "dgx-spark setup exits 0" 0 "$(cat "$WORK/dgx-setup/rc")"
expect_eq "dgx-spark setup adds only SOSIM_RUNTIME and the db path to .env.example" \
  "SOSIM_RUNTIME=dgx-spark ZEP_COMPAT_DB_PATH=$WORK/dgx-setup/sb/data/zep_compat.sqlite3" \
  "$(diff "$WORK/dgx-setup/sb/.env.example" "$WORK/dgx-setup/sb/.env" | sed -n 's/^> \([A-Z_]*=.*\)/\1/p' | tr '\n' ' ' | sed 's/ $//')"
expect_has "dgx-spark setup downloads the same weights" "$(cat "$WORK/dgx-setup/calls")" "hf download RedHatAI/Qwen3.6-35B-A3B-NVFP4"
expect_lacks "dgx-spark setup pins no revision" "$(cat "$WORK/dgx-setup/calls")" "--revision"

SEED_ENV="$WORK/legacy.env" run wrapper gb10 aarch64 scripts/provision_dgx_spark.sh start
expect_eq "provision_dgx_spark.sh: embeddings container identical" "$DGX_EMBED" "$(run_line wrapper sosim-embed)"
expect_eq "provision_dgx_spark.sh: LLM container identical" "$DGX_LLM" "$(run_line wrapper sosim-llm)"
expect_eq "provision_dgx_spark.sh: LLM argv identical, argument for argument" \
  "docker $(bracket "${DGX_LLM#docker }")" "$(run_argv wrapper sosim-llm)"

# --- 2. l40s ------------------------------------------------------------------

echo "l40s"
# What the l40s profile asks for, read from the profile itself so that tuning
# it does not break this test — only the wiring is checked here.
eval "$(env -i bash -c 'source "$1"; for k in VLLM_IMAGE LLM_MODEL_REPO GPU_MEM_UTIL EMBED_GPU_MEM_UTIL MAX_MODEL_LEN MAX_NUM_SEQS TOOL_CALL_PARSER LLM_EXTRA_ARGS EMBED_EXTRA_ARGS VLLM_ENTRYPOINT; do printf "L40S_%s=%q\n" "$k" "${!k:-}"; done' _ "$REPO/scripts/runtimes/l40s.sh")"

unset SEED_ENV
run l40s l40s x86_64 scripts/provision_l40s.sh setup
expect_eq "setup exits 0" 0 "$(cat "$WORK/l40s/rc")"
expect_has "setup records the runtime in .env" "$(cat "$WORK/l40s/sb/.env")" "SOSIM_RUNTIME=l40s"
expect_has "setup downloads the l40s weights" "$(cat "$WORK/l40s/calls")" "hf download $L40S_LLM_MODEL_REPO"
expect_has "setup seeds the thinking switch into a new .env" "$(cat "$WORK/l40s/sb/.env")" "SIM_MODEL_EXTRA_BODY='{\"chat_template_kwargs\":{\"enable_thinking\":false}}'"

# The bare script now picks the recorded runtime up by itself.
run l40s l40s x86_64 scripts/provision_local.sh start
expect_eq "bare start after l40s setup exits 0" 0 "$(cat "$WORK/l40s/rc")"
llm=$(run_line l40s sosim-llm)
embed=$(run_line l40s sosim-embed)
if [[ "$L40S_VLLM_ENTRYPOINT" == vllm ]]; then
  expect_has "LLM enters through vllm" "$llm" "--entrypoint vllm $L40S_VLLM_IMAGE serve $L40S_LLM_MODEL_REPO "
  expect_has "embeddings enter through vllm" "$embed" "--entrypoint vllm $L40S_VLLM_IMAGE serve BAAI/bge-m3 "
  expect_lacks "no doubled 'vllm serve'" "$llm" "serve vllm serve"
else
  expect_has "LLM uses the image entrypoint" "$llm" "$L40S_VLLM_IMAGE vllm serve $L40S_LLM_MODEL_REPO "
fi
expect_has "LLM memory fraction" "$llm" "--gpu-memory-utilization $L40S_GPU_MEM_UTIL "
expect_has "LLM context" "$llm" "--max-model-len $L40S_MAX_MODEL_LEN "
expect_has "LLM batch" "$llm" "--max-num-seqs $L40S_MAX_NUM_SEQS "
expect_has "LLM tool parser" "$llm" "--tool-call-parser $L40S_TOOL_CALL_PARSER"
[[ -n "$L40S_LLM_EXTRA_ARGS" ]] && expect_has "LLM extra flags" "$llm" " $L40S_LLM_EXTRA_ARGS"
expect_has "embeddings memory fraction" "$embed" "--gpu-memory-utilization $L40S_EMBED_GPU_MEM_UTIL "
[[ -n "$L40S_EMBED_EXTRA_ARGS" ]] && expect_has "embeddings extra flags" "$embed" " $L40S_EMBED_EXTRA_ARGS"
expect_lacks "no page-cache drop on a discrete card" "$(cat "$WORK/l40s/calls")" "drop_caches"
expect_has "backend runs with the GPU hidden" "$(grep 'started backend' "$WORK/l40s/sb/data/logs/backend.log" | tail -1)" "env CUDA_VISIBLE_DEVICES= "
expect_has "the shim runs with the GPU hidden" "$(grep 'started zep-shim' "$WORK/l40s/sb/data/logs/zep-shim.log" | tail -1)" "env CUDA_VISIBLE_DEVICES= "
# Serial start: the embeddings server is healthy before the LLM is created.
order=$(grep -nE '^docker run .*sosim-llm|^curl .*:8081/v1/models' "$WORK/l40s/calls" | head -1)
expect_has "embeddings waited on before the LLM is created" "$order" "8081/v1/models"

# --- 3. selection rules -------------------------------------------------------

echo "selection"
run l40s l40s x86_64 scripts/provision_dgx_spark.sh start
expect_eq "explicit runtime contradicting .env is refused" 1 "$(cat "$WORK/l40s/rc")"
expect_has "  ...and says which wrapper to use" "$(cat "$WORK/l40s/out")" "scripts/provision_l40s.sh start"
expect_lacks "  ...before creating any container" "$(cat "$WORK/l40s/calls")" "docker run"

run l40s l40s x86_64 scripts/provision_dgx_spark.sh status
expect_eq "a contradicting runtime only warns for status" 0 "$(cat "$WORK/l40s/rc")"
expect_has "  ...and does warn" "$(cat "$WORK/l40s/out")" "this host was set up for runtime 'l40s'"

run bogus gb10 aarch64 scripts/provision_local.sh --runtime nope doctor
expect_eq "unknown runtime is refused" 1 "$(cat "$WORK/bogus/rc")"
expect_has "  ...naming the ones that exist" "$(cat "$WORK/bogus/out")" "dgx-spark"

run list gb10 aarch64 scripts/provision_local.sh runtimes
expect_has "runtimes lists dgx-spark" "$(cat "$WORK/list/out")" "dgx-spark"
expect_has "runtimes lists l40s" "$(cat "$WORK/list/out")" "l40s"

SEED_ENV="$WORK/legacy.env" run detect l40s x86_64 scripts/provision_local.sh runtimes
expect_has "an L40S with nothing recorded is detected as l40s" "$(cat "$WORK/detect/out")" "bare commands use l40s (detected from the GPU)"
expect_has "runtimes names the real DGX wrapper" "$(cat "$WORK/detect/out")" "run: scripts/provision_dgx_spark.sh"

SEED_ENV="$WORK/legacy.env" run detect-stop l40s x86_64 scripts/provision_local.sh stop
expect_lacks "stop does not probe the GPU" "$(cat "$WORK/detect-stop/calls")" "nvidia-smi"

SEED_ENV="$WORK/legacy.env" run mismatch l40s x86_64 scripts/provision_local.sh --runtime dgx-spark doctor
expect_has "doctor flags dgx-spark forced onto an L40S" "$(cat "$WORK/mismatch/out")" "has compute capability 8.9; runtime 'dgx-spark' was built for"
expect_has "  ...and the CPU architecture" "$(cat "$WORK/mismatch/out")" "this host is x86_64"
expect_has "doctor checks amd64 manifests on x86_64" "$(cat "$WORK/mismatch/out")" "amd64 manifests"

# doctor on the l40s runtime: the torch probe must not go through the image's
# `vllm serve` entrypoint, and PyTorch's sm_86 build must count for an 8.9 card.
SEED_ENV="$WORK/l40s/sb/.env" EXTRA_ENV="STUB_ARCHES=sm_75_sm_80_sm_86_sm_90" \
  run l40s-doctor l40s x86_64 scripts/provision_l40s.sh doctor
probe=$(grep -E '^docker run --rm .*get_arch_list' "$WORK/l40s-doctor/calls")
if [[ "$L40S_VLLM_ENTRYPOINT" == vllm ]]; then
  expect_has "doctor probes torch through python3, not the entrypoint" "$probe" "--entrypoint python3 $L40S_VLLM_IMAGE -c"
fi
expect_has "doctor accepts sm_86 kernels on an 8.9 card" "$(cat "$WORK/l40s-doctor/out")" "sm_86 present"
expect_has "doctor preflight runs with the GPU hidden" "$(cat "$WORK/l40s-doctor/calls")" "python[cuda=]["
expect_has "doctor preflight gets HF_HOME pinned to the repo root" "$(cat "$WORK/l40s-doctor/calls")" "[hf=$WORK/l40s-doctor/sb/data/hf-cache] $WORK/l40s-doctor/sb/backend/scripts/run_parallel_simulation.py --preflight-only"
expect_has "doctor warns when the Twitter model is not where simulations look" "$(cat "$WORK/l40s-doctor/out")" "Twitter/twhin-bert-base is not under HF_HOME=$WORK/l40s-doctor/sb/data/hf-cache"
mkdir -p "$WORK/l40s-doctor/sb/data/hf-cache/hub/models--Twitter--twhin-bert-base"
SEED_ENV="$WORK/l40s/sb/.env" run l40s-doctor l40s x86_64 scripts/provision_l40s.sh doctor
expect_has "  ...and confirms it when it is" "$(cat "$WORK/l40s-doctor/out")" "Twitter recommender model cached under HF_HOME="
expect_has "doctor reports the vGPU license" "$(cat "$WORK/l40s-doctor/out")" "vGPU license: Licensed"

SEED_ENV="$WORK/l40s/sb/.env" EXTRA_ENV="STUB_LICENSE=Unlicensed" \
  run unlicensed l40s x86_64 scripts/provision_l40s.sh doctor
expect_has "doctor flags an unlicensed vGPU" "$(cat "$WORK/unlicensed/out")" "vGPU license status is 'Unlicensed'"

SEED_ENV="$WORK/l40s/sb/.env" EXTRA_ENV="LLM_MODEL_REVISION=abc123" \
  run rev l40s x86_64 scripts/provision_l40s.sh start
expect_has "LLM_MODEL_REVISION reaches vLLM" "$(run_line rev sosim-llm)" "--revision abc123 --tokenizer-revision abc123"

SEED_ENV="$WORK/l40s/sb/.env" EXTRA_ENV="STUB_FREE=30000" \
  run tight l40s x86_64 scripts/provision_l40s.sh start
expect_has "start reports a card without room for both servers" "$(cat "$WORK/tight/out")" "vLLM will refuse to start"
expect_has "  ...as a problem with the host, not a service that failed" "$(cat "$WORK/tight/out")" "problem(s) were reported before startup"
SEED_ENV="$WORK/l40s/sb/.env" EXTRA_ENV="STUB_FREE=42500" \
  run thin l40s x86_64 scripts/provision_l40s.sh start
expect_has "thin but sufficient memory only warns" "$(cat "$WORK/thin/out")" "enough to start"
expect_eq "  ...and start still exits 0" 0 "$(cat "$WORK/thin/rc")"

# The wrong wrapper on a host that has recorded nothing: refused before anything
# is created, because the hardware positively matches the other profile.
SEED_ENV="$WORK/legacy.env" run wrong-wrapper gb10 aarch64 scripts/provision_l40s.sh start
expect_eq "provision_l40s.sh on a DGX Spark is refused" 1 "$(cat "$WORK/wrong-wrapper/rc")"
expect_has "  ...naming the right wrapper" "$(cat "$WORK/wrong-wrapper/out")" "scripts/provision_dgx_spark.sh start"
expect_lacks "  ...before creating any container" "$(cat "$WORK/wrong-wrapper/calls")" "docker run"
expect_has "  ...and says how to force it, keeping the runtime" "$(cat "$WORK/wrong-wrapper/out")" "SOSIM_FORCE_RUNTIME=1 scripts/provision_l40s.sh start"
SEED_ENV="$WORK/legacy.env" EXTRA_ENV="SOSIM_FORCE_RUNTIME=1" run forced gb10 aarch64 scripts/provision_l40s.sh start
expect_has "SOSIM_FORCE_RUNTIME=1 overrides the refusal" "$(cat "$WORK/forced/calls")" "docker run -d --name sosim-llm"

# An .env carried over from the other machine: its record is foreign here.
{ cat "$WORK/legacy.env"; echo "SOSIM_RUNTIME=l40s"; } >"$WORK/foreign.env"
SEED_ENV="$WORK/foreign.env" run foreign gb10 aarch64 scripts/provision_local.sh start
expect_eq "a DGX whose .env records l40s refuses to start it" 1 "$(cat "$WORK/foreign/rc")"
expect_lacks "  ...before creating any container" "$(cat "$WORK/foreign/calls")" "docker run"
SEED_ENV="$WORK/foreign.env" run foreign2 gb10 aarch64 scripts/provision_dgx_spark.sh start
expect_has "  ...and the DGX wrapper calls the record foreign" "$(cat "$WORK/foreign2/out")" "the record is probably wrong"

run emptyrt gb10 aarch64 scripts/provision_local.sh --runtime= status
expect_eq "--runtime= with no name is refused" 1 "$(cat "$WORK/emptyrt/rc")"

# A host switching to l40s (record deleted, .env otherwise kept) gets the seed.
SEED_ENV="$WORK/legacy.env" run switch l40s x86_64 scripts/provision_l40s.sh setup
expect_has "setup seeds the thinking switch when it first records l40s" "$(cat "$WORK/switch/sb/.env")" "SIM_MODEL_EXTRA_BODY='{"
# The offline wipe path (a copied .env, straight to start) is warned about.
SEED_ENV="$WORK/legacy.env" run noseed l40s x86_64 scripts/provision_l40s.sh start
expect_has "start warns when the l40s thinking switch is missing" "$(cat "$WORK/noseed/out")" "expects SIM_MODEL_EXTRA_BODY in .env"

# A driver that does not answer: one timed-out query, then no more probing.
SEED_ENV="$WORK/legacy.env" run hang hang aarch64 scripts/provision_local.sh start
expect_has "a hung nvidia-smi is reported, not waited on again" "$(cat "$WORK/hang/out")" "did not answer within 20s"
expect_lacks "  ...and nvidia-smi -q is skipped" "$(cat "$WORK/hang/calls")" "nvidia-smi -q"
expect_eq "  ...and the DGX containers still start unchanged" "$DGX_LLM" "$(run_line hang sosim-llm)"

# A blanked (not deleted) record is re-recorded by setup.
{ cat "$WORK/legacy.env"; echo "SOSIM_RUNTIME="; } >"$WORK/blank.env"
SEED_ENV="$WORK/blank.env" run blank l40s x86_64 scripts/provision_l40s.sh setup
expect_eq "setup records the runtime over a blanked SOSIM_RUNTIME= line" "l40s" \
  "$(grep -E '^SOSIM_RUNTIME=' "$WORK/blank/sb/.env" | tail -1 | cut -d= -f2-)"

# An .env that already has the seed, unquoted-equal: no false "suggests" note.
{ cat "$WORK/legacy.env"; echo "SIM_MODEL_EXTRA_BODY='{\"chat_template_kwargs\":{\"enable_thinking\":false}}'"; } >"$WORK/seeded.env"
SEED_ENV="$WORK/seeded.env" run seeded l40s x86_64 scripts/provision_l40s.sh setup
expect_lacks "an identical seed is not reported as a mismatch" "$(cat "$WORK/seeded/out")" "SIM_MODEL_EXTRA_BODY={"

# .env overrides the model: doctor judges the live container by what start used.
{ cat "$WORK/l40s/sb/.env"; echo "LLM_MODEL_REPO=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"; } >"$WORK/awq.env"
SEED_ENV="$WORK/awq.env" run awq l40s x86_64 scripts/provision_l40s.sh start
expect_has "a model set in .env reaches the container" "$(run_line awq sosim-llm)" "serve cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit "
printf '%s\n' "$(run_line awq sosim-llm | sed 's/^docker run .* serve /serve /')" >"$WORK/awq/state/args"
EXTRA_ENV="STUB_INSPECT_ARGS=$WORK/awq/state/args" run awq l40s x86_64 scripts/provision_l40s.sh doctor
expect_lacks "doctor does not call that container stale" "$(cat "$WORK/awq/out")" "does not serve"

# Recreating only the LLM: the embeddings server's real usage is already out of
# "free", so the check asks for the LLM's share alone.
rm -f "$WORK/l40s/state/containers/sosim-llm"
EXTRA_ENV="STUB_FREE=38000" run l40s l40s x86_64 scripts/provision_l40s.sh start
expect_has "recreating the LLM alone is still memory-checked" "$(cat "$WORK/l40s/out")" "vLLM will refuse to start"
expect_lacks "  ...for the LLM's share only" "$(cat "$WORK/l40s/out")" "and the embeddings server need"

# Every profile must load and validate on its own.
for profile in "$REPO"/scripts/runtimes/*.sh; do
  name="$(basename "$profile" .sh)"
  run "profile-$name" gb10 aarch64 scripts/provision_local.sh --runtime "$name" status
  expect_eq "profile $name loads" 0 "$(cat "$WORK/profile-$name/rc")"
done

# --- summary ------------------------------------------------------------------

echo
if (( ${#FAILED[@]} == 0 )); then
  echo "all $PASS checks passed"
  exit 0
fi
echo "${#FAILED[@]} of $((PASS + ${#FAILED[@]})) checks FAILED:"
printf '  - %s\n' "${FAILED[@]}"
exit 1
