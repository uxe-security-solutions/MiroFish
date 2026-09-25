#!/usr/bin/env bash
#
# provision_local.sh — bring up SoSim entirely on this machine.
#
# Everything hardware-specific lives in a runtime profile, scripts/runtimes/<name>.sh:
#
#   dgx-spark  NVIDIA DGX Spark (GB10, aarch64, 128GB unified memory) — the default
#   l40s       NVIDIA L40S 48GB (Ada, x86_64), including the L40S-48C vGPU
#
# Select one with a wrapper (scripts/provision_dgx_spark.sh, scripts/provision_l40s.sh),
# with --runtime <name>, or with SOSIM_RUNTIME=<name>. `setup` records the choice
# in .env, so later commands on the same host get the same runtime back; with
# nothing recorded, the profile matching this host's GPU is used.
#
#   ./scripts/provision_local.sh setup     # deps, submodule, .env, models  (needs network)
#   ./scripts/provision_local.sh start     # bring every service up        (offline)
#   ./scripts/provision_local.sh all       # setup + start
#   ./scripts/provision_local.sh status | logs [svc] | stop | doctor | test | runtimes
#
# Add -v (or VERBOSE=1) to echo every external command as it runs.
#
# Failures never pass silently: each one is printed as it happens, the offending
# service's log tail is dumped, and every failure is listed again at exit with a
# non-zero status.
#
# Nothing here talks to a hosted API at runtime. `setup` is the only stage that
# needs the internet, and only to download packages and model weights.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# uv and `uv tool install` put binaries here. Export unconditionally: doing it
# only inside install_uv meant that on a second run (uv already present) the
# path was never added, and the `hf` CLI installed later could not be found.
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

DATA_DIR="$ROOT/data"
RUN_DIR="$DATA_DIR/run"
LOG_DIR="$DATA_DIR/logs"
HF_CACHE="${HF_CACHE_DIR:-$DATA_DIR/hf-cache}"

# What the user typed to get here. The per-runtime wrappers exec this script and
# set it, so every "run: ..." hint names the wrapper that selected the runtime,
# not this file — which, run bare on a host with no recorded runtime, would
# fall back to dgx-spark.
SELF="${SOSIM_ENTRYPOINT:-$0}"

# --- tunables (override via environment) -------------------------------------
#
# Only what is the same on every runtime is set here. The image, the weights,
# the memory fractions, the batch size and the GPU the build expects come from
# the runtime profile, which load_runtime sources once the command is known.

# Runtime profiles, and the one used when nothing selects a runtime. It stays
# dgx-spark because that is what every host provisioned before profiles existed
# was running, and such a host has no SOSIM_RUNTIME recorded anywhere.
RUNTIME_DIR="$ROOT/scripts/runtimes"
DEFAULT_RUNTIME=dgx-spark

FALKORDB_IMAGE="${FALKORDB_IMAGE:-falkordb/falkordb:latest}"
FALKORDB_VOLUME="${FALKORDB_VOLUME:-sosim_falkordb}"

# Names this stack used before the product was renamed to SoSim. The containers
# ran with --restart unless-stopped, so they come back after a reboot and keep
# holding the GPU and ports 8000/8081/6379; the sosim-* containers cannot bind
# while they are alive, so retire_legacy_infra removes them before start. The
# volume is only reported, never removed — deleting an operator's disk is not
# this script's call, even when the data in it is already written off.
LEGACY_CONTAINERS=(mirofish-llm mirofish-embed mirofish-falkordb)
LEGACY_VOLUME=mirofish_falkordb

LLM_SERVED_NAME="${LLM_SERVED_NAME:-local-llm}"
EMBED_MODEL_REPO="${EMBED_MODEL_REPO:-BAAI/bge-m3}"

LLM_PORT="${LLM_PORT:-8000}"
EMBED_PORT="${EMBED_PORT:-8081}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
FALKORDB_UI_PORT="${FALKORDB_UI_PORT:-3001}"
SHIM_PORT="${SHIM_PORT:-8088}"
BACKEND_PORT="${BACKEND_PORT:-5001}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

# How containers are given the GPU. `--gpus all` suits the legacy nvidia
# runtime; hosts wired up through CDI may need `--device nvidia.com/gpu=all`
# instead. Override if the LLM container reports "could not select device
# driver".
GPU_FLAGS="${GPU_FLAGS:---gpus all}"

NODE_MAJOR_REQUIRED=20   # vite 7 needs ^20.19 || >=22.12

# Health-gate patience, in 2-second polls. Raise these on slower storage (a
# first model load reads tens of GB); lower them to fail fast while testing.
EMBED_WAIT_TRIES="${EMBED_WAIT_TRIES:-180}"      # ~6 min
# The LLM's own patience is left to the runtime profile, which knows how long a
# first load (weights, compile, CUDA graphs) takes there; load_runtime falls
# back to 300 (~10 min).
LLM_WAIT_TRIES="${LLM_WAIT_TRIES:-}"
SHIM_WAIT_TRIES="${SHIM_WAIT_TRIES:-90}"         # ~3 min
BACKEND_WAIT_TRIES="${BACKEND_WAIT_TRIES:-90}"
FRONTEND_WAIT_TRIES="${FRONTEND_WAIT_TRIES:-90}"

# --- output helpers ----------------------------------------------------------

if [[ -t 1 ]]; then
  R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; B=$'\033[34m'; D=$'\033[2m'; N=$'\033[0m'
else
  R=; G=; Y=; B=; D=; N=
fi
step() { printf '\n%s==>%s %s\n' "$B" "$N" "$*"; }
ok()   { printf '  %s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '  %s!%s %s\n' "$Y" "$N" "$*" >&2; }
die()  { printf '\n%sERROR:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }
note() { printf '    %s%s%s\n' "$D" "$*" "$N"; }

# VERBOSE=1 (or -v / --verbose) echoes every external command before it runs.
VERBOSE="${VERBOSE:-0}"

# Every non-fatal failure is recorded here and reprinted at exit, so a problem
# 200 lines up the scrollback cannot be missed, and the exit code reflects it.
FAILURES=()

fail() {
  FAILURES+=("$*")
  printf '  %s✗ FAILED:%s %s\n' "$R" "$N" "$*" >&2
}

vrun() {
  if [[ "$VERBOSE" == 1 ]]; then
    printf '    %s$ %s%s\n' "$D" "$*" "$N"
  fi
  "$@"
}

have() { command -v "$1" >/dev/null 2>&1; }

# Print the tail of a service's log. This is what turns "it timed out" into
# something actionable without the operator having to go hunting.
dump_log() {
  local name="$1" lines="${2:-60}"
  printf '\n  %s--- last %s lines of %s ---%s\n' "$Y" "$lines" "$name" "$N" >&2
  case "$name" in
    llm|embed|falkordb)
      docker logs --tail "$lines" "sosim-$name" 2>&1 | sed 's/^/  | /' >&2 \
        || printf '  | (no container logs; was it ever created?)\n' >&2
      ;;
    *)
      if [[ -f "$LOG_DIR/$name.log" ]]; then
        tail -n "$lines" "$LOG_DIR/$name.log" | sed 's/^/  | /' >&2
      else
        printf '  | (no log file at data/logs/%s.log)\n' "$name" >&2
      fi
      ;;
  esac
  printf '  %s--- end %s ---%s\n\n' "$Y" "$name" "$N" >&2
}

# A container that dies on startup (bad flag, wrong arch, OOM) otherwise only
# surfaces as a health-check timeout minutes later. Catch it immediately.
assert_container_alive() {
  local name="$1" grace="${2:-4}"
  sleep "$grace"
  if container_up "sosim-$name"; then
    ok "$name container is running"
    return 0
  fi
  local state
  state=$(docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} {{.State.Error}}' \
          "sosim-$name" 2>/dev/null || echo 'never created')
  fail "$name container is not running ($state)"
  dump_log "$name" 80
  return 1
}

report_failures() {
  if (( ${#FAILURES[@]} == 0 )); then
    return 0
  fi
  printf '\n%s================ %s FAILURE(S) ================%s\n' \
    "$R" "${#FAILURES[@]}" "$N" >&2
  local i=1
  for f in "${FAILURES[@]}"; do
    printf '  %s%s.%s %s\n' "$R" "$i" "$N" "$f" >&2
    i=$((i + 1))
  done
  printf '\n  Inspect a service:  %s logs [llm|embed|falkordb|zep-shim|backend|frontend]\n' "$SELF" >&2
  printf '  Re-check config:    %s doctor\n\n' "$SELF" >&2
  return 1
}

# Report the exact line and command on an unexpected abort, and always print the
# failure summary on the way out.
trap 'rc=$?; if (( rc != 0 )); then printf "\n%sABORTED%s line %s: %s (exit %s)\n" "$R" "$N" "$LINENO" "$BASH_COMMAND" "$rc" >&2; fi' ERR
trap 'rc=$?; report_failures || rc=1; exit $rc' EXIT

# Run a command with a hard deadline. Every probe in this script goes through
# this: an unbounded `docker run` probe is what made an earlier version of the
# script appear to hang forever with no output.
run_bounded() {
  local seconds="$1"; shift
  if have timeout; then
    timeout "$seconds" "$@"
    return $?
  fi
  # Fallback for hosts without coreutils timeout.
  "$@" & local pid=$!
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if (( waited >= seconds )); then
      kill -TERM "$pid" 2>/dev/null || true
      return 124
    fi
    sleep 1; waited=$((waited + 1))
  done
  wait "$pid"
}

# Ask the embeddings server for one vector and report its width.
probe_embedding_dim() {
  local body
  body=$(printf '{"model":"%s","input":"dimension probe"}' "$EMBED_MODEL_REPO")
  run_bounded 30 curl -fsS "http://127.0.0.1:$EMBED_PORT/v1/embeddings" \
    -H 'Content-Type: application/json' -d "$body" 2>/dev/null \
    | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["data"][0]["embedding"]))' 2>/dev/null
}

# =============================================================================
# runtime profile
# =============================================================================
#
# A runtime profile (scripts/runtimes/<name>.sh) holds every default that
# depends on the hardware: the vLLM image and how it is entered, the weights,
# the memory fractions, context length, batch size, and the GPU and CPU
# architecture the combination was chosen for. Which profile applies, most
# explicit first:
#
#   1. --runtime <name> (what the wrappers, scripts/provision_dgx_spark.sh and
#      scripts/provision_l40s.sh, pass), or
#      SOSIM_RUNTIME in the environment;
#   2. SOSIM_RUNTIME in .env, which `setup` records, so that a later bare
#      `provision_local.sh start` on the same host gets the same profile back;
#   3. the one profile whose CPU architecture AND GPU compute capability both
#      match this host — so a bare command on an L40S whose .env was wiped does
#      not quietly start the DGX image with memory fractions sized for 128GB;
#   4. dgx-spark. A host provisioned before profiles existed has no record, and
#      must go on getting exactly what it always got. (A GB10 is also what
#      step 3 finds on such a host, so the two agree there.)
#
# An explicit choice that contradicts the record is refused for the commands
# that act on it. The .env was set up for the recorded runtime and the weights
# on disk are that runtime's, so obeying would start a model that was never
# downloaded — which fails minutes later inside vLLM, not here.

RUNTIME=""
RUNTIME_SOURCE=""
RUNTIME_ARG=""

runtime_names() {
  local f
  for f in "$RUNTIME_DIR"/*.sh; do
    [[ -f "$f" ]] || continue
    f="${f##*/}"
    printf '%s\n' "${f%.sh}"
  done
}

list_runtimes() {
  step "Runtime profiles (scripts/runtimes/)"
  local name label recorded
  recorded="$(env_file_value SOSIM_RUNTIME)"
  for name in $(runtime_names); do
    label=$(sed -n 's/^RUNTIME_LABEL="\(.*\)"$/\1/p' "$RUNTIME_DIR/$name.sh" | head -1)
    printf '  %-10s %s\n' "$name" "$label"
    if [[ "$name" == "$recorded" ]]; then
      printf '  %-10s run: %s  %s(recorded in .env)%s\n' "" "$(runtime_launcher "$name")" "$G" "$N"
    else
      printf '  %-10s run: %s\n' "" "$(runtime_launcher "$name")"
    fi
  done
  if [[ -z "$recorded" ]]; then
    local detected
    detected="$(detect_runtime)"
    if [[ -n "$detected" ]]; then
      note "no runtime recorded in .env; bare commands use $detected (detected from the GPU)"
    else
      note "no runtime recorded in .env; bare commands use $DEFAULT_RUNTIME (the default)"
    fi
  fi
}

# detect_runtime
# Print the single profile whose RUNTIME_HOST_ARCH and RUNTIME_COMPUTE_CAP both
# match this host, or nothing — when the GPU cannot be queried, when no profile
# fits, and when more than one does (guessing between them would be worse than
# the documented default).
detect_runtime() {
  have nvidia-smi || return 0
  local cap arch name match=""
  cap=$(run_bounded 20 nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | tr -d ' ') || return 0
  [[ "$cap" =~ ^[0-9]+\.[0-9]+$ ]] || return 0
  arch="$(uname -m)"
  for name in $(runtime_names); do
    # shellcheck disable=SC1090
    if ( source "$RUNTIME_DIR/$name.sh" >/dev/null 2>&1 \
         && [[ "${RUNTIME_HOST_ARCH:-}" == "$arch" && " ${RUNTIME_COMPUTE_CAP:-} " == *" $cap "* ]] ); then
      [[ -z "$match" ]] || return 0
      match="$name"
    fi
  done
  printf '%s' "$match"
}

# load_runtime <command>
load_runtime() {
  local cmd="$1" recorded detected="" detect_ran=0
  recorded="$(env_file_value SOSIM_RUNTIME)"

  # Only the commands that act on the hardware probe it. stop, logs and test use
  # nothing from the profile, and a wedged driver is exactly when someone runs
  # `stop` — it must not wait on nvidia-smi first.
  local probe=0
  case "$cmd" in setup|start|all|doctor|runtimes) probe=1 ;; esac

  if [[ -n "$RUNTIME_ARG" ]]; then
    RUNTIME="$RUNTIME_ARG"; RUNTIME_SOURCE="--runtime"
  elif [[ -n "${SOSIM_RUNTIME:-}" ]]; then
    RUNTIME="$SOSIM_RUNTIME"; RUNTIME_SOURCE="SOSIM_RUNTIME in the environment"
  elif [[ -n "$recorded" ]]; then
    RUNTIME="$recorded"; RUNTIME_SOURCE="recorded in .env"
  else
    if (( probe )); then detected="$(detect_runtime)"; detect_ran=1; fi
    if [[ -n "$detected" ]]; then
      RUNTIME="$detected"; RUNTIME_SOURCE="detected from this host's GPU"
    elif (( probe )); then
      RUNTIME="$DEFAULT_RUNTIME"; RUNTIME_SOURCE="default"
    else
      RUNTIME="$DEFAULT_RUNTIME"; RUNTIME_SOURCE="none recorded; not probed for '$cmd'"
    fi
  fi

  local profile="$RUNTIME_DIR/$RUNTIME.sh"
  if [[ ! "$RUNTIME" =~ ^[a-z0-9][a-z0-9-]*$ || ! -f "$profile" ]]; then
    die "unknown runtime '$RUNTIME' ($RUNTIME_SOURCE). Available: $(runtime_names | tr '\n' ' ')"
  fi

  # Refuse, for the commands that create containers or write .env, a runtime
  # this host positively belongs to another profile for: the other wrapper run
  # by mistake, or an .env carried over from the other machine. Positively
  # means detect_runtime named exactly one other profile — a host whose GPU
  # cannot be queried, or matches no profile, is let through as before, so a
  # DGX Spark (which always detects as dgx-spark) can never trip this.
  local hw=""
  case "$cmd" in
    setup|start|all)
      if [[ "${SOSIM_FORCE_RUNTIME:-0}" != 1 ]]; then
        # One probe per invocation: if detection already ran above, its answer
        # (a name, or nothing) is the answer here too.
        if (( detect_ran )); then hw="$detected"; else hw="$(detect_runtime)"; fi
      fi
      ;;
  esac

  if [[ -n "$recorded" && "$recorded" != "$RUNTIME" ]]; then
    local keep
    keep="$(runtime_launcher "$recorded")"
    case "$cmd" in
      setup|start|all)
        printf '\n%sERROR:%s this host was set up for runtime %s (SOSIM_RUNTIME in .env),\n' "$R" "$N" "'$recorded'" >&2
        printf '       but %s asks for %s.\n\n' "$RUNTIME_SOURCE" "'$RUNTIME'" >&2
        if [[ "$hw" == "$RUNTIME" ]]; then
          printf '  This host looks like %s, not %s, so the record is probably wrong -\n' "'$RUNTIME'" "'$recorded'" >&2
          printf '  an .env copied from another machine. Delete the SOSIM_RUNTIME line from\n' >&2
          printf '  .env, review the concurrency keys in it against scripts/runtimes/%s.sh,\n' "$RUNTIME" >&2
          printf '  and run setup again with the runtime you want.\n\n' >&2
        else
          printf '  The .env and the downloaded weights belong to %s. To keep using it:\n' "'$recorded'" >&2
          printf '      %s %s\n\n' "$keep" "$cmd" >&2
          printf '  To really switch this host to %s: stop the stack, delete the\n' "'$RUNTIME'" >&2
          printf '  SOSIM_RUNTIME line from .env, review the concurrency keys in it against\n' >&2
          printf '  scripts/runtimes/%s.sh, then run setup again (it fetches the new weights).\n\n' "$RUNTIME" >&2
        fi
        exit 1
        ;;
      *)
        warn "this host was set up for runtime '$recorded' (.env), but $RUNTIME_SOURCE asks for '$RUNTIME'"
        ;;
    esac
  elif [[ -n "$hw" && "$hw" != "$RUNTIME" ]]; then
    printf '\n%sERROR:%s this host'"'"'s CPU and GPU match runtime %s, but %s selects %s.\n\n' \
      "$R" "$N" "'$hw'" "$RUNTIME_SOURCE" "'$RUNTIME'" >&2
    printf '  For this host:  %s %s\n' "$(runtime_launcher "$hw")" "$cmd" >&2
    [[ "$RUNTIME_SOURCE" == "recorded in .env" ]] && \
      printf '  If the SOSIM_RUNTIME line in .env came from another machine, delete it.\n' >&2
    printf '  To use %s here anyway: SOSIM_FORCE_RUNTIME=1 %s %s\n\n' "'$RUNTIME'" "$(runtime_launcher "$RUNTIME")" "$cmd" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "$profile"

  # A profile that forgets a key would otherwise surface as an unbound-variable
  # abort deep inside start, or — worse — as `docker run` with an empty flag.
  local key missing=()
  for key in RUNTIME_LABEL RUNTIME_HOST_ARCH RUNTIME_COMPUTE_CAP RUNTIME_TORCH_ARCH \
             VLLM_IMAGE LLM_MODEL_REPO GPU_MEM_UTIL EMBED_GPU_MEM_UTIL \
             EMBED_MAX_MODEL_LEN MAX_MODEL_LEN MAX_NUM_SEQS TOOL_CALL_PARSER \
             DROP_PAGE_CACHE GPU_START_ORDER HIDE_GPU_FROM_APP; do
    [[ -n "${!key:-}" ]] || missing+=("$key")
  done
  (( ${#missing[@]} == 0 )) || die "runtime profile $profile does not set: ${missing[*]}"

  VLLM_ENTRYPOINT="${VLLM_ENTRYPOINT:-}"
  LLM_WAIT_TRIES="${LLM_WAIT_TRIES:-300}"
  LLM_MODEL_REVISION="${LLM_MODEL_REVISION:-}"
  ENV_SEED="${ENV_SEED:-}"
  LLM_EXTRA_ARGS="${LLM_EXTRA_ARGS:-}"
  EMBED_EXTRA_ARGS="${EMBED_EXTRA_ARGS:-}"
  GPU_MEM_PRECHECK="${GPU_MEM_PRECHECK:-0}"
  # What the embeddings server really occupies on the card (bge-m3 in fp16 plus
  # its CUDA context), for GPU_MEM_PRECHECK. Its --gpu-memory-utilization is only
  # a startup floor, not a reservation.
  EMBED_EXPECTED_MIB="${EMBED_EXPECTED_MIB:-3072}"

  # Embeddings run on the SAME vLLM image. HuggingFace TEI was the obvious
  # choice but publishes no arm64 image at all — every cpu-* tag is amd64-only
  # (checked against the registry), and the arm64 CUDA tag its docs mention does
  # not resolve. Reusing the vLLM image means one fewer dependency and a
  # guaranteed arm64 build.
  EMBED_IMAGE="${EMBED_IMAGE:-$VLLM_IMAGE}"

  finalize_runtime
}

# finalize_runtime
# Validate, and derive from, the values that .env may also set. Runs once the
# profile is loaded and again after load_env sources .env, so that a value .env
# supplies is both checked and actually used (the GPU-hiding prefix, above all,
# must follow HIDE_GPU_FROM_APP as it finally stands).
finalize_runtime() {
  [[ "$MAX_NUM_SEQS" =~ ^[0-9]+$ ]] || die "MAX_NUM_SEQS='$MAX_NUM_SEQS' is not an integer"
  case "$GPU_START_ORDER" in
    parallel|serial) ;;
    *) die "GPU_START_ORDER='$GPU_START_ORDER' must be 'parallel' or 'serial'" ;;
  esac
  case "${VLLM_ENTRYPOINT:-}" in
    ''|vllm) ;;
    *) die "VLLM_ENTRYPOINT='$VLLM_ENTRYPOINT' must be empty (keep the image's entrypoint) or 'vllm'" ;;
  esac

  # Prefix for SoSim's own Python processes (backend, shim, doctor's preflight):
  # the backend's x86_64 torch has CUDA, and OASIS would take the GPU if it could
  # see it — see start_backend. Empty keeps the command exactly as it was.
  APP_ENV=()
  if [[ "$HIDE_GPU_FROM_APP" == 1 ]]; then
    APP_ENV=(env CUDA_VISIBLE_DEVICES=)
  fi
}

# runtime_launcher <name>: the command that runs a runtime — its wrapper if it
# has one (dgx-spark -> scripts/provision_dgx_spark.sh), else the engine.
runtime_launcher() {
  local wrapper="scripts/provision_${1//-/_}.sh"
  if [[ -x "$ROOT/$wrapper" ]]; then
    printf '%s' "$wrapper"
  else
    printf '%s' "scripts/provision_local.sh --runtime $1"
  fi
}

# check_runtime_hardware
# Compare this host with the hardware the runtime profile was written for. The
# wrong profile does not fail cleanly: memory fractions sized for a 128GB
# unified pool on a 48GB card, an image whose kernels skip this GPU, or weights
# that were never downloaded here all fail minutes later inside vLLM, or serve
# badly, or OOM. A GPU that cannot be queried is reported and let through — the
# containers are the definitive test, and start_llm dumps their logs if they
# cannot get a device.
RUNTIME_CHECKED=0
check_runtime_hardware() {
  (( RUNTIME_CHECKED == 0 )) || return 0
  RUNTIME_CHECKED=1
  step "Runtime: $RUNTIME ($RUNTIME_SOURCE)"
  note "$RUNTIME_LABEL — scripts/runtimes/$RUNTIME.sh"

  local host_arch
  host_arch="$(uname -m)"
  if [[ "$host_arch" == "$RUNTIME_HOST_ARCH" ]]; then
    ok "host arch $host_arch"
  else
    fail "this host is $host_arch, but runtime '$RUNTIME' is for $RUNTIME_HOST_ARCH (see: $SELF runtimes)"
  fi

  if ! have nvidia-smi; then
    note "nvidia-smi not found; the GPU was not checked against the runtime"
    return 0
  fi
  # One bounded query: a wedged driver must not stall start here (see run_bounded).
  local line gpu cap rc=0
  line=$(run_bounded 20 nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1) || rc=$?
  if (( rc == 124 )); then
    # Timed out: the driver is not answering, and every further query would
    # only wait out its own deadline too.
    warn "nvidia-smi did not answer within 20s; the GPU was not checked against the runtime"
    return 0
  fi
  gpu="${line%,*}"
  cap="${line##*,}"; cap="${cap// /}"
  if [[ ! "$cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
    note "GPU ${gpu:-unknown}: compute capability not reported, so not checked"
  elif [[ " $RUNTIME_COMPUTE_CAP " == *" $cap "* ]]; then
    ok "GPU $gpu, compute capability $cap"
  else
    fail "GPU $gpu has compute capability $cap; runtime '$RUNTIME' was built for $RUNTIME_COMPUTE_CAP"
    warn "  Its image and weights may not load here at all. List the alternatives with"
    warn "  '$SELF runtimes'."
  fi

  # An NVIDIA vGPU guest (the L40S-48C is one) is throttled once it is found
  # unlicensed, and a simulation run lasts hours. Only a guest that reports
  # itself as a vGPU is judged: bare-metal GPUs may print a license line of
  # their own (often N/A), and that means nothing here.
  local smi_q vmode license
  smi_q=$(run_bounded 30 nvidia-smi -q 2>/dev/null) || smi_q=""
  vmode=$(sed -n 's/^[[:space:]]*Virtualization Mode[[:space:]]*:[[:space:]]*//p' <<<"$smi_q" | head -1)
  if [[ "$vmode" == VGPU* ]]; then
    license=$(sed -n 's/^[[:space:]]*License Status[[:space:]]*:[[:space:]]*//p' <<<"$smi_q" | head -1)
    if [[ "$license" == Licensed* ]]; then
      ok "vGPU license: $license"
    else
      fail "vGPU license status is '${license:-not reported}'"
      warn "  An unlicensed vGPU runs at full speed only briefly before NVIDIA's driver"
      warn "  degrades it, which a multi-hour simulation will not survive. Check with:"
      warn "      nvidia-smi -q | grep -i -A2 license"
    fi
  fi

  return 0
}

# check_gpu_memory
# Discrete cards only (GPU_MEM_PRECHECK=1). vLLM refuses to start unless the
# device has --gpu-memory-utilization x total FREE at that moment, and both
# servers draw on the same card. Runs after .env is read, so it judges the
# fractions start will really use. Skipped while the LLM runs — its own usage is
# then part of the picture; with only the embeddings server up (the usual state
# while recreating the LLM), what that server uses is already out of "free".
check_gpu_memory() {
  [[ "$GPU_MEM_PRECHECK" == 1 ]] || return 0
  container_up sosim-llm && return 0
  local embed_up=0
  container_up sosim-embed && embed_up=1
  local mem total free
  mem=$(run_bounded 20 nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) || mem=""
  total="${mem%,*}"; total="${total// /}"
  free="${mem##*,}"; free="${free// /}"
  [[ "$total" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] || return 0
  # FREE, not total minus used: a vGPU can hold part of its framebuffer in
  # reserve, and that part is neither used nor available.
  #
  # With the servers started one after the other, the LLM needs
  # GPU_MEM_UTIL x total free once the embeddings server has taken what it
  # really uses (EMBED_EXPECTED_MIB) — below that vLLM refuses to start.
  # Charging the embeddings server its whole --gpu-memory-utilization instead
  # is only a comfort margin: short of it, things still start.
  local need comfort
  if (( embed_up )); then
    need=$(awk -v t="$total" -v a="$GPU_MEM_UTIL" 'BEGIN { printf "%d", t * a }')
    comfort=$need
  else
    need=$(awk -v t="$total" -v a="$GPU_MEM_UTIL" -v e="$EMBED_EXPECTED_MIB" \
             'BEGIN { printf "%d", t * a + e }')
    comfort=$(awk -v t="$total" -v a="$GPU_MEM_UTIL" -v b="$EMBED_GPU_MEM_UTIL" \
             'BEGIN { printf "%d", t * (a + b) }')
  fi
  if (( free < need )); then
    fail "GPU memory: ${free}MiB free of ${total}MiB, but the LLM (GPU_MEM_UTIL=$GPU_MEM_UTIL)$( (( embed_up )) || printf ' and the embeddings server') need ~${need}MiB — vLLM will refuse to start"
    warn "  Whatever holds that memory has to go, or GPU_MEM_UTIL comes down:"
    run_bounded 20 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
      | sed 's/^/      /' >&2 || true
  elif (( free < comfort )); then
    warn "GPU memory: ${free}MiB free of ${total}MiB — enough to start (~${need}MiB), but thin"
    warn "  against GPU_MEM_UTIL + EMBED_GPU_MEM_UTIL (${comfort}MiB). Watch for OOMs under load."
  else
    ok "GPU memory: ${free}MiB free of ${total}MiB; ~${need}MiB is needed"
  fi
  return 0
}

# warn_missing_env_seed
# A runtime's ENV_SEED keys are written only into an .env that setup creates or
# first records the runtime in. An .env made any other way — copied by hand,
# carried over from another host, the offline wipe path — can lack them; say so
# here rather than let the first simulation find out.
warn_missing_env_seed() {
  [[ -n "${ENV_SEED:-}" ]] || return 0
  local seed_line key
  while IFS= read -r seed_line; do
    [[ "$seed_line" == *=* ]] || continue
    key="${seed_line%%=*}"
    if ! grep -qE "^${key}=" "$ROOT/.env" 2>/dev/null; then
      warn "runtime '$RUNTIME' expects $key in .env and it is not set. Add this line:"
      warn "    $seed_line"
    fi
  done <<<"$ENV_SEED"
  return 0
}

# =============================================================================
# preflight
# =============================================================================

preflight() {
  step "Preflight"

  [[ "$(uname -s)" == "Linux" ]] || warn "Not Linux ($(uname -s)); GPU containers will not work."
  ok "arch: $(uname -m)"

  have docker || die "docker not found. Install Docker Engine, then re-run."
  docker info >/dev/null 2>&1 || die "cannot talk to the Docker daemon (add yourself to the 'docker' group?)."
  ok "docker: $(docker --version | cut -d' ' -f3 | tr -d ,)"

  if have nvidia-smi; then
    ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    # Check the runtime is registered by asking the daemon. Do NOT probe by
    # running a container: `docker run --rm --gpus all <image> true` looks
    # harmless but pulls the image and, for any image with an ENTRYPOINT (e.g.
    # falkordb's run.sh), `true` becomes an argument that the entrypoint
    # ignores — so the container starts its real service and never exits.
    # Report the evidence rather than guessing. Not finding the legacy runtime
    # is NOT conclusive: hosts wired up through CDI expose the GPU without it.
    # The definitive test is the LLM container itself, and if `docker run`
    # cannot get a device it says so and start_llm reports the failure.
    local gpu_evidence=()
    docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia \
      && gpu_evidence+=("docker runtime")
    have nvidia-container-runtime && gpu_evidence+=("nvidia-container-runtime")
    have nvidia-ctk && gpu_evidence+=("nvidia-ctk")
    compgen -G "/etc/cdi/*.yaml" >/dev/null 2>&1 && gpu_evidence+=("CDI /etc/cdi")
    compgen -G "/var/run/cdi/*.yaml" >/dev/null 2>&1 && gpu_evidence+=("CDI /var/run/cdi")
    if (( ${#gpu_evidence[@]} > 0 )); then
      ok "GPU container support: ${gpu_evidence[*]}"
    else
      warn "no nvidia container runtime or CDI spec found. If the LLM container"
      warn "reports 'could not select device driver', install nvidia-container-toolkit"
      warn "or set GPU_FLAGS (e.g. GPU_FLAGS='--device nvidia.com/gpu=all')."
    fi
    note "containers will request the GPU with: $GPU_FLAGS"
  else
    warn "nvidia-smi not found. The LLM container needs a working NVIDIA container runtime."
  fi

  # -Pk is POSIX and works on both GNU and BSD df; --output=avail is GNU-only.
  local free_gb
  free_gb=$(df -Pk "$ROOT" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}')
  if [[ -n "$free_gb" && "$free_gb" -lt 120 ]]; then
    warn "only ${free_gb}GB free here. Model weights alone run 25–65GB; 120GB+ recommended."
  else
    ok "disk: ${free_gb}GB free"
  fi

  check_runtime_hardware
}

# =============================================================================
# setup
# =============================================================================

install_system_deps() {
  step "System packages"
  if ! have apt-get; then
    warn "no apt-get; install the equivalents of: build-essential python3-dev git curl fonts-inter"
    return
  fi
  # build-essential + python3-dev are NOT optional: psutil is pinned to 5.9.8,
  # which publishes no linux-aarch64 wheel, so uv builds it from source. It is
  # the only package in the lockfile that does.
  # The UI ships no webfont (an air-gapped box cannot fetch one), so the font
  # stack falls back to whatever is installed. fonts-inter and
  # fonts-jetbrains-mono are what the design tokens ask for.
  local pkgs=(build-essential python3-dev git curl ca-certificates
              fonts-inter fonts-jetbrains-mono)
  local missing=()
  for p in "${pkgs[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then ok "all present"; return; fi
  note "installing: ${missing[*]}"
  note "this needs sudo — enter your password if prompted"
  vrun sudo apt-get update -qq || fail "apt-get update failed"
  vrun sudo apt-get install -y "${missing[@]}" \
    || fail "apt-get install failed for: ${missing[*]}"
  ok "installed"
}

install_uv() {
  step "uv"
  if have uv; then ok "uv $(uv --version | cut -d' ' -f2)"; return 0; fi
  curl -fsSL https://astral.sh/uv/install.sh | sh
  hash -r 2>/dev/null || true
  have uv || die "uv install failed; add \$HOME/.local/bin to PATH and re-run."
  ok "installed uv"
}

install_node() {
  step "Node.js"
  local major=0 current="not installed"
  if have node; then
    current=$(node -v)
    major=$(node -v | sed 's/^v\([0-9]*\).*/\1/')
  fi
  if (( major >= NODE_MAJOR_REQUIRED )); then ok "node $current"; return 0; fi
  warn "node: $current (need >= $NODE_MAJOR_REQUIRED)"

  # The repo's package.json claims node>=18, but the pinned vite@7 and
  # @vitejs/plugin-vue@6 both require ^20.19 || >=22.12. npm ci only warns
  # about the mismatch, so an 18.x box installs cleanly and then misbehaves.
  warn "node ${major:-none} is too old (vite 7 needs >= $NODE_MAJOR_REQUIRED). Installing Node 22 LTS."
  if have apt-get; then
    note "this needs sudo — enter your password if prompted"
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt-get install -y nodejs
  else
    die "install Node 22 LTS manually, then re-run."
  fi
  ok "node $(node -v)"
}

init_submodule() {
  step "Graphiti submodule"
  # Checks out the commit this repo records — which is what we want, and is
  # already the default. Do NOT add --remote: that advances the submodule to the
  # tip of its branch, silently moving off the pinned commit. (An earlier
  # version wrote `--remote=false` trying to be explicit; --remote is a boolean
  # flag with no value, so git rejected the entire command with a usage error.)
  if ! vrun git submodule update --init --recursive; then
    fail "git submodule update failed — check SSH access to the graphiti fork"
    return 1
  fi
  if [[ ! -f "$ROOT/third_party/graphiti/server/graph_service/zep_compat/router.py" ]]; then
    fail "submodule is present but has no zep_compat layer; third_party/graphiti is on the wrong commit"
    note "expected the commit recorded by this repo: $(git ls-tree HEAD third_party/graphiti | awk '{print substr($3,1,12)}')"
    note "got: $(git -C third_party/graphiti rev-parse --short=12 HEAD 2>/dev/null || echo '<none>')"
    return 1
  fi
  ok "graphiti @ $(git -C third_party/graphiti rev-parse --short HEAD)"
}

make_env() {
  step "Environment file"
  mkdir -p "$DATA_DIR" "$RUN_DIR" "$LOG_DIR" "$HF_CACHE"
  local fresh_env=0
  if [[ -f "$ROOT/.env" ]]; then
    ok ".env exists (left untouched)"
  else
    cp "$ROOT/.env.example" "$ROOT/.env"
    fresh_env=1
    ok "created .env from .env.example"
  fi
  # Record the runtime, so a later bare `provision_local.sh start` on this host
  # loads the same profile instead of the dgx-spark fallback. load_runtime has
  # already refused a --runtime that disagrees with an existing record, so this
  # only ever writes a key that is absent.
  local new_record=0
  [[ -n "$(env_file_value SOSIM_RUNTIME)" ]] || new_record=1
  if (( new_record )) && grep -qE '^SOSIM_RUNTIME=' "$ROOT/.env"; then
    # The line is there but empty — blanked rather than deleted — so nothing is
    # recorded and ensure_env_key would leave it that way. The last assignment
    # wins for both `source` and env_file_value, so append.
    printf '\n# Added by provision_local.sh — %s\nSOSIM_RUNTIME=%s\n' \
      "the runtime profile (scripts/runtimes/$RUNTIME.sh) this host was set up for" "$RUNTIME" >>"$ROOT/.env"
    ok "set SOSIM_RUNTIME=$RUNTIME in .env"
  else
    ensure_env_key SOSIM_RUNTIME "$RUNTIME" \
      "the runtime profile (scripts/runtimes/$RUNTIME.sh) this host was set up for"
  fi

  # Keys the runtime profile wants in its .env (ENV_SEED, one KEY=VALUE per line,
  # the value written exactly as given): into a copy made just now, or an .env
  # that is only now being recorded for this runtime (a host switching to it).
  # Only ever ADDED when absent — a value the operator set, even to something
  # else, is theirs and is never rewritten.
  if (( fresh_env || new_record )) && [[ -n "$ENV_SEED" ]]; then
    local seed_line
    while IFS= read -r seed_line; do
      [[ "$seed_line" == *=* ]] || continue
      ensure_env_key "${seed_line%%=*}" "${seed_line#*=}" "seeded by the $RUNTIME runtime profile"
    done <<<"$ENV_SEED"
  fi
  # Keep the served model name in .env consistent with what vLLM will answer to.
  if ! grep -q "^LLM_MODEL_NAME=$LLM_SERVED_NAME$" "$ROOT/.env" 2>/dev/null; then
    note "check LLM_MODEL_NAME in .env matches LLM_SERVED_NAME ($LLM_SERVED_NAME)"
  fi

  # Couple the simulation's concurrency to what this vLLM will actually run at
  # once. Both platforms run together, so the endpoint sees twice the
  # per-platform cap; half of --max-num-seqs keeps the running batch full with
  # no standing queue. Left uncoupled, the client default (30 per platform, 60
  # total) floods a 16-slot server and every request times out waiting in line.
  local want_semaphore=$(( MAX_NUM_SEQS / 2 ))
  (( want_semaphore >= 1 )) || want_semaphore=1
  ensure_env_key SIM_LLM_SEMAPHORE "$want_semaphore" \
    "half of vLLM --max-num-seqs=$MAX_NUM_SEQS, and both platforms run at once"
  ensure_env_key SIM_MODEL_TIMEOUT 300 \
    "a queued request's wait counts against this"
  ensure_env_key SIM_MODEL_MAX_RETRIES 1 \
    "a retry re-enters the same queue, so retries multiply load"
  # The cap that decides whether a reasoning model can answer at all. Without
  # it camel-ai sends no max_tokens, vLLM lets the generation run to
  # --max-model-len minus the prompt, and a model that opens with a <think>
  # block spends the whole timeout before it ever reaches a tool call.
  ensure_env_key SIM_MODEL_MAX_TOKENS 1024 \
    "uncapped, one <think> block runs to --max-model-len and times the request out"

  # Same coupling for graph ingest, where the two knobs MULTIPLY: the shim runs
  # ZEP_COMPAT_BATCH_CONCURRENCY episodes at once and Graphiti fans each one out
  # SEMAPHORE_LIMIT wide, so peak in-flight requests is the product. At the old
  # 4 x 6 = 24 against --max-num-seqs=16 the excess queues and the wait counts
  # against each request's timeout — which is how a 62-episode build spent 50
  # minutes and then died on one openai.APITimeoutError. The embeddings server
  # shares this GPU, so the product gets half the batch, not all of it.
  #
  # SEMAPHORE_LIMIT especially must be written out here: graphiti_core's own
  # default is 20, so an .env that never mentions the key runs 4 x 20 = 80 deep
  # against 16 slots with nothing in any config file to point at.
  local llm_budget=$(( MAX_NUM_SEQS / 2 ))
  (( llm_budget >= 1 )) || llm_budget=1
  local want_batch=2
  (( want_batch <= llm_budget )) || want_batch=1
  local want_fanout=$(( llm_budget / want_batch ))
  (( want_fanout >= 1 )) || want_fanout=1
  ensure_env_key ZEP_COMPAT_BATCH_CONCURRENCY "$want_batch" \
    "x SEMAPHORE_LIMIT must stay under half of vLLM --max-num-seqs=$MAX_NUM_SEQS"
  ensure_env_key SEMAPHORE_LIMIT "$want_fanout" \
    "graphiti's own default is 20, which would flood a $MAX_NUM_SEQS-slot server"

  # Those two calls are advice, not enforcement, and on a FRESH install they do
  # not fire at all: .env was copied from .env.example a few lines up, so both
  # keys are already present at their shipped values and ensure_env_key returns
  # without writing. They only ever act on a pre-existing .env that is missing a
  # key. Meanwhile MAX_NUM_SEQS is overridable and README tells operators to
  # sweep it through 1/4/8/16/32, which silently invalidates the product.
  # So check the invariant itself against what .env EFFECTIVELY says.
  check_ingest_budget \
    "$(env_file_value ZEP_COMPAT_BATCH_CONCURRENCY)" \
    "$(env_file_value SEMAPHORE_LIMIT)" \
    "$llm_budget" || true

  # SIM_PREFLIGHT_TIMEOUT used to ship at 60, sized for a check that asked the
  # endpoint for one token of small talk. That check now sends an agent-shaped
  # request, so a .env of that vintage holds it to a fifth of what the agents
  # themselves get and fails endpoints the run would have tolerated. The key is
  # never rewritten (it may have been tuned), but an operator reading a preflight
  # failure has to know the bar it was held to.
  local preflight_timeout model_timeout
  preflight_timeout="$(env_file_value SIM_PREFLIGHT_TIMEOUT)"
  model_timeout="$(env_file_value SIM_MODEL_TIMEOUT)"
  model_timeout="${model_timeout:-300}"
  if [[ -n "$preflight_timeout" ]] \
     && [[ "$preflight_timeout" =~ ^[0-9]+$ ]] \
     && [[ "${model_timeout%.*}" =~ ^[0-9]+$ ]] \
     && (( preflight_timeout < ${model_timeout%.*} )); then
    warn "SIM_PREFLIGHT_TIMEOUT=$preflight_timeout is BELOW SIM_MODEL_TIMEOUT=$model_timeout."
    warn "  The preflight now sends an agent-shaped request, so this holds it to a"
    warn "  stricter bar than the agents run against and can fail an endpoint the"
    warn "  run would tolerate. Delete the key to track SIM_MODEL_TIMEOUT."
  fi

  # Pin the shim's sqlite state to an ABSOLUTE path under this repo's own
  # gitignored data/. This is non-relocating BY CONSTRUCTION: ensure_env_key
  # writes only when the key is ABSENT, and an absent key already resolved to
  # exactly this file — that is what start_shim's
  # ${ZEP_COMPAT_DB_PATH:-$DATA_DIR/...} default has produced all along. An .env
  # that does carry the key keeps whatever it says, live database and all.
  # Writing it out is what removes the two-defaults-depending-on-.env-vintage
  # ambiguity that had someone inspecting an empty decoy while the real database
  # sat somewhere else.
  ensure_env_key ZEP_COMPAT_DB_PATH "$DATA_DIR/zep_compat.sqlite3" \
    "absolute, so it no longer depends on the shim's working directory"
}

# ensure_env_key <key> <value> [why]
# Add the key with this value when .env does not set it at all. An existing
# value is never overwritten — it may have been tuned deliberately — but a
# disagreement is reported, because a silent mismatch here is what makes a
# simulation die on timeouts.
ensure_env_key() {
  local key="$1" value="$2" why="${3:-}" current
  if grep -qE "^${key}=" "$ROOT/.env" 2>/dev/null; then
    current=$(unquote_env_value "$(grep -E "^${key}=" "$ROOT/.env" | head -1 | cut -d= -f2-)")
    if [[ "$current" != "$(unquote_env_value "$value")" ]]; then
      note "$key=$current in .env (this host suggests $value)"
    fi
    return 0
  fi
  {
    printf '\n# Added by provision_local.sh'
    [[ -n "$why" ]] && printf ' — %s' "$why"
    printf '\n%s=%s\n' "$key" "$value"
  } >> "$ROOT/.env"
  ok "set $key=$value in .env"
}

# unquote_env_value <raw-right-hand-side>
# Turn the text after the '=' into the string `source` would actually produce.
# `cut -d= -f2-` hands back the line verbatim, so SEMAPHORE_LIMIT="4" arrived as
# the three-character string "4" (quotes included) and failed every ^[0-9]+$
# guard downstream — the ingest-budget check silently declined to validate a
# perfectly legal .env. Quoting is normal in a dotenv file, so strip it here
# rather than demanding operators write these keys bare.
unquote_env_value() {
  local v="$1"
  v="${v#"${v%%[![:space:]]*}"}"            # leading space, before the quote test
  case "$v" in
    \"*) v="${v#\"}"; v="${v%%\"*}" ;;      # "…"  — stop at the closing quote
    \'*) v="${v#\'}"; v="${v%%\'*}" ;;      # '…'
    # Bare value: bash treats '#' as starting a comment only after whitespace,
    # so `KEY=abc#def` really is abc#def. Match that, don't truncate at any '#'.
    *)   [[ "$v" =~ ^([^#]*)[[:space:]]\# ]] && v="${BASH_REMATCH[1]}" ;;
  esac
  v="${v#"${v%%[![:space:]]*}"}"            # and again, now inside the quotes
  v="${v%"${v##*[![:space:]]}"}"
  printf '%s' "$v"
}

# env_file_value <key>
# What .env EFFECTIVELY holds for a key — i.e. what `source` ends up with, so
# the LAST assignment wins, not the first, and quotes and inline comments are
# resolved rather than returned as text. Empty when the key is absent.
env_file_value() {
  unquote_env_value "$(grep -E "^$1=" "$ROOT/.env" 2>/dev/null | tail -1 | cut -d= -f2- || true)"
}

# check_ingest_budget <batch> <fanout> <budget>
# The graph-ingest invariant, in one place because make_env and doctor have to
# agree on it. The shim ingests <batch> episodes at once and Graphiti fans each
# one out <fanout> LLM calls wide, so peak in-flight requests is the PRODUCT.
# vLLM admits only --max-num-seqs and queues the rest, and a queued request's
# wait counts against its own timeout: 4 x 6 = 24 against 16 slots is what made
# a 62-episode build run for 50 minutes and die on one openai.APITimeoutError.
# <budget> is half of --max-num-seqs because the embeddings server is a second
# vLLM sharing this GPU.
#
# A broken (or uncheckable) invariant is recorded through fail(), so it is
# reprinted in the exit summary and makes the script exit non-zero, exactly like
# a container that would not start. It does not abort the caller — callers use
# the `|| true` form, the same as start_llm and friends — because the rest of
# setup/doctor is still worth running. Warning alone was not enough: this is the
# misconfiguration that scrolled past twice and cost two 50-minute builds.
check_ingest_budget() {
  local batch="$1" fanout="$2" budget="$3"
  if [[ ! "$batch" =~ ^[0-9]+$ || ! "$fanout" =~ ^[0-9]+$ ]]; then
    fail "cannot check the ingest concurrency invariant — not integers:" \
         "ZEP_COMPAT_BATCH_CONCURRENCY='$batch' SEMAPHORE_LIMIT='$fanout'"
    return 1
  fi
  local peak=$(( batch * fanout ))
  if (( peak > budget )); then
    fail "INGEST OVER-COMMITTED: ZEP_COMPAT_BATCH_CONCURRENCY x SEMAPHORE_LIMIT = $batch x $fanout = $peak concurrent LLM requests, against a budget of $budget"
    warn "  The budget is half of vLLM --max-num-seqs=$MAX_NUM_SEQS, because the embeddings"
    warn "  server is a second vLLM on the same GPU. The excess queues, and the wait counts"
    warn "  against each request's own timeout — this is exactly how a 62-episode"
    warn "  build spent 50 minutes and then died on one openai.APITimeoutError."
    warn "  Lower either key in .env, or raise MAX_NUM_SEQS and re-run '$SELF setup'."
    return 1
  fi
  ok "ingest concurrency $batch x $fanout = $peak (budget $budget, --max-num-seqs=$MAX_NUM_SEQS)"
  return 0
}

install_python_deps() {
  step "Python dependencies"
  note "backend (this compiles psutil from source on aarch64; be patient)"
  if ( cd "$ROOT/backend" && vrun uv sync --frozen ); then
    ok "backend"
  else
    fail "backend dependency install failed (uv sync --frozen in backend/)"
  fi
  note "zep-compat shim"
  if ( cd "$ROOT/third_party/graphiti/server" && vrun uv sync --extra dev ); then
    ok "shim"
  else
    fail "shim dependency install failed (uv sync --extra dev in third_party/graphiti/server/)"
  fi
}

install_node_deps() {
  step "Frontend dependencies"
  note "npm ci for the root workspace, then the frontend (a few minutes)"
  vrun npm ci --no-audit --no-fund || fail "npm ci failed in the repo root"
  vrun npm ci --prefix frontend --no-audit --no-fund || fail "npm ci failed in frontend/"
  [[ -d "$ROOT/frontend/node_modules" ]] || fail "frontend/node_modules is missing after npm ci"
  ok "installed"
}

fetch_models() {
  step "Model weights  (the only stage that needs the internet)"
  export HF_HOME="$HF_CACHE"
  local hf_bin=""
  if have hf; then hf_bin=hf
  elif have huggingface-cli; then hf_bin=huggingface-cli
  else
    note "installing the huggingface_hub CLI"
    uv tool install -q "huggingface_hub[cli]" \
      || pip install -q --user "huggingface_hub[cli]" \
      || fail "could not install the huggingface_hub CLI"
    hash -r 2>/dev/null || true
    if have hf; then
      hf_bin=hf
    elif have huggingface-cli; then
      hf_bin=huggingface-cli
    else
      fail "no hf/huggingface-cli on PATH after install; model weights not downloaded"
      return 1
    fi
  fi
  note "using: $hf_bin"

  # LLM_MODEL_REVISION pins the LLM to one commit of its repo, for checkpoints
  # whose weights have been replaced in place. Unset, `main` is fetched as ever.
  local rev_args=()
  [[ -n "$LLM_MODEL_REVISION" ]] && rev_args=(--revision "$LLM_MODEL_REVISION")
  note "downloading $LLM_MODEL_REPO${LLM_MODEL_REVISION:+ @ $LLM_MODEL_REVISION}"
  HF_HUB_OFFLINE=0 vrun "$hf_bin" download "$LLM_MODEL_REPO" ${rev_args[@]+"${rev_args[@]}"} \
    || fail "failed to download model weights: $LLM_MODEL_REPO"
  note "downloading $EMBED_MODEL_REPO"
  HF_HUB_OFFLINE=0 vrun "$hf_bin" download "$EMBED_MODEL_REPO" \
    || fail "failed to download model weights: $EMBED_MODEL_REPO"

  # A Twitter simulation loads this at runtime; a Reddit-only run never does.
  # Fetch it now or the first Twitter run fails with HF_HUB_OFFLINE=1 set.
  note "downloading Twitter/twhin-bert-base (OASIS Twitter recommender, ~1GB)"
  HF_HUB_OFFLINE=0 vrun "$hf_bin" download Twitter/twhin-bert-base || \
    warn "twhin-bert-base not cached — Twitter simulations will fail offline (Reddit is fine)"

  if grep -q '^GRAPHITI_RERANKER=bge' "$ROOT/.env" 2>/dev/null; then
    note "downloading BAAI/bge-reranker-v2-m3 (GRAPHITI_RERANKER=bge)"
    HF_HUB_OFFLINE=0 "$hf_bin" download BAAI/bge-reranker-v2-m3 || warn "reranker not cached"
  fi
  ok "models cached under $HF_CACHE"
}

pull_images() {
  step "Container images"
  note "the vLLM image is several GB; progress is shown so a long pull is not"
  note "mistaken for a hang"
  for image in $(printf '%s\n' "$FALKORDB_IMAGE" "$VLLM_IMAGE" "$EMBED_IMAGE" | sort -u); do
    printf '\n  --- %s\n' "$image"
    if docker image inspect "$image" >/dev/null 2>&1; then
      ok "already present"
      continue
    fi
    vrun docker pull "$image" || fail "could not pull $image"
  done
  ok "done"
}

# =============================================================================
# service control
# =============================================================================

# shellcheck disable=SC1090
load_env() {
  [[ -f "$ROOT/.env" ]] || die ".env missing. Run: $SELF setup"
  set -a
  source "$ROOT/.env"
  set +a
  # .env writes HF_HOME relative to the repo root (./data/hf-cache), and nothing
  # started from here runs in the repo root: the backend runs in backend/ and
  # each simulation in its own directory, where a relative HF_HOME points at
  # nothing and the offline Twitter recommender cannot be found. The backend and
  # the simulation scripts pin it themselves too (backend/scripts/env_paths.py);
  # this covers everything else this script launches.
  HF_HOME="${HF_HOME:-$HF_CACHE}"
  case "$HF_HOME" in
    /*|\~*) ;;
    *) HF_HOME="$ROOT/${HF_HOME#./}" ;;
  esac
  export HF_HOME
  # .env may set runtime values too (it always could override the tunables);
  # re-check them and rebuild what derives from them.
  finalize_runtime
}

# hf_cached <repo>: does the HF cache hold a snapshot of <repo>? Both vLLM
# containers run with HF_HUB_OFFLINE=1, so a missing one is fatal to them.
hf_cached() {
  local snaps="$HF_CACHE/hub/models--${1//\//--}/snapshots"
  [[ -d "$snaps" && -n "$(ls -A "$snaps" 2>/dev/null)" ]]
}

# venv_has <venv dir> <package>: is <package> installed in that venv? Looked up
# on disk rather than by importing it, which would cost an interpreter start.
venv_has() {
  local d
  for d in "$1"/lib/python*/site-packages/"$2"; do
    [[ -e "$d" ]] && return 0
  done
  return 1
}

# `start` on a host where `setup` never finished (or failed partway) used to
# start every service anyway, and each one then failed on its own: the vLLM
# containers crash-looped on a model the offline cache did not have, the Python
# services died on a missing import, and vite was not found — 25 minutes of
# health-check timeouts to say "run setup". Check what setup produces first,
# and stop in seconds with the list of what is missing.
check_setup_complete() {
  step "Setup artifacts"
  local missing=()
  container_up sosim-llm || hf_cached "$LLM_MODEL_REPO" \
    || missing+=("LLM weights $LLM_MODEL_REPO are not in $HF_CACHE")
  container_up sosim-embed || hf_cached "$EMBED_MODEL_REPO" \
    || missing+=("embedding weights $EMBED_MODEL_REPO are not in $HF_CACHE")
  venv_has "$ROOT/backend/.venv" flask \
    || missing+=("backend dependencies are not installed (backend/.venv has no flask)")
  venv_has "$ROOT/third_party/graphiti/server/.venv" uvicorn \
    || missing+=("shim dependencies are not installed (third_party/graphiti/server/.venv has no uvicorn)")
  [[ -e "$ROOT/frontend/node_modules/.bin/vite" ]] \
    || missing+=("frontend dependencies are not installed (frontend/node_modules/.bin/vite)")
  if (( ${#missing[@]} == 0 )); then
    ok "models cached, dependencies installed"
    return 0
  fi
  local m
  for m in "${missing[@]}"; do fail "$m"; done
  die "setup has not completed on this host; nothing was started. Run: $SELF setup"
}

# container_restarts <container>: its RestartCount, or nothing if unknown.
container_restarts() {
  local n
  n=$(run_bounded 10 docker inspect -f '{{.RestartCount}}' "$1" 2>/dev/null) || return 0
  [[ "$n" =~ ^[0-9]+$ ]] && printf '%s' "$n"
}

# Wait for an HTTP endpoint. On failure the caller gets a dumped log, so a
# timeout always comes with a reason attached.
#
# With a container name, a container that dies while we wait ends the wait at
# once: under --restart unless-stopped a vLLM that fails on startup is revived
# forever, stays "running" to `docker ps`, and would otherwise only surface as a
# 15-minute timeout. A restart since the wait began is that crash loop.
wait_for_http() {
  local url="$1" name="$2" tries="${3:-120}" logname="${4:-}" container="${5:-}"
  local restarts0="" restarts
  [[ -n "$container" ]] && restarts0=$(container_restarts "$container")
  printf '    waiting for %s (%s, up to %ss) ' "$name" "$url" "$((tries * 2))"
  for _ in $(seq "$tries"); do
    # Quiet while polling: -S would print a connection error on every attempt
    # and bury the progress dots. The real error is reported once, below.
    if curl -fsS -o /dev/null --max-time 2 "$url" 2>/dev/null; then
      printf ' %sup%s\n' "$G" "$N"
      return 0
    fi
    if [[ -n "$container" ]]; then
      restarts=$(container_restarts "$container")
      if [[ -n "$restarts0" && -n "$restarts" ]] && (( restarts > restarts0 )); then
        printf ' %sCRASHED%s\n' "$R" "$N"
        fail "$name container $container crashed and is restarting (restart count $restarts0 -> $restarts); see its log below"
        [[ -n "$logname" ]] && dump_log "$logname" 80
        return 1
      fi
    fi
    printf '.'; sleep 2
  done
  printf ' %sTIMEOUT%s\n' "$R" "$N"
  local last_error
  last_error=$(curl -sS -o /dev/null --max-time 2 "$url" 2>&1 || true)
  [[ -n "$last_error" ]] && note "curl says: $last_error"
  fail "$name did not become healthy at $url within $((tries * 2))s"
  [[ -n "$logname" ]] && dump_log "$logname" 80
  return 1
}

container_up() { [[ -n "$(docker ps -q -f "name=^$1$" 2>/dev/null)" ]]; }
container_exists() { [[ -n "$(docker ps -aq -f "name=^$1$" 2>/dev/null)" ]]; }
volume_exists() { docker volume inspect "$1" >/dev/null 2>&1; }

# The containers and the FalkorDB volume were renamed mirofish-* -> sosim-* with
# no data migration. Two consequences, and both are handled here rather than
# left for the operator to discover:
#
#   1. The old containers must go, or the new ones cannot bind their ports. They
#      are removed, not stopped: --restart unless-stopped would revive a stopped
#      one on the next boot.
#   2. The old volume is now unreferenced and every graph in it is unreachable.
#      That is by design, but it is not something to discover from a mysteriously
#      empty graph list, so it is stated plainly and the reclaim command printed.
#
# Idempotent: a machine that never ran the old names sees a single "none found".
retire_legacy_infra() {
  step "Pre-rename containers and volume"

  local stale=()
  local c
  for c in "${LEGACY_CONTAINERS[@]}"; do
    container_exists "$c" && stale+=("$c")
  done

  if (( ${#stale[@]} == 0 )); then
    ok "none found"
  else
    note "these predate the SoSim rename and hold ports $LLM_PORT, $EMBED_PORT and $FALKORDB_PORT"
    for c in "${stale[@]}"; do
      if vrun docker rm -f "$c" >/dev/null 2>&1; then
        ok "removed $c"
      else
        fail "could not remove the pre-rename container $c; sosim-${c#mirofish-} cannot bind its port while it exists"
      fi
    done
  fi

  if volume_exists "$LEGACY_VOLUME"; then
    warn "the pre-rename volume '$LEGACY_VOLUME' is now orphaned."
    warn "FalkorDB starts on '$FALKORDB_VOLUME', which is empty, so EVERY graph built"
    warn "before the rename is gone. That is deliberate: there is no data migration."
    warn "Nothing reads '$LEGACY_VOLUME' again. To reclaim the disk it holds, run:"
    warn "    docker volume rm $LEGACY_VOLUME"
  fi
}

start_falkordb() {
  step "FalkorDB"
  if container_up sosim-falkordb; then ok "already running"; return 0; fi
  docker rm -f sosim-falkordb >/dev/null 2>&1 || true
  if ! vrun docker run -d --name sosim-falkordb --restart unless-stopped \
    -p "127.0.0.1:$FALKORDB_PORT:6379" -p "127.0.0.1:$FALKORDB_UI_PORT:3000" \
    -v "$FALKORDB_VOLUME:/var/lib/falkordb/data" \
    -e BROWSER=1 \
    "$FALKORDB_IMAGE" >/dev/null; then
    fail "could not create the FalkorDB container"
    return 1
  fi
  note "started on 127.0.0.1:$FALKORDB_PORT"
  assert_container_alive falkordb 3
}

# Report when the LIVE container was started with different serving flags than
# this script would use now. Recreating it is not this script's call — it would
# drop a model that took minutes to load, possibly mid-run — but silently
# serving the old flags is how a fixed default fails to reach the one host that
# needed it. The tool-call parser is checked by name because getting it wrong
# does not fail loudly: vLLM logs a parser traceback and still answers 200, so
# every agent request comes back without the tool call it asked for.
warn_if_llm_flags_stale() {
  local live
  live=$(docker inspect --format '{{join .Args " "}}' sosim-llm 2>/dev/null) || return 0
  [[ -n "$live" ]] || return 0

  local live_parser=""
  if [[ "$live" =~ --tool-call-parser[[:space:]]+([^[:space:]]+) ]]; then
    live_parser="${BASH_REMATCH[1]}"
  fi
  if [[ -n "$live_parser" && "$live_parser" != "$TOOL_CALL_PARSER" ]]; then
    warn "the RUNNING container serves --tool-call-parser $live_parser, not $TOOL_CALL_PARSER."
    warn "  A parser that does not match the model still returns 200, with the tool call"
    warn "  left unparsed in the message content — so agents record no actions. To apply"
    warn "  the current setting:  docker rm -f sosim-llm && $SELF start"
  fi

  # Sizing flags only take effect when the container is created, so an override
  # given to `start` while it runs (GPU_MEM_UTIL=0.85 ... start) changes nothing.
  local pair flag want have_val
  for pair in "--gpu-memory-utilization=$GPU_MEM_UTIL" "--max-num-seqs=$MAX_NUM_SEQS" \
              "--max-model-len=$MAX_MODEL_LEN"; do
    flag="${pair%%=*}"; want="${pair#*=}"; have_val=""
    if [[ "$live" =~ $flag[[:space:]]+([^[:space:]]+) ]]; then
      have_val="${BASH_REMATCH[1]}"
    fi
    if [[ -n "$have_val" && "$have_val" != "$want" ]]; then
      warn "the RUNNING container has $flag $have_val, not $want — sizing applies only when"
      warn "  the container is created:  docker rm -f sosim-llm && $SELF start"
    fi
  done

  warn_if_container_stale sosim-llm "$VLLM_IMAGE" "$LLM_MODEL_REPO" "$live"
}

# warn_if_container_stale <container> <image> <model> [live-args]
# A container left over from another runtime profile (or an older default) keeps
# answering on the same port under the same served name, and --restart
# unless-stopped brings it back after every reboot, so nothing downstream can
# tell. Say so where it can still be noticed. Silent when everything matches.
warn_if_container_stale() {
  local name="$1" image="$2" model="$3" live="${4:-}" live_image
  if [[ -z "$live" ]]; then
    live=$(docker inspect --format '{{join .Args " "}}' "$name" 2>/dev/null) || return 0
  fi
  if [[ -n "$live" && " $live " != *" $model "* ]]; then
    warn "the RUNNING $name does not serve $model, which runtime '$RUNTIME' expects."
    warn "  To apply the current setting:  docker rm -f $name && $SELF start"
  fi
  live_image=$(docker inspect --format '{{.Config.Image}}' "$name" 2>/dev/null) || live_image=""
  if [[ -n "$live_image" && "$live_image" != "$image" ]]; then
    warn "the RUNNING $name was created from $live_image, not $image."
    warn "  To apply the current setting:  docker rm -f $name && $SELF start"
  fi
}

# set_vllm_invocation <image>
# Everything a vLLM `docker run` needs after its own flags: the image, and how
# `vllm serve` is reached inside it. The NGC image's entrypoint execs whatever
# command it is given, so it gets `vllm serve ...`. Upstream vllm/vllm-openai
# bakes `vllm serve` into its ENTRYPOINT instead, and handed `vllm serve` again
# it would run `vllm serve vllm serve <model>`; a runtime that uses such an
# image sets VLLM_ENTRYPOINT=vllm, which replaces the entrypoint and leaves
# `serve` as the first word of the command. Either way the container ends up
# running the same `vllm serve <model> <flags>`.
#
# Written to the global VLLM_INVOCATION array: bash 3.2 has no namerefs.
set_vllm_invocation() {
  local image="$1"
  if [[ "$VLLM_ENTRYPOINT" == vllm ]]; then
    VLLM_INVOCATION=(--entrypoint vllm "$image" serve)
  else
    VLLM_INVOCATION=("$image" vllm serve)
  fi
}

start_llm() {
  step "LLM server (vLLM)"
  if container_up sosim-llm; then
    ok "already running"
    warn_if_llm_flags_stale
    return
  fi
  docker rm -f sosim-llm >/dev/null 2>&1 || true

  # Unified memory means the OS page cache eats into the KV cache budget. On a
  # discrete card it does not, and the runtime profile turns this off.
  if [[ "$DROP_PAGE_CACHE" == 1 ]]; then
    note "dropping the page cache to free unified memory (sudo; skipped if refused)"
    sync
    run_bounded 20 sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || \
      note "page cache not dropped (needs passwordless sudo); fine, just less headroom"
  fi

  # Two independent needs, both served by this one endpoint:
  #  - OASIS agents use native OpenAI tool calling  -> the tool-call flags
  #  - Graphiti uses response_format json_schema    -> constrained decoding
  note "model=$LLM_MODEL_REPO  gpu-mem=$GPU_MEM_UTIL  max-len=$MAX_MODEL_LEN"
  note "max-num-seqs=$MAX_NUM_SEQS  tool-call-parser=$TOOL_CALL_PARSER"
  [[ -n "$LLM_EXTRA_ARGS" ]] && note "extra flags ($RUNTIME): $LLM_EXTRA_ARGS"
  set_vllm_invocation "$VLLM_IMAGE"
  # The tokenizer is pinned too: offline, an unpinned tokenizer resolves `main`,
  # which a revision-only download never recorded.
  local rev_args=()
  if [[ -n "$LLM_MODEL_REVISION" ]]; then
    rev_args=(--revision "$LLM_MODEL_REVISION" --tokenizer-revision "$LLM_MODEL_REVISION")
    note "revision=$LLM_MODEL_REVISION"
  fi
  # LLM_EXTRA_ARGS is split on whitespace on purpose, like GPU_FLAGS; the
  # runtime profile documents what that allows.
  # shellcheck disable=SC2086
  if ! vrun docker run -d --name sosim-llm --restart unless-stopped \
    $GPU_FLAGS --ipc=host \
    -p "127.0.0.1:$LLM_PORT:8000" \
    -v "$HF_CACHE:/hf" \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_DISABLE_TELEMETRY=1 \
    "${VLLM_INVOCATION[@]}" "$LLM_MODEL_REPO" \
      --served-model-name "$LLM_SERVED_NAME" \
      --host 0.0.0.0 --port 8000 \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --enable-auto-tool-choice \
      --tool-call-parser "$TOOL_CALL_PARSER" \
      ${rev_args[@]+"${rev_args[@]}"} $LLM_EXTRA_ARGS >/dev/null; then
    fail "could not create the vLLM container"
    return 1
  fi
  note "starting (first load can take several minutes)"
  # 12s: long enough for an immediate flag/arch rejection to show up.
  assert_container_alive llm 12
}

start_embeddings() {
  step "Embeddings server"
  if container_up sosim-embed; then
    ok "already running"
    warn_if_container_stale sosim-embed "$EMBED_IMAGE" "$EMBED_MODEL_REPO"
    return
  fi
  docker rm -f sosim-embed >/dev/null 2>&1 || true
  # vLLM in pooling mode exposes an OpenAI-compatible /v1/embeddings.
  # `--runner pooling` supersedes the older `--task embed`; copying an older
  # recipe with --task embed will fail on a current image.
  note "model=$EMBED_MODEL_REPO  gpu-mem=$EMBED_GPU_MEM_UTIL"
  [[ -n "$EMBED_EXTRA_ARGS" ]] && note "extra flags ($RUNTIME): $EMBED_EXTRA_ARGS"
  set_vllm_invocation "$EMBED_IMAGE"
  # shellcheck disable=SC2086
  if ! vrun docker run -d --name sosim-embed --restart unless-stopped \
    $GPU_FLAGS --ipc=host \
    -p "127.0.0.1:$EMBED_PORT:8000" \
    -v "$HF_CACHE:/hf" \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_DISABLE_TELEMETRY=1 \
    "${VLLM_INVOCATION[@]}" "$EMBED_MODEL_REPO" \
      --runner pooling \
      --host 0.0.0.0 --port 8000 \
      --gpu-memory-utilization "$EMBED_GPU_MEM_UTIL" \
      --max-model-len "$EMBED_MAX_MODEL_LEN" \
      $EMBED_EXTRA_ARGS >/dev/null; then
    fail "could not create the embeddings container"
    return 1
  fi
  note "starting on 127.0.0.1:$EMBED_PORT (model id: $EMBED_MODEL_REPO)"
  assert_container_alive embed 12
}

pidfile() { echo "$RUN_DIR/$1.pid"; }

# start_bg <name> <cwd> <command...>
# The cwd is an argument rather than the caller using ( cd X && start_bg ... ):
# a subshell would discard everything fail() appends to FAILURES, and swallow
# the return code too.
start_bg() {
  local name="$1" cwd="$2"; shift 2
  local pf; pf="$(pidfile "$name")"
  if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then
    ok "$name already running (pid $(cat "$pf"))"
    return
  fi
  mkdir -p "$RUN_DIR" "$LOG_DIR"
  # setsid so the child gets its own process group: SoSim's simulation
  # runner spawns OASIS children with start_new_session=True and cleans them
  # up via killpg on SIGTERM. A hard kill of the parent orphans that whole
  # group, which keeps holding the sqlite DB and burning LLM capacity.
  if [[ "$VERBOSE" == 1 ]]; then
    printf '    %s$ %s%s\n' "$D" "$*" "$N"
  fi
  printf '=== started %s at %s in %s: %s\n' \
    "$name" "$(date -u +%FT%TZ)" "$cwd" "$*" >>"$LOG_DIR/$name.log"
  # setsid gives the child its own process group so do_stop can signal the
  # whole tree (see the killpg note above). Not every host ships it (macOS does
  # not), so branch rather than expanding a possibly-empty array — that trips
  # `set -u` on bash 3.2.
  if have setsid; then
    ( cd "$cwd" && exec setsid "$@" ) >>"$LOG_DIR/$name.log" 2>&1 &
  else
    warn "setsid not found; $name will not get its own process group"
    ( cd "$cwd" && exec "$@" ) >>"$LOG_DIR/$name.log" 2>&1 &
  fi
  local pid=$!
  echo "$pid" >"$pf"
  # A process that exits immediately (bad interpreter, import error, port in
  # use) would otherwise only show up as a health-check timeout.
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pf"
    fail "$name exited immediately after launch"
    dump_log "$name" 80
    return 1
  fi
  ok "$name started (pid $pid), logging to data/logs/$name.log"
}

start_shim() {
  step "Zep-compatible shim (Graphiti-backed)"
  local shim_dir="$ROOT/third_party/graphiti/server"
  local venv="$shim_dir/.venv/bin/python"
  if [[ ! -x "$venv" ]]; then
    fail "shim venv missing at $venv — run: $SELF setup"
    return 1
  fi

  # The shim resolves ZEP_COMPAT_DB_PATH against ITS OWN cwd, which start_bg
  # sets to $shim_dir — not the repo root, not $DATA_DIR. Which file is live
  # therefore depends on this host's .env vintage, and BOTH outcomes are real:
  # an .env carrying the ./data/zep_compat.sqlite3 this repo used to ship lands
  # under $shim_dir, while an .env that never mentions the key falls through to
  # the ${:-} default below and lands in $DATA_DIR. Either way the note printed
  # a relative path, so someone chasing live batch state ran sqlite3 against
  # $ROOT/data/zep_compat.sqlite3, where sqlite3 SILENTLY CREATED an empty
  # database, and nearly mutated that instead of the real one.
  #
  # So: resolve a relative value against the shim's cwd — the same place the
  # shim itself would land on, so an existing install's database does NOT move —
  # and export it absolute, so the note below names the file actually in use.
  local db_path="${ZEP_COMPAT_DB_PATH:-$DATA_DIR/zep_compat.sqlite3}"
  case "$db_path" in
    /*) ;;
    *)  db_path="$shim_dir/${db_path#./}" ;;
  esac
  export ZEP_COMPAT_DB_PATH="$db_path"
  note "sqlite state: $ZEP_COMPAT_DB_PATH"

  # Both locations can hold a REAL database — see the vintage note above — so
  # report the second one without judging it. Do NOT call it dead and do NOT
  # suggest deleting it: on a host whose .env predates ZEP_COMPAT_DB_PATH this
  # is the file the shim has been writing all along, and the empty decoy sqlite3
  # leaves behind on a mistyped path looks identical from the outside. Print how
  # to tell them apart instead and let the operator decide.
  local other_db="$DATA_DIR/zep_compat.sqlite3"
  if [[ "$ZEP_COMPAT_DB_PATH" != "$other_db" && -f "$other_db" ]]; then
    warn "a second database file exists at $other_db"
    warn "This run uses the path printed above and leaves that one untouched, but"
    warn "either can be the live one: an .env without ZEP_COMPAT_DB_PATH resolved"
    warn "there. Compare them before touching either — size, mtime, and batches:"
    warn "    ls -l '$ZEP_COMPAT_DB_PATH' '$other_db'"
    warn "    sqlite3 '$other_db' 'select count(*) from batches;'"
    warn "'no such table: batches' (or zero rows, 0 bytes) means that file is the"
    warn "empty decoy. If it is the one with your history, point ZEP_COMPAT_DB_PATH"
    warn "in .env at it — as an absolute path — rather than deleting anything."
  fi

  # Same GPU hiding as the backend (start_backend): the shim only touches torch
  # for GRAPHITI_RERANKER=bge, and that reranker must not land on the vLLM card.
  start_bg zep-shim "$shim_dir" ${APP_ENV[@]+"${APP_ENV[@]}"} \
    "$venv" -m uvicorn graph_service.zep_compat.app:app \
    --host 127.0.0.1 --port "$SHIM_PORT"
}

start_backend() {
  step "SoSim backend"
  # Must be the venv interpreter: simulation_runner spawns children with
  # sys.executable, so a system python here means every simulation child dies
  # on `import oasis`.
  local venv="$ROOT/backend/.venv/bin/python"
  if [[ ! -x "$venv" ]]; then
    fail "backend venv missing at $venv — run: $SELF setup"
    return 1
  fi
  # On x86_64 the backend's PyPI torch ships CUDA, and OASIS puts its Twitter
  # recommender on the GPU whenever torch can see one (oasis/social_platform/
  # recsys.py). The vLLM servers have already claimed nearly all of a discrete
  # card, so each simulation subprocess would either die on a CUDA OOM or take
  # memory the KV cache was sized to have. Hiding the GPU gives these processes
  # what they get on the DGX Spark anyway: CPU torch. Simulation children
  # inherit the environment, so this covers them too.
  if [[ "$HIDE_GPU_FROM_APP" == 1 ]]; then
    note "GPU hidden from the backend and its simulations (CUDA_VISIBLE_DEVICES='')"
  fi
  start_bg backend "$ROOT/backend" ${APP_ENV[@]+"${APP_ENV[@]}"} "$venv" run.py
}

start_frontend() {
  step "Frontend"
  start_bg frontend "$ROOT/frontend" npm run dev -- --port "$FRONTEND_PORT"
}

do_start() {
  load_env
  # FAILURES is global and `all` runs make_env in this same process, so a config
  # advisory recorded there (the ingest-budget check) is already in the array
  # before a single service has been touched. It still belongs in the exit
  # summary and the exit code — it is a real misconfiguration — but it is NOT
  # evidence that a service failed to come up, and gating on the raw count made a
  # healthy stack report "the stack is NOT fully up". Gate on what THIS function
  # adds instead.
  local failures_at_start=${#FAILURES[@]}
  # Must run before anything binds a port: the pre-rename containers are still
  # holding 8000, 8081 and 6379 on any machine that ran this stack before.
  retire_legacy_infra || true
  # Each step records its own failures and we deliberately continue, so one
  # broken service still yields a full picture instead of stopping at the first
  # problem. report_failures() (EXIT trap) sets the exit code.
  # What check_runtime_hardware reports (a license, a mismatch, a tight card) is
  # a warning about this host, not a service that failed to come up: count it
  # with the "reported before startup" problems, still in the exit code.
  local hw_before=${#FAILURES[@]}
  check_runtime_hardware || true
  check_gpu_memory || true
  failures_at_start=$(( failures_at_start + ${#FAILURES[@]} - hw_before ))
  warn_missing_env_seed
  check_setup_complete
  start_falkordb        || true
  start_embeddings      || true
  if [[ "$GPU_START_ORDER" == serial ]]; then
    # One vLLM at a time. Each sizes its KV cache from what the device reports
    # while it profiles, and on a discrete card with little slack a second
    # server loading its weights at that moment is counted against the first
    # (or trips vLLM's "Error in memory profiling" check). The embeddings server
    # is small and quick, so it goes first and the LLM measures a settled GPU.
    wait_for_http "http://127.0.0.1:$EMBED_PORT/v1/models" "embeddings" \
      "$EMBED_WAIT_TRIES" embed sosim-embed || true
    start_llm           || true
  else
    start_llm           || true
    wait_for_http "http://127.0.0.1:$EMBED_PORT/v1/models" "embeddings" \
      "$EMBED_WAIT_TRIES" embed sosim-embed || true
  fi
  wait_for_http "http://127.0.0.1:$LLM_PORT/v1/models" "LLM" \
    "$LLM_WAIT_TRIES" llm sosim-llm || true

  start_shim || true
  wait_for_http "http://127.0.0.1:$SHIM_PORT/healthcheck" "shim" \
    "$SHIM_WAIT_TRIES" zep-shim || true

  start_backend || true
  wait_for_http "http://127.0.0.1:$BACKEND_PORT/health" "backend" \
    "$BACKEND_WAIT_TRIES" backend || true

  start_frontend || true
  wait_for_http "http://127.0.0.1:$FRONTEND_PORT" "frontend" \
    "$FRONTEND_WAIT_TRIES" frontend || true

  if (( ${#FAILURES[@]} == failures_at_start )); then
    summary
    # Anything recorded BEFORE do_start (on the `all` path that means make_env,
    # whose only entries are config advisories) is still printed by
    # report_failures on the way out, and still makes the script exit non-zero.
    # Say so here, or the summary and the exit code look like they disagree.
    # This counts every such entry, not only advisories - it is a "something
    # earlier objected" note, so keep the wording non-specific about what.
    if (( failures_at_start > 0 )); then
      warn "every service is up, but $failures_at_start problem(s) were reported before startup"
    fi
  else
    step "Started with errors"
    warn "the stack is NOT fully up; see the failure list below"
    do_status
  fi
}

do_stop() {
  step "Stopping"
  for name in frontend backend zep-shim; do
    local pf; pf="$(pidfile "$name")"
    if [[ -f "$pf" ]]; then
      local pid; pid="$(cat "$pf")"
      # Negative PID = whole process group, so OASIS children go too.
      kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
      kill -KILL "-$pid" 2>/dev/null || true
      rm -f "$pf"
      ok "$name stopped"
    fi
  done
  for c in sosim-llm sosim-embed sosim-falkordb; do
    docker stop "$c" >/dev/null 2>&1 && ok "$c stopped" || true
  done
}

do_status() {
  step "Status  (runtime: $RUNTIME, $RUNTIME_SOURCE)"
  printf '  %-12s %-9s %s\n' SERVICE STATE ENDPOINT
  for row in "falkordb:sosim-falkordb:127.0.0.1:$FALKORDB_PORT" \
             "embeddings:sosim-embed:http://127.0.0.1:$EMBED_PORT/v1/models" \
             "llm:sosim-llm:http://127.0.0.1:$LLM_PORT/v1/models"; do
    IFS=: read -r label container rest <<<"$row"
    local endpoint="${row#"$label:$container:"}"
    if container_up "$container"; then
      printf '  %-12s %s%-9s%s %s\n' "$label" "$G" running "$N" "$endpoint"
    else
      printf '  %-12s %s%-9s%s %s\n' "$label" "$R" down "$N" "$endpoint"
    fi
  done
  for row in "zep-shim:http://127.0.0.1:$SHIM_PORT/healthcheck" \
             "backend:http://127.0.0.1:$BACKEND_PORT/health" \
             "frontend:http://127.0.0.1:$FRONTEND_PORT"; do
    IFS=: read -r label _ <<<"$row"
    local endpoint="${row#"$label:"}"
    local pf; pf="$(pidfile "$label")"
    if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then
      printf '  %-12s %s%-9s%s %s\n' "$label" "$G" running "$N" "$endpoint"
    else
      printf '  %-12s %s%-9s%s %s\n' "$label" "$R" down "$N" "$endpoint"
    fi
  done
}

do_logs() {
  local svc="${1:-}"
  case "$svc" in
    llm)        docker logs -f --tail 200 sosim-llm ;;
    embed*)     docker logs -f --tail 200 sosim-embed ;;
    emb)        docker logs -f --tail 200 sosim-embed ;;
    falkor*)    docker logs -f --tail 200 sosim-falkordb ;;
    ''|all)     tail -n 100 -f "$LOG_DIR"/*.log ;;
    *)          tail -n 200 -f "$LOG_DIR/$svc.log" ;;
  esac
}

do_test() {
  step "Test suites (no GPU, no network, no database)"
  # The runtime profiles first: it stubs Docker and the GPU out entirely, needs
  # no venv, and is what pins the DGX Spark containers against drift.
  printf '\n  --- runtime profiles\n'
  if vrun bash "$ROOT/scripts/tests/test_runtimes.sh"; then
    ok "runtime profile checks passed"
  else
    fail "runtime profile checks failed (see the output above)"
  fi
  local suite
  for suite in "backend:$ROOT/backend" "shim:$ROOT/third_party/graphiti/server"; do
    local label="${suite%%:*}" dir="${suite#*:}"
    if [[ ! -x "$dir/.venv/bin/python" ]]; then
      fail "$label venv missing at $dir/.venv — run: $SELF setup"
      continue
    fi
    printf '\n  --- %s\n' "$label"
    if ( cd "$dir" && vrun .venv/bin/python -m pytest tests/ -q ); then
      ok "$label suite passed"
    else
      fail "$label test suite failed (see the pytest output above)"
    fi
  done
}

check_gpu_arch() {
  step "GPU architecture support in $VLLM_IMAGE"
  # The torch inside the image has to carry kernels this GPU can run
  # (RUNTIME_TORCH_ARCH, from the runtime profile — sm_121 for GB10). A torch that
  # only compiles through sm_120 fails on GB10 at runtime with errors that do not
  # mention the architecture at all.
  local arches
  if ! docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
    warn "$VLLM_IMAGE not pulled yet; run '$SELF setup' first to check this"
    return 0
  fi
  # Same entrypoint handling as the servers: an image whose ENTRYPOINT is
  # `vllm serve` would treat `python -c ...` as a model name.
  local probe=("$VLLM_IMAGE" python)
  [[ "$VLLM_ENTRYPOINT" == vllm ]] && probe=(--entrypoint python3 "$VLLM_IMAGE")
  # Bounded, and only against an image already on disk — see run_bounded.
  if arches=$(run_bounded 180 docker run --rm $GPU_FLAGS "${probe[@]}" \
      -c 'import torch; print(" ".join(torch.cuda.get_arch_list()))' 2>/dev/null); then
    note "torch arch list: $arches"
    local want found=""
    for want in $RUNTIME_TORCH_ARCH; do
      [[ "$arches" == *"$want"* ]] && { found="$want"; break; }
    done
    if [[ -n "$found" ]]; then
      ok "$found present"
    else
      warn "$RUNTIME_TORCH_ARCH NOT in the arch list — this image may fail on this GPU. Try another tag."
    fi
  else
    warn "could not run the image with GPU access ($GPU_FLAGS)."
    warn "This doubles as the GPU passthrough test. If the LLM container also"
    warn "fails, try: GPU_FLAGS='--device nvidia.com/gpu=all' $SELF start"
  fi
}

# Read-only counterpart to retire_legacy_infra: doctor reports, start removes.
report_legacy_infra() {
  step "Pre-rename leftovers"
  local found=0 c
  for c in "${LEGACY_CONTAINERS[@]}"; do
    if container_exists "$c"; then
      warn "$c still exists and blocks the matching sosim-* container; '$SELF start' removes it"
      found=1
    fi
  done
  if volume_exists "$LEGACY_VOLUME"; then
    warn "volume '$LEGACY_VOLUME' is orphaned; its graphs are gone by design."
    warn "Reclaim the disk with:  docker volume rm $LEGACY_VOLUME"
    found=1
  fi
  (( found == 0 )) && ok "nothing left from the pre-rename names"
  return 0
}

do_doctor() {
  step "Doctor"
  # Judge everything against the values start uses: .env is read after the
  # profile there, so it is here too.
  if [[ -f "$ROOT/.env" ]]; then load_env; fi
  preflight
  check_gpu_memory
  report_legacy_infra
  check_gpu_arch

  # Docker names architectures differently from uname.
  local docker_arch
  case "$(uname -m)" in
    aarch64|arm64) docker_arch=arm64 ;;
    x86_64|amd64)  docker_arch=amd64 ;;
    *)             docker_arch="$(uname -m)" ;;
  esac
  step "$docker_arch manifests"
  for image in $(printf '%s\n' "$FALKORDB_IMAGE" "$VLLM_IMAGE" "$EMBED_IMAGE" | sort -u); do
    local manifest
    manifest=$(run_bounded 60 docker manifest inspect "$image" 2>&1) || manifest=""
    if [[ -z "$manifest" || "$manifest" == *"manifest unknown"* || "$manifest" == *"no such manifest"* ]]; then
      warn "$image: manifest not found — that tag probably does not exist"
    elif grep -q "\"architecture\": *\"$docker_arch\"" <<<"$manifest"; then
      ok "$image has an $docker_arch build"
    elif ! grep -q '"manifests"' <<<"$manifest"; then
      ok "$image is a single-arch image (assuming it matches this host)"
    else
      warn "$image has NO $docker_arch build. It will not run on this host."
      warn "  architectures offered: $(grep -o '"architecture": *"[a-z0-9]*"' <<<"$manifest" \
            | grep -v unknown | sed 's/.*"\([a-z0-9]*\)"$/\1/' | sort -u | tr '\n' ' ')"
    fi
  done

  step "NVFP4 kernels"
  # Only native on Blackwell. Elsewhere vLLM falls back to Marlin (FP4 weights,
  # bf16 activations) — fine — or, failing that, to EMULATION, which serves the
  # same model orders of magnitude slower: every agent request then times out
  # and the run records nothing. The choice is logged once, at load.
  # Judge the container that is actually serving, not only the configuration:
  # after a model switch that never recreated it, the two differ.
  local live_llm=""
  if container_exists sosim-llm; then
    live_llm=$(docker inspect --format '{{join .Args " "}}' sosim-llm 2>/dev/null) || live_llm=""
    warn_if_llm_flags_stale
  fi
  if [[ "$LLM_MODEL_REPO" != *NVFP4* && "$live_llm" != *NVFP4* ]]; then
    note "$LLM_MODEL_REPO is not an NVFP4 checkpoint; nothing to check"
  elif ! container_exists sosim-llm; then
    note "LLM container not created yet; start the stack and re-run doctor"
  else
    local moe_backend
    moe_backend=$(run_bounded 60 docker logs sosim-llm 2>&1 \
      | grep -m1 -oE "Using '?[A-Za-z0-9_]+'? (NvFp4|NVFP4) MoE backend" || true)
    if [[ -z "$moe_backend" ]]; then
      note "no NVFP4 MoE backend line in the LLM log (still loading, or this vLLM does not log it)"
    elif [[ "$moe_backend" == *EMULATION* ]]; then
      fail "vLLM is EMULATING NVFP4: '$moe_backend'. Expect every agent request to time out."
      warn "  On a non-Blackwell GPU it should pick MARLIN. Check the image and the"
      warn "  vLLM log (docker logs sosim-llm | grep -i fp4), or serve an AWQ build instead."
    else
      ok "$moe_backend"
    fi
  fi

  step "Embedding dimension"
  # EMBEDDING_DIM is a one-way door: the vector index is created with it, so a
  # mismatch means re-embedding every graph later. And Graphiti TRUNCATES longer
  # vectors silently rather than erroring. Check it live while that is cheap.
  local dim
  dim=$(probe_embedding_dim) || dim=""
  if [[ -z "$dim" ]]; then
    note "embeddings server not reachable; start the stack and re-run doctor"
  elif [[ "$dim" == "${EMBEDDING_DIM:-1024}" ]]; then
    ok "server returns $dim dims, matching EMBEDDING_DIM"
  else
    warn "server returns $dim dims but EMBEDDING_DIM=${EMBEDDING_DIM:-1024}."
    warn "Fix this BEFORE ingesting anything — Graphiti truncates silently."
  fi

  step "Simulation LLM request"
  # The embedding probe above proves that server answers; this proves the CHAT
  # server can do the thing a simulation actually needs, which is to return a
  # TOOL CALL over an agent-sized prompt inside the run's own timeout. Those
  # are different questions: an endpoint that answers "ping" instantly can
  # still fail every agent request, and when it does the round loop absorbs the
  # failures and the run reports full rounds against an empty action log. Ask
  # the real question here, where it costs one request.
  local sim_py="$ROOT/backend/.venv/bin/python"
  if [[ ! -x "$sim_py" ]]; then
    note "backend venv missing at $sim_py — run: $SELF setup"
  else
    load_env
    local llm_budget_s="${SIM_MODEL_TIMEOUT:-300}"
    # Give the wrapper a little more than the request itself is allowed, so a
    # bounded-out probe means the request hung, not that we cut it short.
    local probe_budget_s=$(( ${llm_budget_s%.*} + 30 ))
    if run_bounded "$probe_budget_s" ${APP_ENV[@]+"${APP_ENV[@]}"} "$sim_py" \
         "$ROOT/backend/scripts/run_parallel_simulation.py" \
         --preflight-only --twitter-only; then
      ok "the endpoint returns a tool call over an agent-sized prompt"
    else
      fail "the chat endpoint cannot serve a simulation (see the line above)"
    fi
  fi

  step "Config sanity"
  load_env
  warn_missing_env_seed
  [[ -n "${ZEP_BASE_URL:-}" ]] && ok "ZEP_BASE_URL=$ZEP_BASE_URL" \
    || warn "ZEP_BASE_URL unset — SoSim would talk to Zep Cloud."
  [[ -z "${ZEP_API_URL:-}" ]] && ok "ZEP_API_URL unset (required)" \
    || die "ZEP_API_URL is set; the app refuses to boot. Use ZEP_BASE_URL."
  [[ "${GRAPHITI_TELEMETRY_ENABLED:-}" == "false" ]] && ok "Graphiti telemetry off" \
    || warn "GRAPHITI_TELEMETRY_ENABLED is not false — Graphiti will call out to PostHog."
  [[ "${FLASK_DEBUG:-false}" == "false" ]] && ok "FLASK_DEBUG off" \
    || warn "FLASK_DEBUG is on; the reloader fork breaks simulation process tracking."
  if grep -qE '^LLM_BOOST_[A-Z_]+=\s*$' "$ROOT/.env"; then
    warn "LLM_BOOST_* keys are present but blank. They must be absent entirely."
  fi

  # The graph-ingest concurrency invariant. doctor is the only place it gets
  # re-checked after setup: make_env's ensure_env_key calls never overwrite, and
  # README tells operators to sweep MAX_NUM_SEQS through 1/4/8/16/32, so the host
  # that actually hit the incident would otherwise still get no signal at all.
  local budget=$(( MAX_NUM_SEQS / 2 ))
  (( budget >= 1 )) || budget=1
  # An ABSENT key is not the value .env.example documents — each side falls back
  # to its own code default: 4 in the shim (zep_compat/runtime.py) and 20 in
  # graphiti_core (helpers.py). Check what will really be used, not what is
  # written down.
  local batch="${ZEP_COMPAT_BATCH_CONCURRENCY:-4}"
  local fanout="${SEMAPHORE_LIMIT:-}"
  if [[ -z "$fanout" ]]; then
    warn "SEMAPHORE_LIMIT is not set in .env. graphiti_core does NOT fall back to"
    warn "the value .env.example documents — its code default is 20, so that is"
    warn "what graph ingest will actually run at:"
    fanout=20
  fi
  check_ingest_budget "$batch" "$fanout" "$budget" || true

  # Same class of bug as SEMAPHORE_LIMIT above, one layer down: an ABSENT
  # GRAPHITI_LLM_MAX_TOKENS does not mean "whatever .env.example documents", it
  # means the shim's own code default of 16384 (zep_compat/runtime.py) — which is
  # precisely the configuration of the build that ran ~50 minutes and died on one
  # openai.APITimeoutError. Nothing in any config file would point at it.
  local max_tokens="${GRAPHITI_LLM_MAX_TOKENS:-}"
  if [[ -z "$max_tokens" ]]; then
    warn "GRAPHITI_LLM_MAX_TOKENS is not set in .env. The shim does NOT fall back to"
    warn "the value .env.example documents — runtime.py's code default is 16384, so"
    warn "every extraction call gets the same budget as the run that spent ~50 minutes"
    warn "and then died on openai.APITimeoutError. Set it explicitly in .env."
  elif [[ ! "$max_tokens" =~ ^[0-9]+$ ]]; then
    warn "GRAPHITI_LLM_MAX_TOKENS='$max_tokens' is not an integer; the shim casts it"
    warn "with int() at startup and will raise ValueError before serving anything."
  else
    ok "GRAPHITI_LLM_MAX_TOKENS=$max_tokens"
    # Deliberately no "good value" here, because there isn't one: 16384 timed out
    # and 4096 truncated mid-JSON, both in production.
    note "no cap is a fix — 16384 timed out, 4096 truncated (JSONDecodeError at"
    note "char 4148: ~1 char/token, the leading hypothesis being a runaway list of"
    note "integers). A cap hit now raises a named TruncatedResponseError rather"
    note "than a JSONDecodeError, and is not retried; this bounds the damage only."
  fi

  # Read by the shim (zep_compat/runtime.py) and handed straight to AsyncOpenAI
  # alongside max_retries=0, so it is the only thing bounding a hung extraction.
  local llm_timeout="${GRAPHITI_LLM_REQUEST_TIMEOUT:-}"
  if [[ -z "$llm_timeout" ]]; then
    note "GRAPHITI_LLM_REQUEST_TIMEOUT unset; runtime.py's own default of 180s applies"
  elif [[ ! "$llm_timeout" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    warn "GRAPHITI_LLM_REQUEST_TIMEOUT='$llm_timeout' is not a number; the shim casts it"
    warn "with float() when it first builds an LLM client, so this fails on the first"
    warn "graph build (or preflight), not at startup - the shim starts up fine."
  else
    ok "GRAPHITI_LLM_REQUEST_TIMEOUT=${llm_timeout}s (with max_retries=0)"
  fi

  # Not validated anywhere downstream: graphiti compares this to 'json_object'
  # exactly and treats everything else — including a typo — as json_schema, so a
  # misspelled fallback would look like it was applied and change nothing.
  case "${GRAPHITI_STRUCTURED_OUTPUT_MODE:-json_schema}" in
    json_schema)
      ok "GRAPHITI_STRUCTURED_OUTPUT_MODE=json_schema (schema sent as response_format)"
      note "vLLM is EXPECTED to enforce it with constrained decoding, but that is"
      note "UNCONFIRMED here: nothing pins the structured-output backend, and xgrammar"
      note "has historically treated array maxItems as unsupported and ignored it at"
      note "decode time. To confirm which backend this server chose (not '$SELF logs',"
      note "which follows the stream and would never return):"
      note "  docker logs sosim-llm | grep -iE 'guided|structured.?output|xgrammar|outlines'"
      note "If the runaway-array truncation returns, GRAPHITI_STRUCTURED_OUTPUT_MODE="
      note "json_object is the zero-code fallback."
      ;;
    json_object)
      warn "GRAPHITI_STRUCTURED_OUTPUT_MODE=json_object: the schema is injected into"
      warn "the prompt and NOT enforced. That is the intended zero-code fallback for the"
      warn "runaway-array truncation — a switch to shape/validation errors means"
      warn "generation now terminates — but expect messier extraction. The schemas do"
      warn "carry their own ceilings now, so go back to json_schema once you have"
      warn "confirmed this server actually honours them at decode time."
      ;;
    *)
      warn "GRAPHITI_STRUCTURED_OUTPUT_MODE='$GRAPHITI_STRUCTURED_OUTPUT_MODE' is not one of"
      warn "json_schema / json_object. graphiti only tests for 'json_object', so this"
      warn "silently behaves as json_schema — nothing rejects it and nothing logs it."
      ;;
  esac

  [[ -d "$HF_CACHE/hub" ]] && ok "HF cache present at $HF_CACHE" \
    || warn "no HF cache yet; run '$SELF setup' before going offline."

  # What a Twitter simulation actually loads, looked up where it will look:
  # HF_HOME as .env sets it, resolved against the repo root (see load_env), which
  # need not be where setup downloaded to (HF_CACHE_DIR).
  if [[ -d "$HF_HOME/hub/models--Twitter--twhin-bert-base" ]]; then
    ok "Twitter recommender model cached under HF_HOME=$HF_HOME"
  else
    warn "Twitter/twhin-bert-base is not under HF_HOME=$HF_HOME, where simulations look."
    warn "  Twitter and parallel simulations will fail offline (Reddit-only runs are fine)."
    warn "  Run '$SELF setup', or point HF_HOME in .env at the cache setup filled ($HF_CACHE)."
  fi
}

summary() {
  local ip; ip=$(hostname -I 2>/dev/null | awk '{print $1}'); ip="${ip:-<this-host>}"
  cat <<EOF

$(printf '%s' "$G")SoSim is up.$(printf '%s' "$N")

  Open:  http://$ip:$FRONTEND_PORT

  Runtime: $RUNTIME — $RUNTIME_LABEL

  Expose ONLY this port on the network:

    $FRONTEND_PORT/tcp   frontend (Vite). It proxies /api to the backend, so this
                         single port serves the whole application.

  Everything else is bound to 127.0.0.1 and must NOT be exposed:

    $BACKEND_PORT   backend API        $SHIM_PORT   Zep-compatible shim
    $LLM_PORT   vLLM (OpenAI API)  $EMBED_PORT   embeddings (vLLM)
    $FALKORDB_PORT   FalkorDB           $FALKORDB_UI_PORT   FalkorDB browser UI

  To reach the UI by hostname rather than IP, set VITE_ALLOWED_HOSTS in .env.

  $SELF status | logs [llm|backend|zep-shim|frontend|embed|falkordb] | stop

EOF
}

# =============================================================================

# The header comment, from its first line down to (not including) the paragraph
# about hosted APIs — anchored on text rather than line numbers, so the header
# can grow without cutting the help short.
usage() {
  sed -n '3,/^# Nothing here talks to a hosted API/p' "${BASH_SOURCE[0]}" \
    | sed '$d' | sed 's/^# \{0,1\}//'
}

# -v/--verbose anywhere in the arguments echoes every external command.
# --runtime <name> / --runtime=<name> selects a runtime profile (load_runtime).
ARGS=()
expect_runtime=0
for arg in "$@"; do
  if (( expect_runtime )); then
    [[ -n "$arg" ]] || die "--runtime needs a name. Available: $(runtime_names | tr '\n' ' ')"
    RUNTIME_ARG="$arg"; expect_runtime=0; continue
  fi
  case "$arg" in
    -v|--verbose) VERBOSE=1 ;;
    --runtime)    expect_runtime=1 ;;
    --runtime=*)  RUNTIME_ARG="${arg#--runtime=}"
                  [[ -n "$RUNTIME_ARG" ]] || die "--runtime needs a name. Available: $(runtime_names | tr '\n' ' ')" ;;
    *) ARGS+=("$arg") ;;
  esac
done
(( expect_runtime == 0 )) || die "--runtime needs a name. Available: $(runtime_names | tr '\n' ' ')"
set -- "${ARGS[@]:-}"

if [[ "$VERBOSE" == 1 ]]; then
  note "verbose mode: every external command is echoed before it runs"
fi

CMD="${1:-all}"
case "$CMD" in
  -h|--help|help) usage; exit 0 ;;
  runtimes)       list_runtimes; exit 0 ;;
  setup|start|all|stop|status|logs|test|doctor) ;;
  *) usage; die "unknown command: $CMD" ;;
esac

load_runtime "$CMD"

case "$CMD" in
  setup)
    preflight; install_system_deps; install_uv; install_node
    init_submodule; make_env; install_python_deps; install_node_deps
    pull_images; fetch_models
    step "Setup complete"; note "next: $SELF start"
    ;;
  start)  do_start ;;
  all)
    preflight; install_system_deps; install_uv; install_node
    init_submodule; make_env; install_python_deps; install_node_deps
    pull_images; fetch_models; do_start
    ;;
  stop)   do_stop ;;
  status) do_status ;;
  logs)   shift; do_logs "${1:-}" ;;
  test)   do_test ;;
  doctor) do_doctor ;;
esac
