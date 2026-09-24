# shellcheck shell=bash
#
# Runtime profile: NVIDIA DGX Spark — GB10 (sm_121, Blackwell), aarch64,
# 128GB UNIFIED memory shared by the CPU and the GPU.
#
# This is the original target, and the one provision_local.sh falls back to when
# nothing selects a runtime, so a host provisioned before runtime profiles
# existed keeps behaving exactly as it did. Every value below is the default the
# script shipped with before the split; change one and the DGX build changes.
#
# Sourced by provision_local.sh — not meant to be run. Each value can still be
# overridden from the environment (GPU_MEM_UTIL=0.6 ./scripts/provision_dgx_spark.sh start).

RUNTIME_LABEL="NVIDIA DGX Spark (GB10, sm_121, aarch64, 128GB unified memory)"
# What this profile was chosen and tested for; doctor, setup and start report a
# host that differs. (Other GPUs are not necessarily unable to load these
# weights — vLLM falls back to Marlin W4A16 kernels for NVFP4 on anything from
# compute capability 7.5 — but nothing else was sized or measured here.)
RUNTIME_HOST_ARCH=aarch64
RUNTIME_COMPUTE_CAP=12.1
RUNTIME_TORCH_ARCH=sm_121

# NGC's vLLM build is the tested path on GB10. Upstream vllm/vllm-openai has
# been reported broken on this chip (its bundled torch compiles only through
# sm_120; GB10 is sm_121) — `doctor` checks for that explicitly.
VLLM_IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.05.post1-py3}"
# The NGC image's own entrypoint execs its arguments, so the containers are
# given `vllm serve ...` as their command. Empty = keep the image's entrypoint.
VLLM_ENTRYPOINT="${VLLM_ENTRYPOINT:-}"

LLM_MODEL_REPO="${LLM_MODEL_REPO:-RedHatAI/Qwen3.6-35B-A3B-NVFP4}"

# Fraction of TOTAL device memory. On unified memory this competes with the OS,
# the container runtime and the page cache. Published DGX Spark recipes range
# 0.4–0.87; 0.90 has been observed getting the engine SIGTERM'd by earlyoom,
# which does NOT look like an OOM in the logs. Start conservative.
# NOTE: this is a fraction of TOTAL device memory, and the embeddings server is
# a second vLLM process on the same pool, so the two must sum well under 1.0
# alongside the OS and page cache.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
# bge-m3 is ~2.2GB of weights; it needs very little.
EMBED_GPU_MEM_UTIL="${EMBED_GPU_MEM_UTIL:-0.08}"
EMBED_MAX_MODEL_LEN="${EMBED_MAX_MODEL_LEN:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# vLLM will admit this many concurrent sequences. Decode on GB10 is
# bandwidth-bound (~273 GB/s) and divides across sequences, so admitting 32
# does not serve 32 at single-stream speed. Sweep before trusting a number.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
# Qwen3 family. This MUST match the tool-call dialect the model actually emits,
# and hermes — the old default here — does not match the shipped
# RedHatAI/Qwen3.6-35B-A3B-NVFP4. That build emits the XML form
# (<function=name><parameter=x>...), hermes expects JSON inside <tool_call>, and
# every single agent request therefore died in the server with:
#
#   ERROR hermes_tool_parser.py:139 Error in extracting tool call from response
#   json.decoder.JSONDecodeError: Expecting value: line 2 column 1 (char 1)
#
# That failure is close to invisible from the client: vLLM logs the traceback,
# then returns 200 with the unparsed markup dumped into message.content. OASIS
# sees an answer carrying no tool call, records no action, and the run reports
# full rounds against an empty action log — which is exactly how it presented,
# as a simulation that "ran" for 7.7 hours and produced nothing.
#
# Measured on a DGX Spark (2026-09-14): with qwen3_xml the preflight returns
# finish_reason=tool_calls in 8.1s; with hermes the identical request comes back
# as prose. Change this only for a model whose dialect you have checked, and
# check it with:
#
#   backend/.venv/bin/python backend/scripts/run_parallel_simulation.py --preflight-only
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"

# Unified memory means the OS page cache eats into the KV cache budget, so the
# page cache is dropped (with passwordless sudo, if available) before the LLM
# loads.
DROP_PAGE_CACHE="${DROP_PAGE_CACHE:-1}"

# The embeddings server and the LLM are launched back to back and then both
# waited on, as they always have been here.
GPU_START_ORDER="${GPU_START_ORDER:-parallel}"

# SoSim's own processes get CPU-only torch on aarch64 (PyPI publishes no CUDA
# wheels for it), so there is no GPU to hide from them.
HIDE_GPU_FROM_APP="${HIDE_GPU_FROM_APP:-0}"

# Nothing beyond the shared flags provision_local.sh always passes.
LLM_EXTRA_ARGS="${LLM_EXTRA_ARGS:-}"
EMBED_EXTRA_ARGS="${EMBED_EXTRA_ARGS:-}"
