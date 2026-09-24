# shellcheck shell=bash
#
# Runtime profile: NVIDIA L40S — Ada Lovelace (compute capability 8.9), 48GB of
# DISCRETE GDDR6 (~864 GB/s), x86_64. Written for, and named after, the
# L40S-48C: an NVIDIA vGPU for Compute profile that gives one VM the whole card
# (1:1, no time-slicing with other VMs), reported by nvidia-smi as
# "NVIDIA L40S-48C". A bare-metal L40S matches the same profile.
#
# Sourced by provision_local.sh — not meant to be run. Every value can be
# overridden from the environment or from .env, e.g.
#   docker rm -f sosim-llm && GPU_MEM_UTIL=0.78 ./scripts/provision_l40s.sh start
# (sizing applies when a container is created; start never replaces a running one).
#
# What differs from dgx-spark, and why, in one place:
#
#   image         upstream vllm/vllm-openai, pinned. It compiles sm_89 kernels
#                 (the NGC build's amd64 arch list skips 8.9) and ships CUDA 13.0,
#                 inside what the 595.x vGPU 20.1 guest driver supports (13.2).
#   entrypoint    that image's ENTRYPOINT is already `vllm serve`.
#   weights       the SAME checkpoint as the DGX. Ada has no FP4 tensor cores,
#                 so vLLM serves it with Marlin kernels: FP4 weights, bf16
#                 activations — no less precise than the DGX's native FP4.
#   memory        48GB that nothing else shares, but TWO vLLM servers on it.
#   start order   one server at a time (see GPU_START_ORDER).
#   page cache    not dropped: it is host RAM, not the card's.
#   app GPU       hidden from SoSim's own processes (see HIDE_GPU_FROM_APP).
#
# Everything above the LLM (.env, concurrency keys, tool-call parser, the
# thinking switch in SIM_MODEL_EXTRA_BODY) is the same as on the DGX: the chat
# template is byte-identical across these Qwen3.6-35B-A3B builds.

RUNTIME_LABEL="NVIDIA L40S 48GB (Ada, sm_89, x86_64), incl. the L40S-48C vGPU"
# What this profile was sized and chosen for; setup, start and doctor report a
# host that differs.
RUNTIME_HOST_ARCH=x86_64
RUNTIME_COMPUTE_CAP=8.9
# Kernels the image's torch must carry for this GPU, any one of them. A cubin
# built for X.y runs on X.z when z >= y, so sm_86 and sm_80 builds serve an 8.9
# card: PyTorch's own wheels list 8.6 and never 8.9, and that is fine.
RUNTIME_TORCH_ARCH="sm_89 sm_86 sm_80"

# Pinned, never `latest`: a floating tag can move to a CUDA the guest driver
# does not have (the cu134 nightlies already need 13.4), and a vGPU guest cannot
# use CUDA forward compatibility to paper over that. v0.30.0 is the first
# release checked against this stack: qwen3_xml, --runner pooling,
# --language-model-only and the NVFP4 Marlin fallback are all in it.
# (amd64 digest at the time of writing, for anyone who wants it immutable:
#  sha256:5f5e535216848d0c52159c8c13a0af04be5f6fe1a84e79914300610796f76d40)
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.30.0}"
# Upstream images set ENTRYPOINT ["vllm", "serve"]. Handing them `vllm serve`
# again would run `vllm serve vllm serve <model>`, so the entrypoint is
# replaced by `vllm` and the command starts at `serve`.
VLLM_ENTRYPOINT="${VLLM_ENTRYPOINT-vllm}"

# The DGX checkpoint. On first start the vLLM log should say it chose the
# MARLIN NvFp4 MoE backend; EMULATION would mean something is wrong, and doctor
# says so. This path is supported in vLLM's code but far less exercised than
# native FP4, so run doctor (it sends a real agent-shaped tool call) before a
# long simulation. If Marlin NVFP4 misbehaves here, the fallback is an AWQ int4
# build of the same base with the same chat template, served by vLLM's
# much-used AWQ-Marlin kernels:
#
#   export LLM_MODEL_REPO=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
#   export LLM_MODEL_REVISION=00fcea2d3bcf5389b518d4fc082e5590e0ba4844
#   ./scripts/provision_l40s.sh setup && docker rm -f sosim-llm && ./scripts/provision_l40s.sh start
#
# (setup downloads it; the running container must go, or start keeps serving
# NVFP4; and both variables must stay set for every later command — put them in
# a shell profile. The revision matters: that repo's weights have been replaced
# in place once already.) Do not reach for an
# FP8 or 8-bit build: at 37-40GB they leave 1-5GB of a 48GB card for the KV
# cache once the embeddings server is up. Never a GPTQ build with desc_act=true:
# v0.30.0 dropped act-order kernels and silently ignores g_idx.
LLM_MODEL_REPO="${LLM_MODEL_REPO:-RedHatAI/Qwen3.6-35B-A3B-NVFP4}"

# Fraction of the card for the LLM, as vLLM reads it: requested = total x this,
# and it refuses to start unless that much is FREE at that moment. Its own CUDA
# context sits outside the fraction, and so does everything else on the card:
# the embeddings server (~2-2.5GB actually used), the desktop session a vGPU VM
# may run (gnome-shell holds ~0.2GB and grows with use), and allocator slack.
#
# 0.80 of 48GB is ~38.4GB: ~21GB of weights with the vision tower skipped,
# ~2.5GB of activations and CUDA graphs, and ~15GB of KV cache and recurrent
# state. This model keeps a KV cache in only 10 of its 40 layers (the rest are
# Gated DeltaNet), so a full 32K-token sequence costs ~0.77GB with vLLM's block
# padding, and 16 of them need ~12.2GB.
#
# Deliberately short of what the card seems to allow: a vGPU can hold 3-6GB of
# a 48C profile's framebuffer in reserve (more with ECC on), out of reach of the
# guest. After the first start, read "Available KV cache memory" in
# `docker logs sosim-llm` and `nvidia-smi --query-gpu=memory.free` with the
# stack up, and only then raise this towards 0.85 (then recreate sosim-llm). Do not tune from the
# "GPU KV cache size: N tokens" line, which vLLM gets wrong for hybrid models.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
# For a pooling model this is only vLLM's startup check (this much must be
# free); it reserves nothing and caps nothing, because an encoder-only model has
# no KV cache to size. bge-m3 loads as fp16 (~1.1GB of weights) and runs in
# ~2-2.5GB. The DGX value, and enough.
EMBED_GPU_MEM_UTIL="${EMBED_GPU_MEM_UTIL:-0.08}"
EMBED_MAX_MODEL_LEN="${EMBED_MAX_MODEL_LEN:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Same as the DGX, so the concurrency keys .env.example ships (8 / 2 x 4) stay
# right. This card has ~3x the DGX's memory bandwidth, so more may pay off —
# sweep 16/24/32 and watch p95 latency before raising it, and re-derive the
# three keys in .env when you do (see README, "Wiping everything").
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
# Same model family and byte-identical chat template as the DGX; see the long
# note in dgx-spark.sh for what a mismatched parser looks like.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"

# The page cache is host RAM; vLLM on a discrete card sizes itself from the
# card's own free memory. Dropping it would only force the weights to be read
# from disk again.
DROP_PAGE_CACHE="${DROP_PAGE_CACHE:-0}"

# vLLM sizes its KV cache from the drop in the DEVICE's free memory while it
# profiles, so a second server allocating (or freeing) at the same moment is
# charged (or credited) to it — a smaller cache, or an oversized one that OOMs
# later, and at the extreme "Error in memory profiling". The embeddings server
# does no such profiling and loads in seconds, so it goes first and the LLM
# measures a settled card.
#
# After a reboot, --restart unless-stopped brings both containers back at once
# with no ordering. If the LLM then comes up with a small cache or does not come
# up, `docker rm -f sosim-llm && ./scripts/provision_l40s.sh start` restores the
# order (with LLM_MODEL_REPO/LLM_MODEL_REVISION set, if you switched to AWQ).
GPU_START_ORDER="${GPU_START_ORDER:-serial}"

# A first load reads ~21GB of weights, compiles, repacks them for Marlin and
# captures CUDA graphs; on a VM's disk that can outlast the 10 minutes the DGX
# is given. (If it still does, the container keeps loading — re-run start.)
LLM_WAIT_TRIES="${LLM_WAIT_TRIES:-450}"   # ~15 min

# The backend's torch has CUDA on x86_64, and OASIS moves its Twitter
# recommender onto the GPU whenever it can see one — twhin-bert with batches of
# 1000, whose attention transients run from ~2GB to tens of GB. Next to vLLM
# that is an OOM. Hidden, it runs on the CPU, exactly as it does on the DGX.
HIDE_GPU_FROM_APP="${HIDE_GPU_FROM_APP:-1}"

# Check before starting that the card has room for both servers (doctor and
# start print what else holds memory when it does not).
GPU_MEM_PRECHECK="${GPU_MEM_PRECHECK:-1}"

# --language-model-only: the checkpoint carries a vision tower SoSim never uses
#   (it sends no images). Skipping it saves ~0.9GB of weights and the
#   image-encoder memory test.
# --max-num-batched-tokens 8192: vLLM picks its prefill chunk from the card's
#   size, 2048 tokens below 70GB and 8192 above, so without this the L40S would
#   prefill Graphiti's long extraction prompts in quarter-size chunks the DGX
#   never uses.
# Plain flags only: this string is split on whitespace, with no quote handling.
# Set LLM_EXTRA_ARGS= (empty) to pass none.
LLM_EXTRA_ARGS="${LLM_EXTRA_ARGS---language-model-only --max-num-batched-tokens 8192}"
EMBED_EXTRA_ARGS="${EMBED_EXTRA_ARGS:-}"

# Added to .env by setup, both when it creates the file and when it first
# records SOSIM_RUNTIME in an existing one (a host switching to l40s). Only ever
# added when absent — a value already in .env is never rewritten — and start and
# doctor warn when it is missing. This model thinks before it answers unless told not to, and with
# SIM_MODEL_MAX_TOKENS=1024 a thinking agent spends its whole budget before it
# reaches the tool call — see SIM_MODEL_EXTRA_BODY in .env.example. The single
# quotes are needed: `source` would strip double quotes out of the JSON.
ENV_SEED="${ENV_SEED-SIM_MODEL_EXTRA_BODY='{\"chat_template_kwargs\":{\"enable_thinking\":false}}'}"
