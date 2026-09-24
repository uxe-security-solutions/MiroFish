<div align="center">

<img src="./static/image/sosim-logo.svg" alt="SoSim" width="360"/>

# SoSim - Social Simulator

**Rehearse a decision against a synthetic population before you make it.**

[![Docker](https://img.shields.io/badge/Docker-Build-2496ED?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue?style=flat-square)](./LICENSE)

</div>

## Overview

SoSim builds a high-fidelity synthetic society from source material you supply, runs
it forward, and reports what happened.

You give it seed material - a news package, a policy draft, an incident report, a
market brief, a novel - and a question in plain English. SoSim extracts the entities
and relationships into a temporal knowledge graph, generates a population of agents
with distinct personas, memories and behavioural rules, and then lets that population
interact on simulated Twitter and Reddit for as many rounds as you allow. Afterwards a
report agent interrogates the resulting graph and writes up the outcome, and you can
interview any individual agent about why it did what it did.

Everything runs on one machine. There is no hosted LLM, no hosted memory service, and
nothing leaves the box once setup has finished.

### Where it is useful

- **Macro** - a rehearsal lab for decision-makers. Test a policy, a disclosure or a
  crisis response against a population before it is real, at zero risk.
- **Micro** - a sandbox for exploring a scenario. Play out an alternative ending, a
  counterfactual, or a "what if we had said nothing".

## How a run works

1. **Graph building** - seed extraction, memory injection, GraphRAG construction.
2. **Environment setup** - entity and relationship extraction, persona generation,
   agent configuration.
3. **Simulation** - Twitter and Reddit run in parallel, the prediction question is
   parsed automatically, and the temporal graph is updated as the run proceeds.
4. **Report generation** - a report agent works the post-simulation graph with a
   toolset of searches, panoramas and agent interviews.
5. **Deep interaction** - chat with any agent in the simulated world, or with the
   report agent about its own report.

## Fully local deployment

Everything runs on one machine: a vLLM chat server, a second vLLM serving embeddings,
FalkorDB, the Zep-compatible shim, the backend and the frontend. What depends on the
hardware - the vLLM image, the weights, GPU memory, start-up order - lives in a
**runtime profile** under [`scripts/runtimes/`](scripts/runtimes/), and each profile
has its own entry script:

| Runtime | Hardware | Entry script |
|---|---|---|
| `dgx-spark` | NVIDIA DGX Spark - GB10 (sm_121), aarch64, 128GB unified memory | [`scripts/provision_dgx_spark.sh`](scripts/provision_dgx_spark.sh) |
| `l40s` | NVIDIA L40S 48GB - Ada (sm_89), x86_64, including the `L40S-48C` vGPU | [`scripts/provision_l40s.sh`](scripts/provision_l40s.sh) |

Both wrap [`scripts/provision_local.sh`](scripts/provision_local.sh), which takes the
same commands and does the work. The DGX Spark profile is exactly what the script
shipped with before profiles existed: same image, same weights, same flags, same
containers - [`scripts/tests/test_runtimes.sh`](scripts/tests/test_runtimes.sh) pins its
`docker run` lines word for word.

| | `dgx-spark` | `l40s` |
|---|---|---|
| vLLM image | NGC `nvcr.io/nvidia/vllm:26.05.post1-py3` | `vllm/vllm-openai:v0.30.0` (compiles sm_89; CUDA 13.0) |
| LLM weights | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` - native FP4 | the same checkpoint - Marlin kernels (FP4 weights, bf16 activations) |
| `GPU_MEM_UTIL` / `EMBED_GPU_MEM_UTIL` | 0.70 / 0.08 of a unified pool | 0.80 / 0.08 of a 48GB card |
| Context / batch | 32768 / 16 sequences | 32768 / 16 sequences |
| Extra LLM flags | none | `--language-model-only --max-num-batched-tokens 8192` |
| Server start order | both launched, then both awaited | embeddings healthy first, then the LLM (allowed ~15 min to load) |
| Page cache dropped before the LLM loads | yes (unified memory) | no (host RAM is not VRAM) |
| GPU visible to backend / simulations | n/a - aarch64 torch is CPU-only | hidden (`CUDA_VISIBLE_DEVICES=`), so they run on CPU as on the DGX |
| New `.env` also gets | - | `SIM_MODEL_EXTRA_BODY` with thinking off (see below) |
| Doctor also checks | sm_121 kernels, arm64 images | sm_89/sm_86 kernels, amd64 images, vGPU license, free VRAM |

Everything above the model server - `.env`, the concurrency keys, the tool-call parser,
the thinking switch - is shared: the chat template is byte-identical across these
Qwen3.6-35B-A3B builds.

### Prerequisites

Docker with an NVIDIA container runtime, `sudo` for package installs, and ~120GB free
disk. The provisioning script installs everything else (uv, Node 22, build tools).

> Node 18 is **not** enough despite `package.json` saying so - the pinned Vite 7
> requires `^20.19 || >=22.12`, and `npm ci` only warns. The script installs Node 22.

### Running it

```bash
git clone --recurse-submodules git@github.com:uxe-security-solutions/MiroFish.git
cd MiroFish
./scripts/provision_dgx_spark.sh all    # on a DGX Spark
./scripts/provision_l40s.sh all         # on an L40S server
```

> The product is SoSim; the Git repository is still named `MiroFish`. That is
> deliberate - the repository is not being renamed, so every clone URL and remote
> path stays as it is.

If you cloned without `--recurse-submodules`, the script initialises the submodule
itself. `all` = `setup` then `start`. Every command works with any of the three
scripts (shown here with the L40S one):

```bash
./scripts/provision_l40s.sh setup     # packages, submodule, .env, images, models - NEEDS NETWORK
./scripts/provision_l40s.sh start     # bring the stack up - fully offline
./scripts/provision_l40s.sh status    # what is up, and where
./scripts/provision_l40s.sh logs llm  # or: backend, zep-shim, frontend, embed, falkordb
./scripts/provision_l40s.sh doctor    # runtime vs this host, GPU kernels, image arch, config sanity
./scripts/provision_l40s.sh test      # the test suites (no GPU, no network)
./scripts/provision_l40s.sh stop
./scripts/provision_local.sh runtimes # list the profiles and which one this host uses
```

Add `-v` (or `VERBOSE=1`) to echo every external command as it runs.

`setup` is the only stage that touches the internet. After it completes the machine
can be disconnected: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and
`GRAPHITI_TELEMETRY_ENABLED=false` are set in `.env.example`.

Review `.env` before the first `start` - it is created from `.env.example` and never
overwritten.

### Which runtime a command uses

Most explicit first:

1. The wrapper you ran (it passes `--runtime <name>`), `provision_local.sh --runtime <name>`,
   or `SOSIM_RUNTIME=<name>` in the environment.
2. `SOSIM_RUNTIME` in `.env`. `setup` writes it, so after the first `setup` the bare
   `./scripts/provision_local.sh start` keeps using the same profile.
3. With nothing recorded, the one profile whose CPU architecture and GPU compute
   capability both match this host (a GB10 gives `dgx-spark`, an L40S `l40s`).
4. Otherwise `dgx-spark` - which is why a DGX Spark set up before profiles existed
   carries on unchanged with the old command.

`setup`, `start` and `all` refuse, before anything is touched:

- a wrapper that contradicts `SOSIM_RUNTIME` in `.env` - the weights on disk belong to
  the recorded runtime, and starting the other one would only fail minutes later inside
  vLLM;
- a runtime this host positively belongs to another profile for - the other wrapper run
  by mistake, or an `.env` carried over from the other machine. The message names the
  right command. `SOSIM_FORCE_RUNTIME=1` overrides it, for a deliberate experiment.

To really move a host, stop the stack, delete the `SOSIM_RUNTIME` line from `.env`, and
run the new runtime's `setup`. `stop`, `logs` and `test` never query the GPU, so they
still work with a wedged driver.

Every profile value can be overridden per command from the environment, e.g.
`GPU_MEM_UTIL=0.78 ./scripts/provision_l40s.sh start`, or for good by adding it to
`.env`, which `start` reads after the profile. **Sizing only applies when a container is
created**: `start` leaves a running server alone (and warns when its flags differ from the
current values), so recreate it to apply a change -
`docker rm -f sosim-llm && GPU_MEM_UTIL=0.78 ./scripts/provision_l40s.sh start`.

A new runtime is one more file in `scripts/runtimes/` (copy the closest one;
`load_runtime` in `provision_local.sh` refuses a profile that leaves a required value
unset) plus a two-line wrapper.

### On the L40S

What the box needs before `setup`: Docker with the NVIDIA Container Toolkit (so
`docker run --gpus all` works in the VM), the vGPU guest driver left in place (not
replaced by a distro CUDA package), and ~70GB free disk for the weights (~25GB), bge-m3,
the recommender model and the vLLM image (~9GB compressed).

- **The vGPU has to be licensed.** An NVIDIA vGPU for Compute guest runs at full speed
  for 20 minutes after boot and then drops to idle-level compute until it gets a
  license. Nothing crashes - `/v1/models` still answers - the agents just stop finishing
  their requests. `doctor`, `setup` and `start` check `nvidia-smi -q` and fail loudly
  when the license status is not `Licensed`.
- **Same weights, different kernels.** Ada has no FP4 tensor cores, so vLLM serves the
  NVFP4 checkpoint through Marlin (weight-only FP4, bf16 activations). That path is
  supported but far less exercised than native FP4: after the first `start`, run
  `doctor` - it reports the MoE backend vLLM chose (it should say `MARLIN`; `EMULATION`
  is a failure) and sends one real agent-shaped tool call. Read a few agent posts from
  the first short run, too: garbled text is the one failure a tool-call check will not
  catch. If it misbehaves, serve an AWQ build of the same base instead, pinned to the
  commit that was checked (its weights have been replaced in place before). `setup`
  downloads it; the running NVFP4 container has to be removed, or `start` keeps it:

  ```bash
  export LLM_MODEL_REPO=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
  export LLM_MODEL_REVISION=00fcea2d3bcf5389b518d4fc082e5590e0ba4844
  ./scripts/provision_l40s.sh setup && docker rm -f sosim-llm && ./scripts/provision_l40s.sh start
  ```

  The exports last only for that shell. Put both lines in the shell profile of whoever
  runs these scripts (or in `.env`, which `start` reads - but `setup` does not), or a
  later recreation - the reboot recovery below, say - quietly goes back to NVFP4.

  Not FP8 or 8-bit builds: at 37-40GB they leave 1-5GB of the card for the KV cache.
- **One vLLM server at a time.** vLLM sizes its KV cache from the card's free memory
  while it profiles, and a second server loading at the same moment skews that. `start`
  therefore waits for the embeddings server before it creates the LLM. After a reboot,
  Docker restarts both containers at once; if the LLM comes back with a small cache (see
  `Available KV cache memory` in `docker logs sosim-llm`) or not at all, recreate it:
  `docker rm -f sosim-llm && ./scripts/provision_l40s.sh start` (with the AWQ variables
  set, if you switched).
- **Headroom.** `GPU_MEM_UTIL=0.80` is ~38.4GB for the LLM, ~15GB of it KV cache -
  enough for all 16 sequences at the full 32K context. The embeddings server (~2-2.5GB),
  both CUDA contexts, any desktop session in the VM (gnome-shell holds ~0.2GB and grows
  with use) and whatever part of the framebuffer the vGPU keeps in reserve share the
  rest. `start` checks the card's free memory before it launches anything. Once the
  stack is up, `Available KV cache memory` in `docker logs sosim-llm` and
  `nvidia-smi --query-gpu=memory.free --format=csv` show how much room is really left;
  raise `GPU_MEM_UTIL` towards 0.85 only from those numbers, and recreate the container to
  apply it (see "Which runtime a command uses"). Running the VM headless
  (`sudo systemctl set-default multi-user.target`) keeps the margin intact.
- **Thinking is switched off in a new `.env`.** This model reasons before it answers
  unless told not to, and within `SIM_MODEL_MAX_TOKENS=1024` a reasoning agent never
  reaches its tool call. When `setup` creates `.env` on this runtime it therefore adds
  `SIM_MODEL_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'`. An
  existing `.env` gets it only when `setup` first records the runtime in it (a host
  switching to `l40s`); otherwise `start` and `doctor` warn that it is missing.
- **The backend's torch has CUDA on x86_64** (~4GB of `nvidia-*` wheels, harmless),
  and OASIS would put its Twitter recommender on the GPU - with batches of 1000 posts
  that is an OOM next to vLLM. The script starts the backend and shim with
  `CUDA_VISIBLE_DEVICES=` so it runs on the CPU, as it does on the DGX. If you run the
  backend by hand (`npm run dev`) on this host, do the same.

### Upgrading from a pre-rename checkout

The containers and the FalkorDB volume were renamed along with the product:
`mirofish-llm` / `mirofish-embed` / `mirofish-falkordb` are now `sosim-llm` /
`sosim-embed` / `sosim-falkordb`, and the volume `mirofish_falkordb` is now
`sosim_falkordb`.

**There is no data migration, and this is intentional.** FalkorDB comes up on a fresh
volume, so every graph built before the rename is gone. The old volume is left on disk
untouched rather than deleted behind your back; `start` names it and prints the command
to reclaim the space:

```bash
docker volume rm mirofish_falkordb
```

The old containers ran with `--restart unless-stopped`, so they come back after a
reboot and keep holding the GPU and ports 8000 / 8081 / 6379. `start` force-removes
them before it brings anything up, because the renamed containers cannot bind those
ports while they are alive.

### Wiping everything and starting again

Sometimes the right move is to throw away every graph, every simulation and every
uploaded document and reinstall. There is no single command for it, because the state
lives in four unrelated places: the shim's SQLite batch record, the FalkorDB volume,
the backend's upload tree, and `.env`.

> **Never delete `data/hf-cache/`.** It is bind-mounted into both vLLM containers with
> `HF_HUB_OFFLINE=1`, so an empty cache does **not** quietly re-download the weights -
> the containers *fail to start*, and on a box that has since been disconnected you
> cannot get them back. This is the most destructive mistake available during a wipe,
> and `rm -rf data/` is how you make it. Delete the named files below, never the
> directory holding them.

The commands below use `provision_l40s.sh`; on a DGX Spark use `provision_dgx_spark.sh`.
Use the wrapper rather than the bare `provision_local.sh`: step 4 moves `.env` - and the
`SOSIM_RUNTIME` recorded in it - out of the way, and the wrapper is what puts it back.

```bash
cd /path/to/MiroFish
./scripts/provision_l40s.sh stop

# 1. The shim's SQLite state: which episodes of which build already committed.
#    BOTH paths, because which one is live depends on your .env vintage - the
#    ./data/zep_compat.sqlite3 this repo used to ship resolves against the shim's
#    own working directory, while an .env without the key lands in the repo's data/.
#    Normally you would compare them before touching either; here you are wiping.
rm -f data/zep_compat.sqlite3
rm -f third_party/graphiti/server/data/zep_compat.sqlite3

# 2. The graph database. `stop` only stops the container, and a volume cannot be
#    removed while any container still references it - even a stopped one.
#    "No such container" / "no such volume" here just means it was already gone.
docker rm -f sosim-falkordb mirofish-falkordb
docker volume rm sosim_falkordb
docker volume rm mirofish_falkordb   # the pre-rename orphan, if you still have it

# 3. Projects, simulations and reports - uploaded sources, agent state, output.
rm -rf backend/uploads/projects backend/uploads/simulations backend/uploads/reports

# 4. Take the current defaults. `setup` NEVER overwrites an existing .env, so without
#    this step the "fresh" install silently keeps every value you had tuned before.
#    data/ is gitignored, so the backup cannot be committed by accident.
mv .env data/env.pre-wipe.bak

./scripts/provision_l40s.sh setup   # recreates .env and records the runtime; weights and images are cached

# 5. Put back anything you had deliberately tuned - BEFORE start, because sizing
#    only applies when the containers are created (see below).
diff data/env.pre-wipe.bak .env

./scripts/provision_l40s.sh start
./scripts/provision_l40s.sh doctor
```

`setup` is the right way back up, and it is quick after a wipe, because the weights are
still in `data/hf-cache/` and the images are still in Docker. Be precise about what it
does, though: it re-creates `.env` from `.env.example`, and then adds only the keys that
are *missing* from it — in practice `SOSIM_RUNTIME`, `ZEP_COMPAT_DB_PATH` (which a
fresh copy does not pin to an absolute path) and, on `l40s`, `SIM_MODEL_EXTRA_BODY`
with thinking off. It does **not**
re-derive the concurrency keys for this host's `MAX_NUM_SEQS`. `ensure_env_key` only
writes a key that is *absent*, and the fresh copy already carries `SIM_LLM_SEMAPHORE`,
`ZEP_COMPAT_BATCH_CONCURRENCY` and `SEMAPHORE_LIMIT` at the values derived for the
default `MAX_NUM_SEQS=16`, so none of them fires (the script's own comment says so).

**If you run a non-default `MAX_NUM_SEQS`, edit those three keys by hand** in the new
`.env` — `SIM_LLM_SEMAPHORE` is half of it, and
`ZEP_COMPAT_BATCH_CONCURRENCY × SEMAPHORE_LIMIT` must stay at or under
`MAX_NUM_SEQS / 2` — then check the result with the same value in the environment:

```bash
MAX_NUM_SEQS=32 ./scripts/provision_l40s.sh doctor
```

Both `setup` and `doctor` re-check that product against the current `MAX_NUM_SEQS` and
report it as a failure when it no longer fits. Neither of them rewrites it for you.

**On an air-gapped box, `setup` will report a failed download for each model** - its
`fetch_models` stage forces `HF_HUB_OFFLINE=0` deliberately, so it always tries the
network. Everything else has already been done by the time it gets there, so the run is
still useful. If you would rather not see the failures, replace step 4 and `setup` with:

```bash
mv .env data/env.pre-wipe.bak && cp .env.example .env
diff data/env.pre-wipe.bak .env    # put back what you had tuned, before start
./scripts/provision_l40s.sh start
./scripts/provision_l40s.sh doctor
```

(`start` does not record the runtime - only `setup` does - so the new `.env` has no
`SOSIM_RUNTIME` line until you add one. Until then, bare `provision_local.sh` commands
pick the profile that matches this host's GPU. On `l40s` it also lacks the thinking-off
line `setup` would have added; `start` warns and prints the line to copy in.)

If you only notice a missing value after `start`, putting it into `.env` is not enough:
the running servers keep what they were created with. Restart the stack so they pick it
up - `./scripts/provision_l40s.sh stop && ./scripts/provision_l40s.sh start` - and then run
`doctor`, which warns when the running LLM's sizing differs from `.env`.

Also *not* part of the wipe: `data/logs/` (append-only, and it holds the evidence for
whatever made you wipe) and `data/run/` (pidfiles, already cleared by `stop`).

### Reading the output when something breaks

Failures are designed to be impossible to miss from the console alone:

- Each one prints `✗ FAILED: <what>` the moment it happens.
- **The relevant log tail is dumped inline** - a service that dies on startup shows
  you *why* (`sh: vite: command not found`, a vLLM flag rejection, an OOM) rather than
  only surfacing as a health-check timeout minutes later.
- Every container is checked for liveness a few seconds after launch, and every
  background process is checked 2s after launch, so an instant exit is caught at the
  point it happens.
- `start` deliberately **continues past a failure** so you get the whole picture in one
  run instead of stopping at the first problem.
- All failures are listed again at exit, and the script **exits non-zero** - so it
  composes with CI or `&&`.

Health-gate patience is tunable, which matters on slower storage (a first model load
reads tens of GB) and for failing fast while testing: `EMBED_WAIT_TRIES`,
`LLM_WAIT_TRIES`, `SHIM_WAIT_TRIES`, `BACKEND_WAIT_TRIES`, `FRONTEND_WAIT_TRIES` - each
counts 2-second polls.

### When a simulation runs but records no actions

A run that reports rounds against an empty action log is not a simulation of a quiet
crowd - it is a model backend that answered none of the agents. The round loop absorbs
per-agent failures on purpose, so nothing in the round count distinguishes the two.

Ask the endpoint directly. This sends **one agent-shaped request** - an agent-sized
prompt, the same tool schemas, the same timeout and output cap the agents get - and
says which limit it hits:

```bash
backend/.venv/bin/python backend/scripts/run_parallel_simulation.py --preflight-only
```

`./scripts/provision_local.sh doctor` runs the same check. The three answers that
matter, and what each one means:

| What it reports | What is wrong | Fix |
|---|---|---|
| `APITimeoutError ... did not finish an agent-sized one within Ns` | The endpoint is up but cannot finish a real request in time - it is overloaded, still loading a model, or wedged | Check `docker logs sosim-llm`; lower `SIM_LLM_SEMAPHORE`, or raise `SIM_MODEL_TIMEOUT` |
| `finish_reason=length ... spent the cap on a reasoning block` | A hybrid model is in reasoning mode, so it never reaches the tool call | Set `SIM_MODEL_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'` in `.env` — **single-quoted**, or `source` strips the inner quotes — or serve vLLM with a matching `--reasoning-parser` |
| `the model DID emit a tool call but the server did not parse it` | `--tool-call-parser` does not match what this model emits. vLLM logs a parser traceback and still returns 200 with the raw markup as content, so from the client it looks like a disobedient model | `TOOL_CALL_PARSER=qwen3_xml` for Qwen3 builds; confirm with `docker logs sosim-llm \| grep -i tool_parser` |
| `answered in prose but called no tool` | Tool calling is off entirely | Check `--enable-auto-tool-choice` |

The knob behind the second row is the one worth knowing about in advance.
**camel-ai sends no `max_tokens` of its own**, and vLLM then lets a generation run to
`--max-model-len` minus the prompt - around 30k tokens on the shipped settings. A model
that opens with a `<think>` block will spend that budget, and at the per-stream decode
rate of a fully batched local server it cannot finish inside any sane timeout: every
agent request times out and the run records nothing. `SIM_MODEL_MAX_TOKENS` (default
1024) bounds it, which turns that silent multi-hour timeout into an immediate
`finish_reason="length"` the preflight can explain.

A run that gets past the preflight and still stalls aborts itself: one round that
activates agents, records nothing, and takes longer than `SIM_MODEL_TIMEOUT` is enough
to stop, because agents that choose to do nothing answer quickly.

### Ports to open on the network

**Expose exactly one port.**

| Port | Service | Expose? |
|---|---|---|
| **3000** | Frontend (Vite) | **Yes** - this is the whole application. It proxies `/api` to the backend. |
| 5001 | Backend API (Flask) | No - reached through the frontend proxy |
| 8088 | Zep-compatible shim | No |
| 8000 | vLLM (OpenAI-compatible) | No |
| 8081 | Embeddings (vLLM, pooling mode) | No |
| 6379 | FalkorDB | No |
| 3001 | FalkorDB browser UI | No - handy over an SSH tunnel |

Everything except 3000 binds to `127.0.0.1`, so a host firewall is a backstop, not the
primary control. For example:

```bash
sudo ufw allow 3000/tcp
```

Then open `http://<host-ip>:3000`. To reach it by hostname instead of IP, set
`VITE_ALLOWED_HOSTS` in `.env` - Vite rejects unknown `Host` headers (bare IPs always
work).

For anything beyond a trusted LAN, put a reverse proxy with TLS in front of 3000 rather
than exposing Vite's dev server directly. Note the backend has no auth of its own:
whoever reaches the UI can drive simulations.

### What runs where

```
browser ──▶ :3000 Vite ──/api──▶ :5001 backend ──▶ :8088 Zep shim ──▶ :6379 FalkorDB
                                      │                   │
                                      └───────────────────┴──▶ :8000 vLLM  (agents + extraction)
                                                          └──▶ :8081 vLLM  (embeddings, pooling)
```

The shim needs an embeddings endpoint because Zep Cloud did embedding server-side and
Graphiti does not. `EMBEDDING_DIM` is **a one-way door**: the vector index dimension is
fixed at creation, so changing embedders later means re-embedding every graph. The
default (bge-m3, 1024) matches Graphiti's default exactly.

### Model choices

Defaults are set per runtime in `scripts/runtimes/<name>.sh` and overridable by
environment:

- **LLM** - `RedHatAI/Qwen3.6-35B-A3B-NVFP4` on both runtimes. A Mixture-of-Experts
  model (3B active of 35B) is the right shape here: decode is bandwidth-bound - ~273 GB/s
  on GB10, ~864 GB/s on the L40S - and MoE reads far fewer weights per token. At ~21GB
  for the language model it also leaves a 48GB card room for the KV cache, which only 10
  of its 40 layers need.
- **Embeddings** - `BAAI/bge-m3` on a second vLLM container in pooling mode, from the
  same image as the LLM. HuggingFace TEI would have been the obvious choice but
  publishes **no arm64 image at all** - every `cpu-*` tag is amd64-only - and one image
  for both servers is one fewer dependency on either runtime. It takes a small slice of
  GPU memory (`EMBED_GPU_MEM_UTIL`, `0.08` on both); `GPU_MEM_UTIL` for the main LLM is
  `0.70` on the DGX's unified pool and `0.80` on the L40S's dedicated card. Each is a
  fraction of *total* memory.
- **Graph DB** - FalkorDB (Redis-based, no JVM). Neo4j works too:
  `GRAPHITI_DB_BACKEND=neo4j`.

Two different requirements land on that one LLM endpoint, and both matter:

- OASIS agents use **native OpenAI tool calling**. Without
  `--enable-auto-tool-choice --tool-call-parser`, every agent silently does nothing.
- Graphiti uses **`response_format: json_schema`**, not tool calling. Its lever is
  vLLM's structured-output backend. Tuning tool-call flags does nothing for extraction
  quality.

### Sizing and expectations, honestly

- **DGX Spark:** `GPU_MEM_UTIL` defaults to a conservative `0.70`. It is a fraction of
  *total* device memory, and on unified memory that competes with the OS, the container
  runtime and the page cache. `0.90` has been reported getting the engine SIGTERM'd by
  `earlyoom`, which does **not** look like an OOM in the logs.
- **L40S:** `GPU_MEM_UTIL` is `0.80` of a card nothing else shares, which leaves ~15GB
  of KV cache and recurrent state - enough for all 16 sequences at the full 32K context.
  Raise it towards `0.85` only after reading `Available KV cache memory` in
  `docker logs sosim-llm`; ignore the `GPU KV cache size: N tokens` line, which vLLM
  misreports for hybrid models like this one.
- `MAX_NUM_SEQS` defaults to 16 on both, and SoSim's own OASIS concurrency is 30.
  Independent reports put practical GB10 concurrency at 5-10 before latency degrades
  badly, and NVIDIA's own vLLM playbook uses `--max-num-seqs 4`. The L40S has ~3x the
  bandwidth and may take more. Either way, sweep 1/4/8/16/32 and measure p95 latency
  before trusting a number - recreating `sosim-llm` for each value, and re-deriving the
  three concurrency keys in `.env` (see the wipe section) for the one you keep.
- Start with **few agents and few rounds**. Every round is many LLM calls per agent;
  even a paid API gets expensive past 40 rounds.
- A **Twitter** simulation loads `Twitter/twhin-bert-base` (~1GB) for its recommender;
  `setup` pre-caches it. **Reddit** needs no model at all, so Reddit-only runs are the
  lighter path.
- SoSim's own Python process runs torch on the **CPU** on both runtimes - on aarch64
  because PyPI has no CUDA wheels, on the L40S because the script hides the GPU from it.
  That is fine - it only uses torch for the recommender model. The GPU is for the vLLM
  servers.

### Testing without a GPU

All three suites run on a laptop, no GPU, no network, no LLM:

```bash
./scripts/provision_local.sh test
```

The first suite, [`scripts/tests/test_runtimes.sh`](scripts/tests/test_runtimes.sh),
runs the provisioning script itself against stubbed `docker`, `nvidia-smi` and `curl`,
for both runtimes. It pins the DGX Spark containers word for word, so a change made for
another runtime cannot quietly alter them, and it checks the L40S wiring (entrypoint,
start order, hidden GPU), runtime selection and the doctor checks. It needs no venv and
runs anywhere with bash 3.2 or later.

The shim's tests are worth knowing about, because they are what makes the drop-in claim
checkable:

- `test_zep_compat_contract.py` - serialises every response model and parses it with
  **zep-cloud's own** `parse_obj_as`, the exact code SoSim runs.
- `test_zep_compat_e2e.py` - drives the shim with a **real `AsyncZep` client** over an
  ASGI transport: real paths, bodies, error mapping, and the header-based pagination
  cursor.
- `test_zep_compat_store.py` - the batch invariants SoSim's reconciliation depends on
  (global `sequence_index`, cursors that must advance, restart durability).
- `test_zep_compat_integration.py` - opt-in, against a real FalkorDB:

  ```bash
  docker run -d --name falkordb-it -p 6399:6379 falkordb/falkordb:latest
  cd third_party/graphiti/server
  ZEP_COMPAT_IT=1 FALKORDB_IT_PORT=6399 uv run --extra dev pytest tests/test_zep_compat_integration.py -v
  ```

## Running from source, without the provisioning script

Useful on a developer machine where the model servers already exist somewhere else.
Point `LLM_BASE_URL` and `ZEP_BASE_URL` in `.env` at whatever is serving them.

| Tool | Version | Check |
|------|---------|-------|
| Node.js | >=20.19 or >=22.12 | `node -v` |
| Python | >=3.11, <3.13 | `python --version` |
| uv | latest | `uv --version` |

```bash
cp .env.example .env       # then edit it
npm run setup:all          # root + frontend + backend dependencies
npm run dev                # frontend on :3000, backend on :5001
```

`npm run backend` and `npm run frontend` start the halves individually.

## Known limitations

- **Not every Zep endpoint is implemented** - only the 17 SoSim actually calls.
  `third_party/graphiti/server/graph_service/zep_compat/WIRE_SPEC.md` lists the
  contract and how it was derived. Users, threads and fact-triples are not covered.
- **`graph_id` becomes a database name.** Graphiti 0.29.3 maps `group_id` onto the
  graph database name, so each SoSim graph is a separate FalkorDB database and the shim
  keeps one Graphiti instance per graph. Node and episode UUIDs are only resolvable
  inside their own graph, so the shim keeps a UUID-to-graph index for the two Zep routes
  that look up a UUID with no `graph_id`.
- **`uuid_cursor` paging is broken on FalkorDB** in graphiti-core 0.29.3: the
  `WHERE n.uuid < $uuid` clause is silently not applied, so every page comes back
  identical. The shim pages with `SKIP`/`LIMIT` instead (`zep_compat/paging.py`). That
  is stable as long as the graph is not written during a drain, which holds because
  SoSim reads only after a batch reaches a terminal status.
- **Extraction quality depends on the local model** honouring `json_schema`. If entity
  extraction comes back thin, that is the thing to tune first - try a larger model
  before blaming the pipeline.
- The report agent uses text `<tool_call>` tags rather than the tool-calling API, so it
  is the most model-agnostic part of the system.

## What this fork changed

SoSim is a fork of [MiroFish](https://github.com/666ghj/MiroFish). Upstream depends on
two hosted services. The LLM was a config change; Zep was not.

| Change | Why |
|---|---|
| **Added `third_party/graphiti`** submodule ([our fork](https://github.com/uxe-security-solutions/graphiti)) | Contains `server/graph_service/zep_compat/`, a Zep Cloud v2 API implementation backed by Graphiti - the same open-source temporal knowledge graph engine Zep Cloud is built on. Zep Community Edition is discontinued and no self-hostable Zep server exists, so this had to be written. |
| **`ZEP_BASE_URL`** in [`backend/app/utils/zep.py`](backend/app/utils/zep.py) | Upstream hard-coded `https://api.getzep.com/api/v2` and rejected any override. Unset, behaviour is unchanged; set, the SDK talks to the local shim. `ZEP_API_URL` stays rejected - the SDK honours it over an explicit `base_url`, so allowing both would make the effective endpoint ambiguous. |
| **`ZEP_INGESTION_WAIT_TIMEOUT_SECONDS`** now configurable | Was a hard-coded 600s. Cloud ingests well inside that; a local LLM extracting entities from a few hundred chunks does not, and overrunning raises `TimeoutError` while ingestion is still healthy. |
| **Frontend uses same-origin URLs** ([`frontend/src/api/index.js`](frontend/src/api/index.js)) | The default `http://localhost:5001` was resolved by the *browser*, so opening the UI from any other machine hit that machine's own port 5001 and failed. Every request path already starts with `/api`, which Vite proxies. **This is why only one port needs exposing.** |
| **Vite config hardened** ([`frontend/vite.config.js`](frontend/vite.config.js)) | `open: false` (headless boxes have no `xdg-open`), `strictPort: true` (it silently slid to 3001, leaving firewall rules pointing at nothing), `host: true`, and `allowedHosts` via env. |
| **Google Fonts removed** ([`frontend/index.html`](frontend/index.html)) | The only external request the frontend made; on an air-gapped box it stalls first paint until it times out. The font stack now falls back to whatever is installed, and `setup` installs `fonts-inter` and `fonts-jetbrains-mono` to match the design tokens. |
| **English-only** | The Chinese locale, the runtime language switcher and the backend locale layer are gone; the UI and every LLM instruction are compiled-in English. |
| **Added [`backend/conftest.py`](backend/conftest.py)** | `pytest` worked only via `python -m pytest`; now either form does. |
| **Added [`scripts/provision_local.sh`](scripts/provision_local.sh)** | One script for the whole stack. |
| **Added runtime profiles** ([`scripts/runtimes/`](scripts/runtimes/), `provision_dgx_spark.sh`, `provision_l40s.sh`) | The same stack on a DGX Spark and on an L40S 48GB server, with everything hardware-specific in one file per runtime. |

## Acknowledgments

SoSim's simulation engine is powered by
**[OASIS (Open Agent Social Interaction Simulations)](https://github.com/camel-ai/oasis)**.
Thanks to the CAMEL-AI team for their open-source work, and to the upstream
[MiroFish](https://github.com/666ghj/MiroFish) project this fork is built on.

## License

AGPL-3.0. See [LICENSE](./LICENSE).
