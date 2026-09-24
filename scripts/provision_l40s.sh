#!/usr/bin/env bash
#
# provision_l40s.sh — provision_local.sh on the NVIDIA L40S runtime
# (Ada Lovelace sm_89, 48GB, x86_64 — including the L40S-48C vGPU;
# scripts/runtimes/l40s.sh).
#
#   ./scripts/provision_l40s.sh setup | start | all | status | logs [svc] | stop | doctor | test
#
# Every command and flag is provision_local.sh's; this only pins the runtime.
# `setup` records it in .env, after which the bare provision_local.sh picks it
# up too.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOSIM_ENTRYPOINT="$0" exec "$here/provision_local.sh" --runtime l40s "$@"
