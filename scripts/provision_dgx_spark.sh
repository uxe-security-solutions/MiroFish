#!/usr/bin/env bash
#
# provision_dgx_spark.sh — provision_local.sh on the NVIDIA DGX Spark runtime
# (GB10, aarch64, 128GB unified memory; scripts/runtimes/dgx-spark.sh).
#
#   ./scripts/provision_dgx_spark.sh setup | start | all | status | logs [svc] | stop | doctor | test
#
# Every command and flag is provision_local.sh's; this only pins the runtime.
# dgx-spark is also what provision_local.sh uses on a host with no runtime
# recorded, so the bare script keeps working on a DGX Spark exactly as before.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOSIM_ENTRYPOINT="$0" exec "$here/provision_local.sh" --runtime dgx-spark "$@"
