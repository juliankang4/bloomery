#!/usr/bin/env bash
# Check each PTX shape table against the kernels its binary and target declare (text only, no build).
# tools/check-ptx-names.py describes source declarations, target cfgs, filters and refusals.
set -euo pipefail
exec python3 "$(dirname "$0")/check-ptx-names.py" check "$@"
