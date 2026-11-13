#!/usr/bin/env bash
set -eu

RELEASE_REPO="pdomain/pdomain-pgdp-measure"
# just ci runs this repo's full gate.
RELEASE_PREFLIGHT="just ci"

. "$(dirname "$0")/release-common.sh"
pdomain_release_main "$@"
