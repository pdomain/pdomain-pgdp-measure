#!/usr/bin/env bash
set -eu

RELEASE_REPO="pdomain/pdomain-pgdp-measure"
# This repo's Makefile has no ci-slow target (unlike pdomain-book-tools);
# make ci already runs lint, typecheck, test, and build.
RELEASE_PREFLIGHT="make ci"

. "$(dirname "$0")/release-common.sh"
pdomain_release_main "$@"
