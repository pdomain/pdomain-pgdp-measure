set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set positional-arguments

# Copy this wrapper and install-hooks with the shell setting to sibling justfiles.
[private]
_ai +args:
    #!/usr/bin/env bash
    set -euo pipefail
    if env AI= just --quiet "$@" > .ci-ai.log 2>&1; then
        printf 'PASS: %s\n' "$*"
    else
        result=$?
        printf 'FAIL: %s (exit %s)\n' "$*" "$result"
        if [[ -f scripts/ai_filter_log.py ]]; then
            uv run python scripts/ai_filter_log.py .ci-ai.log
        else
            tail -50 .ci-ai.log
        fi
        exit "$result"
    fi

install-hooks:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai install-hooks; fi
    if [[ -f .git ]]; then
        echo "hooks: worktree checkout - the canonical repo owns them, skipping"
    elif [[ -n "$(git config --get core.hooksPath || true)" ]]; then
        echo "hooks: core.hooksPath is set - leaving it alone, skipping"
    else
        uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
    fi

setup:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai setup; fi
    uv sync --locked --all-groups
    just --quiet install-hooks

update-hooks:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai update-hooks; fi
    uv run pre-commit autoupdate --freeze

lint:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai lint; fi
    uv run ruff check --select I --fix .
    uv run ruff check --fix .

lint-check:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai lint-check; fi
    uv run ruff format --check .
    uv run ruff check .

format:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai format; fi
    uv run ruff format .

format-check:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai format-check; fi
    uv run ruff format --check .

typecheck:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai typecheck; fi
    uv run basedpyright src --level error

test:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai test; fi
    uv run pytest -n auto

build:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai build; fi
    uv build

pre-commit-check:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai pre-commit-check; fi
    uv run pre-commit run --all-files

[default]
ci:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai ci; fi
    just --quiet setup pre-commit-check lint-check typecheck test build
    docgraph check --strict

ci-slow:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai ci-slow; fi
    just --quiet ci

upgrade-deps:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai upgrade-deps; fi
    for marker in "${UV_PROJECT_ENVIRONMENT:-.venv}/.pdomain-local-mode" "${UV_PROJECT_ENVIRONMENT:-.venv}/.pdomain-dev-local"; do
        if [[ -f "$marker" ]]; then
            echo "Refusing dependency upgrades while $marker exists" >&2
            exit 1
        fi
    done
    uv lock --upgrade
    uv sync --all-groups

release-patch:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai release-patch; fi
    BUMP=patch ./scripts/do-release.sh

release-minor:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai release-minor; fi
    BUMP=minor ./scripts/do-release.sh

release-major:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${AI:-}" ]]; then exec just --quiet _ai release-major; fi
    BUMP=major ./scripts/do-release.sh
