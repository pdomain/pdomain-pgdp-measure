.DEFAULT_GOAL := ci

.PHONY: setup install-hooks test lint lint-check typecheck format build ci

setup: ## Install dependencies (idempotent)
	uv sync --all-groups
	@$(MAKE) --no-print-directory install-hooks

install-hooks: ## (Re)install pre-commit hooks (repairs a stale interpreter path)
	@# `pre-commit install` bakes an absolute interpreter path into .git/hooks.
	@# A hook written against a different environment name, or against a worktree
	@# that has since been deleted, keeps failing until it is rewritten — and a
	@# "skip if the file exists" guard never rewrites it. Rewriting costs ~0.2s,
	@# so do it every time this repo owns its hooks directory.
	@if [ -f .git ]; then \
	  echo "hooks: worktree checkout — the canonical repo owns them, skipping"; \
	elif [ -n "$$(git config --get core.hooksPath 2>/dev/null)" ]; then \
	  echo "hooks: core.hooksPath is set — leaving it alone, skipping"; \
	else \
	  uv run pre-commit install --hook-type pre-commit --hook-type commit-msg; \
	fi

test: ## Run the test suite
	uv run pytest

lint: ## Run ruff with fixes
	uv run ruff check --select I --fix
	uv run ruff check --fix

lint-check: ## Read-only ruff format + check (matches CI)
	uv run ruff format --check .
	uv run ruff check .

typecheck: ## Type-check the package
	@# Errors only. The tree carries informational warnings at this mode and no
	@# baseline file grandfathers them; see the note in pyproject.toml.
	uv run basedpyright src --level error

format: ## Apply ruff formatting
	uv run ruff format .

build: ## Build the distribution
	uv build

ci: lint-check typecheck test build ## Full repository gate
