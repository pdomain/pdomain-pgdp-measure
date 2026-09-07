"""The CLI exposes exactly the five measurement subcommands."""

from __future__ import annotations

import pytest

from pdomain_pgdp_measure.cli import build_parser


def test_parser_exposes_the_five_subcommands() -> None:
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert len(actions) == 1
    assert set(actions[0].choices) == {
        "rank",
        "profile",
        "align",
        "typography",
        "glyphs",
    }


@pytest.mark.parametrize(
    ("command", "required"),
    [
        ("profile", ["--ranking", "--output"]),
        ("align", ["--profile", "--output"]),
        ("typography", ["--alignment", "--profile", "--output"]),
        ("glyphs", ["--alignment", "--profile", "--output"]),
    ],
)
def test_required_flags_survive_the_move(command: str, required: list[str]) -> None:
    parser = build_parser()
    sub = parser._subparsers._group_actions[0].choices[command]
    flags = {opt for action in sub._actions for opt in action.option_strings}
    for flag in required:
        assert flag in flags
