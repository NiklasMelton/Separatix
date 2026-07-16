# Contributing to separatix

Contributions are welcome. Changes should preserve the project's priorities:
transparent recommendations, conservative escalation, deterministic behavior,
memory-aware diagnostics, and graceful handling of unavailable evidence.

## Development setup

Fork and clone the repository, then install the package and development groups:

```bash
poetry install --with dev,docs
```

Create a focused branch from `develop`. Keep required dependencies limited to
the scientific Python stack unless a feature can be provided as an optional
extra.

## Validation

Run the required checks before opening a pull request:

```bash
poetry run pytest
poetry run ruff check separatix tests
poetry run ruff format --check separatix tests
poetry run mypy separatix
poetry run sphinx-build -W --keep-going -b html docs docs/_build/html
```

Warnings fail the documentation build. Add or update tests for behavioral
changes, preferring qualitative assertions and stable decision-path or schema
checks over exact floating-point values.

## Documentation

Update user guides when public behavior changes and public docstrings when an
API signature or contract changes. The Read the Docs site builds the repository
README, project documents, runnable example sources, and public API docstrings,
so broken references or invalid source documentation fail CI.

## Pull requests and required checks

Pull requests should normally target `develop`; release integration can target
`main`. Repository administrators require passing statuses on **Documentation / 
Sphinx build** checks in
addition to the regular test and style checks. The workflow runs on every pull
request targeting either protected branch.

Keep pull requests narrow and describe user-visible behavior, validation
performed, optional-dependency effects, and any skipped checks.
