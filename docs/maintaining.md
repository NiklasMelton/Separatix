# Maintaining the documentation

## Local build

Install the documentation dependency group and run a warning-strict build:

```bash
poetry install --with docs
poetry run sphinx-build -W --keep-going -b html docs docs/_build/html
```

The generated site is written to `docs/_build/html`. Use `-E -a` when a full
environment rebuild is needed.

Every authored Markdown or reStructuredText page under `docs/` must be included
in a toctree. Because warnings fail the build, Sphinx's orphan-document warning
enforces this requirement. `docs/readme.md` includes the repository README in
full, while the contributing guide, license, runnable examples, and public API
docstrings are also included directly from their canonical sources. This keeps
the Read the Docs site complete without maintaining duplicate reference text.

## Merge protection

The GitHub Actions workflow runs a warning-strict Sphinx build for every pull
request targeting `develop` or `main`.

## Dependency updates

Keep the Sphinx, Furo, and MyST Parser constraints in `pyproject.toml` and
`docs/requirements.txt` aligned. After changing them, update `poetry.lock` and
run both a full local documentation build and the package validation suite.
