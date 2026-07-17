# Maintaining the documentation

## Local build

Install the documentation dependency group and run a warning-strict build:

```bash
poetry install --with docs
poetry run sphinx-build -W --keep-going -b html docs docs/_build/html
```

The generated site is written to `docs/_build/html`. Use `-E -a` when a full
environment rebuild is needed.

## Merge protection

The GitHub Actions workflow runs a warning-strict Sphinx build for every pull
request targeting `develop` or `main`.

## Dependency updates

Keep the Sphinx, Furo, and MyST Parser constraints in `pyproject.toml` and
`docs/requirements.txt` aligned. After changing them, update `poetry.lock` and
run both a full local documentation build and the package validation suite.
