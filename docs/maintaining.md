# Maintaining the documentation

## Local build

Install the documentation dependency group and run a warning-strict build:

```bash
poetry install --with docs
poetry run sphinx-build -W --keep-going -b html docs docs/_build/html
```

The generated site is written to `docs/_build/html`. Use `-E -a` when a full
environment rebuild is needed.

## Read the Docs project setup

The repository-level `.readthedocs.yaml` uses the current version 2 schema,
Python 3.12 on Ubuntu 24.04, installs both the package and documentation
requirements, and fails the hosted build on every Sphinx warning.

To connect the hosted service:

1. [Import the GitHub repository into Read the Docs](https://docs.readthedocs.com/platform/stable/tutorial/index.html#importing-the-project-to-read-the-docs).
2. Set `develop` as the default documentation version if development docs should
   lead the site; retain `main` for release-only documentation instead.
3. Enable [pull request builds](https://docs.readthedocs.com/platform/stable/guides/pull-requests.html)
   to receive hosted previews and their Read the Docs status.
4. Trigger the first build and confirm the generated API reference imports the
   installed `separatix` package.

Read the Docs uses the root configuration file automatically after import. The
hosted service account and project slug are intentionally not stored in the
repository.

## Merge protection

The GitHub Actions workflow runs a warning-strict Sphinx build for every pull
request targeting `develop` or `main`. In the repository rules for both
branches, require the check named **Documentation / Sphinx build**. GitHub branch
rules are service-side settings and cannot be enforced by a workflow file alone.

The separate Read the Docs pull-request status can also be required after hosted
preview builds are enabled, but the GitHub Actions check remains the fast,
repository-controlled merge gate.

## Dependency updates

Keep the Sphinx, Furo, and MyST Parser constraints in `pyproject.toml` and
`docs/requirements.txt` aligned. After changing them, update `poetry.lock` and
run both a full local documentation build and the package validation suite.
