# Contributing to bmd-check

bmd-check is a research and training project. Focused contributions from
graduate students are welcome.

## Development setup

bmd-check requires Python 3.12 or newer.

```bash
git clone https://github.com/bmd-lab/bmd-check.git
cd bmd-check
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

On Windows PowerShell, activate the environment with
`.\.venv\Scripts\Activate.ps1`.

## Make a contribution

1. Update your local `main` branch.
2. Create a focused feature branch.
3. Make the smallest change that solves the problem.
4. Add or update tests for changed behavior.
5. Run `python -m pytest` and relevant focused tests.
6. Commit with a clear message.
7. Push the branch and open a pull request.

In the pull request, explain what changed, why it changed, how it was tested,
and any remaining limitations. Keep scientific observation, software testing,
human scientific review, and BMD methodology adoption distinct.

## Safety expectations

bmd-check is an observation and diagnosis layer. Contributions must preserve
its controlled, read-only interfaces unless a separately reviewed project
explicitly introduces an authorized action boundary.

Never commit:

- API keys, tokens, passwords, or other credentials;
- SSH private keys;
- `POTCAR` files or licensed VASP potential contents;
- deployment-local `resources.toml`;
- private or unpublished research data unless its release is approved; or
- calculation data containing sensitive or private material.

The tracked `config/resources.example.toml` must contain examples only. Use
synthetic, minimal test fixtures and never add POTCAR contents to tests.

If credentials are committed accidentally, report the incident immediately to
the repository maintainers. Deleting the secret in a later commit is not
sufficient because it remains in Git history and may already have been copied.

## Review scope

Keep pull requests narrow. Do not mix scientific-policy changes with unrelated
refactoring. Changes that affect scientific interpretation should state their
evidence, assumptions, validation status, and limitations explicitly.

By contributing, you agree that repository-owned contributions are provided
under the project's MIT License. VASP, POTCAR/PAW datasets, and third-party
dependencies remain governed by their own licenses and access terms.
