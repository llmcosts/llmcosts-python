# Publishing llmcosts to PyPI

These steps outline how to release a new version of the `llmcosts` package to PyPI.

1. **Update version numbers** in `pyproject.toml`, `llmcosts/tracker/__init__.py`, and `uv.lock`.
2. **Commit** the changes and push them to the `main` branch.
3. **Tag** the commit using the new version number:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
4. Make sure the repository is public.
5. **Create the `PYPI_API_TOKEN` secret** in the GitHub repository settings with a token generated from your PyPI account.
6. Once the tag is pushed, GitHub Actions will build and upload the release to PyPI automatically.

The included GitHub workflow `.github/workflows/publish.yml` handles building the wheel and uploading it to PyPI whenever a new tag is created on `main`.
