# Releases

Publishing a GitHub release (including a prerelease) triggers
[the release workflow](.github/workflows/release.yml). It builds a wheel and tests
on macOS 26, Ubuntu 24.04 x64, and Ubuntu 24.04 ARM64 with Python
3.10, 3.11, 3.12, and 3.13. Once all matrix jobs pass, separate jobs upload the
wheel to TestPyPI and attach it to the GitHub release.

## One-time setup

Before the first release, create a GitHub environment named `testpypi` and
configure a [Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
on **TestPyPI** for `antsnormflows` with the repository's owner and name,
workflow filename `release.yml`, and environment name `testpypi`. For a project
that does not exist yet, use TestPyPI's pending publisher form. No API token
secret is needed.

## Pre-release actions

Update `src/antsnormflows/_version.py` before tagging a release. The workflow
uses that version; it does not derive the package version from the release tag.
Each TestPyPI upload needs a new package version. To retry only a failed upload,
rerun failed jobs so they reuse the wheel already built and tested.