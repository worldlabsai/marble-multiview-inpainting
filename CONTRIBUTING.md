# Contributing

Contributions are welcome through GitHub issues and pull requests.

## Development setup

Install `uv`, clone the repository, and run:

```bash
uv sync --frozen --all-groups
uv run pytest
uv run ruff check .
uv run pyright
```

Do not attach private customer imagery or unpublished World Labs data to an
issue, fixture, or pull request. Geometry bugs should include a minimal
synthetic or otherwise redistributable fixture.

Changes to camera transforms, mask polarity, resampling, or PNG output require
a focused regression test. User-visible behavior changes also require a README
or `docs/` update.
