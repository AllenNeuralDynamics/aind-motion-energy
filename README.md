# aind-motion-energy

Motion energy computation from behavior videos for neuroscience pipelines. Targets Code Ocean deployment.

## Local development

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
uv sync --dev
uv run pytest
```

## Adding dependencies

```bash
uv add <package>
uv export --no-dev --no-emit-project --format requirements-txt > environment/requirements.txt
```

The second command regenerates `environment/requirements.txt` from the lock file — this is what the Code Ocean Dockerfile installs. Always run both together.
