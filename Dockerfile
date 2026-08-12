# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS base

ARG SOURCE_COMMIT=unknown
ARG SOURCE_BRANCH=unknown
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SURFACE_SOURCE_COMMIT=${SOURCE_COMMIT} \
    SURFACE_SOURCE_BRANCH=${SOURCE_BRANCH}

WORKDIR /workspace
RUN groupadd --system surface && useradd --system --gid surface --create-home surface

COPY pyproject.toml requirements.txt ./
COPY surface_perception ./surface_perception
COPY configs ./configs
RUN python -m pip install --no-cache-dir . && \
    mkdir -p /workspace/runs /workspace/inputs && \
    chown -R surface:surface /workspace

FROM base AS runtime
USER surface
CMD ["surface-perception", "run-all", "--workspace", "/workspace/runs/mvp", "--config", "configs/mvp.json"]

FROM base AS ml
RUN python -m pip install --no-cache-dir ".[ml]"
USER surface
CMD ["python", "-m", "surface_perception.train_real", "--help"]

FROM base AS ci
COPY tests ./tests
COPY scripts ./scripts
RUN chown -R surface:surface /workspace
USER surface
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
