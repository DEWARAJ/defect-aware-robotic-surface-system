FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
COPY pyproject.toml requirements.txt ./
COPY surface_perception ./surface_perception
COPY configs ./configs
RUN python -m pip install --no-cache-dir .

CMD ["surface-perception", "run-all", "--workspace", "/workspace/runs/mvp", "--config", "configs/mvp.json"]

