# belsim: API + dashboard in one container (public demo).
#
#   docker build --platform linux/amd64 -t belsim .
#   docker run --rm -p 8000:8000 belsim
#
# data/raw (download cache of the fetch scripts) is left out by .dockerignore;
# the model runs from data/base and params/.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # numpy/scipy would otherwise start one thread per host core
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    # visitors share the server: read-only, bounded requests, security headers (docs/DEPLOY.md)
    BELSIM_PUBLIC=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY belsim ./belsim
COPY params ./params
COPY data ./data
COPY scenarios ./scenarios
COPY web ./web
RUN uv sync --frozen --no-dev

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "belsim.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--no-server-header"]
