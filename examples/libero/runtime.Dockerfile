FROM umi-libero-runtime-base
# The evaluator verifies the mounted source revisions before running.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
