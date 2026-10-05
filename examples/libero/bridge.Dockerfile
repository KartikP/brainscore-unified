FROM python:3.11-slim-bookworm
RUN python -m pip install build setuptools==68.2.2 wheel
COPY core /src/core
COPY vision /src/vision
COPY language /src/language
COPY unified /src/unified
RUN for repo in core vision language unified; do \
      python -m build --wheel --no-isolation --outdir /wheels /src/$repo; \
    done
RUN python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install -c /src/unified/install/v2-constraints.txt /wheels/*.whl \
      websockets==15.0.1 msgpack==1.1.1 && python -m pip check
ENV PYTHONPATH=/openpi/src:/openpi/packages/openpi-client/src
ENV BS_INSTALL_DEPENDENCIES=no
ENV RESULTCACHING_DISABLE=1
WORKDIR /src/unified
