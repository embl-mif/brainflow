# GPU-ready, no-GUI brainflow: Python API + `brainflow` command line (HPC clusters).
# The torch wheels from PyPI bring their own CUDA libraries, so a plain Python image is enough:
# the GPU only needs the host's NVIDIA driver (`docker run --gpus all`, `apptainer exec --nv`).
FROM python:3.12-slim

# libgomp1: OpenMP for torch and niftyreg (brainreg)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# the version normally comes from the git tag, but .git is not copied in: CI passes it here
ARG VERSION=0.0.0
# torch + CUDA wheels are large: give slow downloads time instead of failing
ARG PIP_DEFAULT_TIMEOUT=300

COPY . /src
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} pip install --no-cache-dir /src \
    && rm -rf /src

ENV KERAS_BACKEND=torch \
    MPLBACKEND=Agg

ENTRYPOINT ["brainflow"]
CMD ["--help"]
