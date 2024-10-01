# Use an official micromamba image as the base image
ARG PYTHON_VERSION=3.10

FROM mambaorg/micromamba:1.5.10


# Set environment variables for micromamba
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_ROOT_PREFIX=/opt/conda

# Switch to root to install all dependencies (using non-root user causes permission issues)
USER root

# Make arg accessible to the rest of the Dockerfile
ARG PYTHON_VERSION

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    bash \
    bc \
    unzip \
    wget \
    gfortran \
    liblapack-dev \
    libblas-dev \
    cmake \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python
RUN micromamba install -y -n base -c conda-forge \ python=${PYTHON_VERSION} && \
    micromamba clean --all --yes

# Install testing dependencies
RUN python -m pip install --upgrade pip \
    && pip install uv \
    && uv pip install flake8 pre-commit pytest pytest-mock pytest-split pytest-cov types-setuptools

# Install Julia
RUN curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.2-linux-x86_64.tar.gz | tar -xz -C /opt \
    && ln -s /opt/julia-1.9.2/bin/julia /usr/local/bin/julia


# Set up Julia environment (ACEpotentials.jl interface)
RUN julia -e 'using Pkg; Pkg.Registry.add("General"); Pkg.Registry.add(Pkg.Registry.RegistrySpec(url="https://github.com/ACEsuit/ACEregistry")); Pkg.add("ACEpotentials"); Pkg.add("DataFrames"); Pkg.add("CSV")'

# Install Buildcell (airss)
RUN curl -fsSL https://www.mtg.msm.cam.ac.uk/files/airss-0.9.3.tgz -o /opt/airss-0.9.3.tgz \
    && tar -xf /opt/airss-0.9.3.tgz -C /opt \
    && rm /opt/airss-0.9.3.tgz \
    && cd /opt/airss \
    && make \
    && make install \
    && make neat

# Add Buildcell to PATH
ENV PATH="${PATH}:/opt/airss/bin"
