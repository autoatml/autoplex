# Use an official micromamba image as the base image
ARG PYTHON_VERSION=3.10

FROM mambaorg/micromamba:1.5.10


# Set environment variables for micromamba
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_NO_LOW_SPEED_LIMIT=1

# Switch to root to install all dependencies (using non-root user causes permission issues)
USER root

# Make arg accessible to the rest of the Dockerfile
ARG PYTHON_VERSION

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    bash \
    bc \
    ffmpeg \
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

# Install Python and clean up tarballs
RUN micromamba install -y -n base -c conda-forge \ python=${PYTHON_VERSION} && \
    micromamba clean --all --yes

# Install Julia
RUN curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.2-linux-x86_64.tar.gz | tar -xz -C /opt \
    && ln -s /opt/julia-1.9.2/bin/julia /usr/local/bin/julia

# Set up Julia environment (ACEpotentials.jl interface)
RUN julia -e 'using Pkg; Pkg.Registry.add("General"); Pkg.Registry.add(Pkg.Registry.RegistrySpec(url="https://github.com/ACEsuit/ACEregistry")); Pkg.add(Pkg.PackageSpec(;name="ACEpotentials", version="0.6.7")); Pkg.add("DataFrames"); Pkg.add("CSV")'

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

# Install LAMMPS (rss)
RUN curl -fsSL https://download.lammps.org/tars/lammps-29Aug2024_update1.tar.gz -o /opt/lammps.tar.gz \
     && tar -xf /opt/lammps.tar.gz -C /opt \
     && rm /opt/lammps.tar.gz \
     && cd /opt/lammps-* \
     && mkdir build \
     && cd build \
     && curl -fsSL https://github.com/wcwitt/lammps-user-pace/archive/main.tar.gz -o libpace.tar.gz \
     && cmake -D PKG_PYTHON=on -D BUILD_SHARED_LIBS=on -DMLIAP_ENABLE_PYTHON=yes -D PKG_KOKKOS=yes -D Kokkos_ARCH_ZEN3=yes -D PKG_PHONON=yes -D PKG_MOLECULE=yes -D PKG_MANYBODY=yes -D Kokkos_ENABLE_OPENMP=yes -D BUILD_OMP=yes -D LAMMPS_EXCEPTIONS=yes -D PKG_ML-PACE=yes -D PACELIB_MD5=$(md5sum libpace.tar.gz | awk '{print $1}') -D CMAKE_EXE_LINKER_FLAGS:STRING="-lgfortran" ../cmake \
     && cmake --build . \
     && make -j 4 install \
     && make install-python \
     && cmake --build . --target clean

# Add LAMMPS to PATH and Update LD_LIBRARY_PATH
ENV PATH="${PATH}:/root/.local/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/root/.local/lib:/opt/conda/lib"

# Set the working directory
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Install autoplex, testing dependencies and clear cache
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir uv \
    && uv pip install pre-commit pytest pytest-mock pytest-split pytest-cov types-setuptools \
    && uv pip install --prerelease=allow .[strict,docs] && uv cache clean && rm -rf /tmp/*
