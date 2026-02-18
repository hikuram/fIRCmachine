FROM nvidia/cuda:13.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_ARCH_LIST="100;120"

# Install system dependencies (pyscf)
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    wget \
    curl \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Clone gpu4pyscf v1.5.2
WORKDIR /opt
RUN git clone --depth 1 https://github.com/pyscf/gpu4pyscf.git

# Build gpu4pyscf with cmake
WORKDIR /opt/gpu4pyscf
RUN cmake -B build -S gpu4pyscf/lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCHITECTURES="${CUDA_ARCH_LIST}" \
    -DBUILD_LIBXC=ON

RUN cd build && make -j4

# Set environment variables for gpu4pyscf
ENV PYTHONPATH="/opt/gpu4pyscf:${PYTHONPATH}"

# Install Python dependencies (pyscf)
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install numpy scipy h5py pyscf cupy-cuda13x cutensor-cu13 basis-set-exchange
RUN pip3 install pyscf-dispersion geometric
#RUN pip3 install gpu4pyscf-libxc-cuda13x gpu4pyscf-cuda13x

# Install additional apt packages (others)
RUN apt-get update && apt-get install -y \
    python3-tk \
    pkg-config \
    cython3 \
#    python3-numpy \
    coinor-libipopt1v5 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (others)
RUN pip3 install pandas morfeus-ml ase rmsd sella orb-models cyipopt pydmf seaborn
RUN git clone --depth 1 https://github.com/hikuram/redox_benchmark.git
RUN pip3 install --no-deps -e redox_benchmark

# alpb may require tblite=0.5.0
# workaround about broken wheel issue (https://github.com/tblite/tblite/pull/300#issuecomment-3706915463)
ENV LDFLAGS="-Wl,--no-as-needed -lmvec -lm -lgfortran"
RUN pip3 install --no-binary=tblite --no-cache-dir tblite

RUN git clone --depth 1 https://github.com/hikuram/fIRCmachine.git　/opt/fIRCmachine
ENV PYTHONPATH="/opt/fIRCmachine/fIRCmachine:${PYTHONPATH}"

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
