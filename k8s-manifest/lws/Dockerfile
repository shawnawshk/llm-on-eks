FROM docker.io/lmsysorg/sglang:v0.5.7

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        git \
        autoconf \
        automake \
        libtool \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Install libfabric with EFA support
RUN cd /tmp && \
    git clone --depth 1 --branch v1.22.0 https://github.com/ofiwg/libfabric.git && \
    cd libfabric && \
    ./autogen.sh && \
    ./configure --prefix=/opt/amazon/efa --enable-efa=yes && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/libfabric

# Set EFA environment variables
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/efa/bin:$PATH

# Install AWS OFI NCCL plugin
RUN cd /tmp && \
    git clone --depth 1 --branch v1.11.0-aws https://github.com/aws/aws-ofi-nccl.git && \
    cd aws-ofi-nccl && \
    ./autogen.sh && \
    ./configure --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/aws-ofi-nccl

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
