# syntax=docker/dockerfile:1.7
ARG UBUNTU=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
FROM --platform=linux/arm64 ${UBUNTU} AS toolchain
ENV DEBIAN_FRONTEND=noninteractive HOME=/work LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends g++ make cmake ninja-build python3 xz-utils file git ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY sidecar-shim/cuda-redist-13.4.sha256 /tmp/cuda-redist-13.4.sha256
RUN set -eux; mkdir -p /work/cuda-13.4 /tmp/dl; cd /tmp/dl; \
    while read -r sha path; do curl -fsSL --retry 3 -o "$(basename "$path")" "https://developer.download.nvidia.com/compute/cuda/redist/$path"; echo "$sha  $(basename "$path")" | sha256sum -c -; done < /tmp/cuda-redist-13.4.sha256; \
    for f in *.tar.xz; do tar -xJf "$f"; done; \
    for d in */; do cp -a "$d/." /work/cuda-13.4/; done; \
    ln -s lib /work/cuda-13.4/lib64; touch /work/cuda-13.4/lib/libcublas.so /work/cuda-13.4/lib/libcublasLt.so; rm -rf /tmp/dl
COPY sidecar-shim/include/cublas_v2.h /work/cuda-13.4/include/cublas_v2.h
ENV PATH=/work/cuda-13.4/bin:${PATH}
FROM toolchain AS engine-configure
COPY . /work/strata/
RUN mkdir -p /work/shim && cp -a /work/strata/sidecar-shim/. /work/shim/
RUN cmake -S /work/strata -B /work/strata-linux -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/work/cuda-13.4/bin/nvcc -DCUDAToolkit_ROOT=/work/cuda-13.4 -DSTRATA_ISA_FLOOR=none -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    && (cd /work/strata-linux && ninja -t commands strata > strata_cmds.txt) \
    && SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk python3 /work/shim/macbuild.py
