#!/usr/bin/env bash
set -euo pipefail

CPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLCHAINS_DIR="${CPP_DIR}/toolchains"
GCC_URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
GCC_DIR="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"

if [[ ! -x "${TOOLCHAINS_DIR}/${GCC_DIR}/bin/aarch64-none-linux-gnu-g++" ]]; then
  mkdir -p "${TOOLCHAINS_DIR}"
  cd "${TOOLCHAINS_DIR}"
  wget -q --show-progress "${GCC_URL}" -O gcc.tar.xz
  tar -xf gcc.tar.xz
  rm -f gcc.tar.xz
fi

if [[ ! -f "${TOOLCHAINS_DIR}/ax650n_bsp_sdk/msp/out/include/ax_engine_api.h" ]]; then
  mkdir -p "${TOOLCHAINS_DIR}"
  cd "${TOOLCHAINS_DIR}"
  git clone https://github.com/AXERA-TECH/ax650n_bsp_sdk.git --depth=1
fi
