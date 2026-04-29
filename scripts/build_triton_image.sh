#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRITON_CONTAINER_VERSION="${TRITON_CONTAINER_VERSION:-24.04}"
TRITON_VERSION="${TRITON_VERSION:-2.45.0}"
UPSTREAM_CONTAINER_VERSION="${UPSTREAM_CONTAINER_VERSION:-24.04}"
DCGM_VERSION="${DCGM_VERSION:-3.2.6}"
VLLM_VERSION="${VLLM_VERSION:-0.4.0.post1}"
TRITON_SERVER_REPO="${TRITON_SERVER_REPO:-https://github.com/triton-inference-server/server.git}"
TRITON_SERVER_DIR="${TRITON_SERVER_DIR:-${ROOT_DIR}/server}"
TRITON_SERVER_REF="${TRITON_SERVER_REF:-r${TRITON_CONTAINER_VERSION}}"
BASE_IMAGE="${BASE_IMAGE:-triton-base:${TRITON_CONTAINER_VERSION}}"
BUILD_DIR="${BUILD_DIR:-${TRITON_SERVER_DIR}/build}"

ensure_triton_server_sources() {
  if [[ ! -e "${TRITON_SERVER_DIR}" ]]; then
    echo "Cloning Triton server sources: ${TRITON_SERVER_REPO} (${TRITON_SERVER_REF})"
    git clone --branch "${TRITON_SERVER_REF}" --single-branch "${TRITON_SERVER_REPO}" "${TRITON_SERVER_DIR}"
  elif [[ ! -d "${TRITON_SERVER_DIR}/.git" ]]; then
    cat >&2 <<EOF
${TRITON_SERVER_DIR} already exists but is not a git checkout.
Remove it or set TRITON_SERVER_DIR to another path.
EOF
    exit 1
  else
    echo "Using existing Triton server sources: ${TRITON_SERVER_DIR}"
  fi

  if [[ ! -f "${TRITON_SERVER_DIR}/TRITON_VERSION" ]]; then
    cat >&2 <<EOF
${TRITON_SERVER_DIR}/TRITON_VERSION not found.
Check that TRITON_SERVER_DIR points to a Triton server checkout.
EOF
    exit 1
  fi

  SERVER_VERSION="$(tr -d '[:space:]' < "${TRITON_SERVER_DIR}/TRITON_VERSION")"
  if [[ "${ALLOW_TRITON_VERSION_MISMATCH:-0}" != "1" && "${SERVER_VERSION}" != "${TRITON_VERSION}" ]]; then
    cat >&2 <<EOF
${TRITON_SERVER_DIR}/TRITON_VERSION is ${SERVER_VERSION}, but this script is configured for ${TRITON_VERSION}.
Expected source ref is ${TRITON_SERVER_REF}.

If this is an existing checkout, switch it manually, for example:
  git -C "${TRITON_SERVER_DIR}" fetch --tags origin
  git -C "${TRITON_SERVER_DIR}" checkout "${TRITON_SERVER_REF}"

Set ALLOW_TRITON_VERSION_MISMATCH=1 only if you intentionally want a mixed source/dependency build.
EOF
    exit 1
  fi
}

ensure_triton_server_sources

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${TRITON_SERVER_DIR}/.venv/bin/python" ]]; then
    PYTHON="${TRITON_SERVER_DIR}/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

patch_generated_dockerfiles() {
  BUILD_DIR="${BUILD_DIR}" DCGM_VERSION="${DCGM_VERSION}" VLLM_VERSION="${VLLM_VERSION}" TRITON_CONTAINER_VERSION="${TRITON_CONTAINER_VERSION}" python3 - <<'PY'
import os
import re
import base64
from pathlib import Path

build_dir = Path(os.environ["BUILD_DIR"])
dcgm_version = os.environ["DCGM_VERSION"]
vllm_version = os.environ["VLLM_VERSION"]
triton_container_version = os.environ["TRITON_CONTAINER_VERSION"]

buildbase = build_dir / "Dockerfile.buildbase"
cmake_build = build_dir / "cmake_build"
runtime = build_dir / "Dockerfile"

docker_install = """# Install docker docker buildx
RUN apt-get update \\
      && apt-get install -y ca-certificates curl gnupg \\
      && install -m 0755 -d /etc/apt/keyrings \\
      && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \\
      && chmod a+r /etc/apt/keyrings/docker.gpg \\
      && echo \\
          "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \\
          "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \\
          tee /etc/apt/sources.list.d/docker.list > /dev/null \\
      && apt-get update \\
      && apt-get install -y docker.io docker-buildx-plugin
"""

astra_docker_install = """# Install Docker CLI from Astra repositories.
RUN apt-get update \\
      && apt-get install -y --no-install-recommends docker.io \\
      && rm -rf /var/lib/apt/lists/*
"""

dcgm_block = f"""ENV DCGM_VERSION {dcgm_version}
# DCGM 3.x is the public apt package line matching Triton 24.04.
RUN apt-get update -qq \\
      && apt-get install --yes --no-install-recommends \\
                   datacenter-gpu-manager=1:{dcgm_version} \\
      && rm -rf /var/lib/apt/lists/*
"""

third_party_patch_py = r'''
from pathlib import Path

path = Path("/opt/triton-third-party/CMakeLists.txt")
text = path.read_text()

libevent_prefix = "    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/libevent\n"
libevent_options = (
    "    -DEVENT__DISABLE_SAMPLES:BOOL=ON\n"
    "    -DEVENT__DISABLE_BENCHMARK:BOOL=ON\n"
    "    -DEVENT__DISABLE_TESTS:BOOL=ON\n"
    "    -DEVENT__DISABLE_REGRESS:BOOL=ON\n"
)
if libevent_options not in text:
    if libevent_prefix not in text:
        raise SystemExit("libevent CMake cache anchor not found")
    text = text.replace(libevent_prefix, libevent_prefix + libevent_options, 1)

old_patch = """  PATCH_COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG}
)
#
# Build patched libevhtp
#"""

patch_script = Path("/opt/triton-third-party/patch_libevent.py")
patch_script.write_text(r"""from pathlib import Path
import sys

p = Path(sys.argv[1]) / "evutil_rand.c"
text = p.read_text()
old = 'void\nevutil_secure_rng_add_bytes(const char *buf, size_t n)\n{\n\tarc4random_addrandom((unsigned char*)buf,\n\t    n>(size_t)INT_MAX ? INT_MAX : (int)n);\n}\n'
new = 'void\nevutil_secure_rng_add_bytes(const char *buf, size_t n)\n{\n\t(void)buf;\n\t(void)n;\n}\n'
if old not in text:
    raise SystemExit("evutil_secure_rng_add_bytes anchor not found")
p.write_text(text.replace(old, new, 1))
""")

new_patch = """  PATCH_COMMAND
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/patch_libevent.py <SOURCE_DIR>
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG}
)
#
# Build patched libevhtp
#"""

if old_patch not in text:
    raise SystemExit("libevent PATCH_COMMAND anchor not found")
text = text.replace(old_patch, new_patch, 1)
path.write_text(text)
'''

third_party_patch_b64 = base64.b64encode(third_party_patch_py.encode()).decode()
third_party_overlay = f"""# Patch Triton third_party recipes for Astra/glibc compatibility.
RUN git clone --depth=1 --single-branch -b r{triton_container_version} \\
          https://github.com/triton-inference-server/third_party.git \\
          /opt/triton-third-party \\
      && python3 -c 'import base64; exec(base64.b64decode("{third_party_patch_b64}").decode())'
"""

compat_re = re.compile(
    r"\n# Extra defensive wiring for CUDA Compat lib\n"
    r"RUN ln -sf \$\{_CUDA_COMPAT_PATH\}/lib\.real \$\{_CUDA_COMPAT_PATH\}/lib \\\n"
    r"\s+&& echo \$\{_CUDA_COMPAT_PATH\}/lib > /etc/ld\.so\.conf\.d/00-cuda-compat\.conf \\\n"
    r"\s+&& ldconfig \\\n"
    r"\s+&& rm -f \$\{_CUDA_COMPAT_PATH\}/lib\n",
)

dcgm_re = re.compile(
    r"ENV DCGM_VERSION .+?\n"
    r"# Install DCGM\. Steps from https://developer\.nvidia\.com/dcgm#Downloads\n"
    r"RUN .*?(?=\n\n)",
    re.S,
)

kitware_re = re.compile(
    r"# Server build requires recent version of CMake \(FetchContent required\)\n"
    r"RUN apt update -q=2 \\\n"
    r"      && apt install -y gpg wget \\\n"
    r"      && wget -O - https://apt\.kitware\.com/keys/kitware-archive-latest\.asc 2>/dev/null \| gpg --dearmor - \|  tee /usr/share/keyrings/kitware-archive-keyring\.gpg >/dev/null \\\n"
    r"      && \. /etc/os-release \\\n"
    r"      && echo \"deb \[signed-by=/usr/share/keyrings/kitware-archive-keyring\.gpg\] https://apt\.kitware\.com/ubuntu/ \$UBUNTU_CODENAME main\" \| tee /etc/apt/sources\.list\.d/kitware\.list >/dev/null \\\n"
    r"      && apt-get update -q=2 \\\n"
    r"      && apt-get install -y --no-install-recommends cmake=3\.27\.7\* cmake-data=3\.27\.7\*",
)

astra_apt_tls_config = """# Astra repo OCSP responses can be rejected by apt inside intermediate containers.
RUN printf '%s\\n' 'Acquire::https::download.astralinux.ru::Verify-Peer "false";' \\
      > /etc/apt/apt.conf.d/99-astra-repo-tls
"""

text = buildbase.read_text()
if docker_install not in text:
    raise SystemExit("Could not find Docker apt install block in Dockerfile.buildbase")
text = text.replace(docker_install, astra_docker_install)
text = kitware_re.sub(
    "# Server build requires recent version of CMake (FetchContent required)\n"
    "RUN pip3 install --upgrade cmake==3.27.7",
    text,
)
text = dcgm_re.sub(dcgm_block, text)
if third_party_overlay not in text:
    text = text.replace("COPY . .\n", f"COPY . .\n{third_party_overlay}", 1)
buildbase.write_text(text)

text = cmake_build.read_text()
third_party_arg = '"-DFETCHCONTENT_SOURCE_DIR_REPO-THIRD-PARTY:PATH=/opt/triton-third-party"'
if third_party_arg not in text:
    text = text.replace(
        '"-DTRITON_THIRD_PARTY_REPO_TAG:STRING=r24.04" ',
        f'"-DTRITON_THIRD_PARTY_REPO_TAG:STRING=r24.04" {third_party_arg} ',
        1,
    )
python_backend_cxx_flags = '"-DCMAKE_CXX_FLAGS:STRING=-Wno-error=deprecated-declarations"'
if python_backend_cxx_flags not in text:
    text = text.replace(
        '"-DTRITON_ENABLE_MEMORY_TRACKER:BOOL=ON" ..',
        f'"-DTRITON_ENABLE_MEMORY_TRACKER:BOOL=ON" {python_backend_cxx_flags} ..',
        1,
    )
cmake_build.write_text(text)

text = runtime.read_text()
if "99-astra-repo-tls" not in text:
    text = text.replace("FROM ${BASE_IMAGE}\n", f"FROM ${{BASE_IMAGE}}\n\n{astra_apt_tls_config}", 1)
text = dcgm_re.sub(dcgm_block, text)
text = compat_re.sub("\n", text)
text = text.replace("pip3 install --upgrade \\\n            wheel \\\n            setuptools \\\n            numpy \\\n            virtualenv", "pip3 install --upgrade \\\n            wheel \\\n            setuptools \\\n            'numpy<2' \\\n            virtualenv")
text = text.replace(
    "pip3 install vllm==0.11.1",
    "pip3 install \\\n"
    "      'numpy<2' \\\n"
    "      'protobuf<5' \\\n"
    "      'huggingface-hub<1' \\\n"
    "      'tokenizers<0.20' \\\n"
    "      'transformers==4.39.3' \\\n"
    f"      vllm=={vllm_version}",
)
text = text.replace("ARG PYVER=3.12", "ARG PYVER=3.11")
runtime.write_text(text)
PY
}

echo "Building base image: ${BASE_IMAGE}"
docker build -t "${BASE_IMAGE}" -f "${ROOT_DIR}/Dockerfile.base" "${ROOT_DIR}"

echo "Building Triton image with metrics + vLLM backend"
(
  cd "${TRITON_SERVER_DIR}"
  "${PYTHON}" ./build.py \
    --dryrun \
    --no-container-pull \
    --no-container-interactive \
    --version "${TRITON_VERSION}" \
    --container-version "${TRITON_CONTAINER_VERSION}" \
    --upstream-container-version "${UPSTREAM_CONTAINER_VERSION}" \
    --image "base,${BASE_IMAGE}" \
    --target-platform linux \
    --target-machine x86_64 \
    --backend python \
    --backend vllm \
    --backend ensemble \
    --endpoint http \
    --endpoint grpc \
    --enable-logging \
    --enable-stats \
    --enable-metrics \
    --enable-gpu-metrics \
    --enable-cpu-metrics \
    --enable-gpu
)

patch_generated_dockerfiles
"${BUILD_DIR}/docker_build"
