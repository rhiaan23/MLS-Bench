#!/bin/bash
# Install mesa/osmesa libraries from .deb packages without apt
# Needed when apt-get fails due to lack of fakeroot privileges
set -e

ARCH=amd64
MIRROR="http://archive.ubuntu.com/ubuntu/pool"
TMPDIR=$(mktemp -d)

# Check if libosmesa is already installed
if ldconfig -p 2>/dev/null | grep -q libOSMesa; then
    echo "libOSMesa already available"
    exit 0
fi

echo "Installing mesa libraries from .deb packages..."

DEBS=(
    "main/libg/libglvnd/libglvnd0_1.4.0-1_${ARCH}.deb"
    "main/libg/libglvnd/libgl1_1.4.0-1_${ARCH}.deb"
    "main/libg/libglvnd/libglx0_1.4.0-1_${ARCH}.deb"
    "main/libg/libglvnd/libegl1_1.4.0-1_${ARCH}.deb"
    "main/libg/libglvnd/libopengl0_1.4.0-1_${ARCH}.deb"
    "main/m/mesa/libosmesa6_23.2.1-1ubuntu3.1~22.04.3_${ARCH}.deb"
    "main/m/mesa/libgl1-mesa-dri_23.2.1-1ubuntu3.1~22.04.3_${ARCH}.deb"
    "main/m/mesa/libglx-mesa0_23.2.1-1ubuntu3.1~22.04.3_${ARCH}.deb"
    "main/m/mesa/libglapi-mesa_23.2.1-1ubuntu3.1~22.04.3_${ARCH}.deb"
    "main/m/mesa/libegl-mesa0_23.2.1-1ubuntu3.1~22.04.3_${ARCH}.deb"
    "main/libx/libx11/libx11-6_1.7.5-1ubuntu0.3_${ARCH}.deb"
    "main/libx/libxcb/libxcb1_1.14-3ubuntu3_${ARCH}.deb"
    "main/libx/libxau/libxau6_1.0.9-1build5_${ARCH}.deb"
    "main/libx/libxdmcp/libxdmcp6_1.1.3-0ubuntu5_${ARCH}.deb"
    "main/libx/libxext/libxext6_1.3.4-1build1_${ARCH}.deb"
    "main/libx/libxfixes/libxfixes3_6.0.0-1_${ARCH}.deb"
    "main/libx/libxxf86vm/libxxf86vm1_1.1.4-1build3_${ARCH}.deb"
    "main/libx/libxshmfence/libxshmfence1_1.3-1build4_${ARCH}.deb"
    "main/libx/libxcb/libxcb-glx0_1.14-3ubuntu3_${ARCH}.deb"
    "main/libx/libxcb/libxcb-shm0_1.14-3ubuntu3_${ARCH}.deb"
    "main/libx/libxcb/libxcb-dri2-0_1.14-3ubuntu3_${ARCH}.deb"
    "main/libx/libxcb/libxcb-dri3-0_1.14-3ubuntu3_${ARCH}.deb"
    "main/libx/libxcb/libxcb-present0_1.14-3ubuntu3_${ARCH}.deb"
    "main/libx/libxcb/libxcb-sync1_1.14-3ubuntu3_${ARCH}.deb"
    "main/libx/libxcb/libxcb-xfixes0_1.14-3ubuntu3_${ARCH}.deb"
    "main/libx/libxcb/libxcb-randr0_1.14-3ubuntu3_${ARCH}.deb"
    "main/libd/libdrm/libdrm2_2.4.113-2~ubuntu0.22.04.1_${ARCH}.deb"
    "main/libd/libdrm/libdrm-common_2.4.113-2~ubuntu0.22.04.1_all.deb"
    "main/l/llvm-toolchain-15/libllvm15_15.0.7-0ubuntu0.22.04.3_${ARCH}.deb"
    "main/libe/libedit/libedit2_3.1-20210910-1build1_${ARCH}.deb"
    "main/i/icu/libicu70_70.1-2_${ARCH}.deb"
    "main/libx/libxml2/libxml2_2.9.13+dfsg-1ubuntu0.11_${ARCH}.deb"
    "main/libb/libbsd/libbsd0_0.11.5-1_${ARCH}.deb"
    "main/libm/libmd/libmd0_1.0.4-1build1_${ARCH}.deb"
)

cd "$TMPDIR"
for deb_path in "${DEBS[@]}"; do
    url="${MIRROR}/${deb_path}"
    fname=$(basename "$deb_path")
    echo "  Fetching $fname ..."
    python3 -c "import urllib.request; urllib.request.urlretrieve('$url', '$fname')" 2>/dev/null || \
        curl -sLO "$url" 2>/dev/null || \
        wget -q "$url" 2>/dev/null || \
        echo "    SKIP (download failed): $fname"
done

echo "Extracting .deb packages..."
for f in *.deb; do
    [ -f "$f" ] && dpkg-deb -x "$f" / 2>/dev/null || ar x "$f" && tar xf data.tar.* -C / 2>/dev/null || true
    rm -f control.tar.* data.tar.* debian-binary 2>/dev/null
done

ldconfig 2>/dev/null || true
rm -rf "$TMPDIR"

echo "Mesa libraries installed."
