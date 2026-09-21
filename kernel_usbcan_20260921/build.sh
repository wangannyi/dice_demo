#!/usr/bin/env bash
set -euo pipefail
BASE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC="$BASE/linux-6.18-4158237f35b8fd62ba198c1627e5a66e5a34c50f"
export PATH="$BASE/tools/usr/bin:$PATH"
export BISON_PKGDATADIR="$BASE/tools/usr/share/bison"
export M4="$BASE/tools/usr/bin/m4"
export HOSTCFLAGS="-O2 -I$BASE/tools/usr/include"
export HOSTLDFLAGS="-L$BASE/tools/usr/lib/riscv64-linux-gnu"
export LD_LIBRARY_PATH="$BASE/tools/usr/lib/riscv64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ARCH=riscv LOCALVERSION=-generic-usbcan
export KBUILD_BUILD_USER=spacemit KBUILD_BUILD_HOST=bianbu-spacemitk3picoitx
export KBUILD_BUILD_VERSION=1.0.7.4+usbcan1
mkdir -p "$BASE/artifacts"
cd "$SRC"
if [[ ! -f "$BASE/config.original" ]]; then
    cp /boot/config-6.18.3-generic "$BASE/config.original"
fi
cp "$BASE/config.original" .config
for name in CAN_8DEV_USB CAN_EMS_USB CAN_ESD_USB CAN_ETAS_ES58X CAN_F81604 CAN_GS_USB CAN_KVASER_USB CAN_MCBA_USB CAN_PEAK_USB CAN_UCAN; do
    scripts/config --module "$name"
done
make olddefconfig
scripts/diffconfig "$BASE/config.original" .config > "$BASE/artifacts/config.diff"
cp .config "$BASE/artifacts/kernel.config"
for name in CAN_8DEV_USB CAN_EMS_USB CAN_ESD_USB CAN_ETAS_ES58X CAN_F81604 CAN_GS_USB CAN_KVASER_USB CAN_MCBA_USB CAN_PEAK_USB CAN_UCAN; do
    grep -qx "CONFIG_$name=m" .config
done
make -s kernelrelease > "$BASE/artifacts/kernelrelease.txt"
date -Is > "$BASE/build.started"
make -j6 all modules dtbs
date -Is > "$BASE/build.finished"
export KERNELRELEASE=$(cat "$BASE/artifacts/kernelrelease.txt")
export SRCARCH=riscv srctree="$SRC" MAKE=make KCONFIG_CONFIG=.config
PACKAGE="linux-image-$KERNELRELEASE"
VERSION=6.18.3-1.0.7.4+usbcan1
scripts/package/builddeb "$PACKAGE"
STAGE="$SRC/debian/$PACKAGE"
SIZE=$(du -sk "$STAGE" | cut -f1)
cat > "$STAGE/DEBIAN/control" <<EOF
Package: $PACKAGE
Version: $VERSION
Architecture: riscv64
Maintainer: spacemit <spacemit@localhost>
Section: kernel
Priority: optional
Installed-Size: $SIZE
Depends: spacemit-flash-dtbs, initramfs-tools, kmod
Provides: linux-image-6.18.3
Description: SpacemiT K3 kernel 6.18.3 with USB CAN modules
 Based on vendor commit 4158237f35b8fd62ba198c1627e5a66e5a34c50f
 and the installed Bianbu 6.18.3-1.0.7.4 configuration.
 Includes gs_usb and the other in-tree USB CAN drivers.
EOF
install -Dm644 "$BASE/artifacts/config.diff" "$STAGE/usr/share/doc/$PACKAGE/config.diff"
install -Dm644 "$SRC/COPYING" "$STAGE/usr/share/doc/$PACKAGE/copyright"
install -Dm644 "$SRC/LICENSES/preferred/GPL-2.0" "$STAGE/usr/share/doc/$PACKAGE/GPL-2.0.txt"
find "$STAGE" -type d -exec chmod go-w {} +
dpkg-deb --root-owner-group -Zxz -z3 --build "$STAGE" "$BASE/artifacts/${PACKAGE}_${VERSION}_riscv64.deb"
cd "$BASE/artifacts"
sha256sum *.deb > SHA256SUMS
date -Is > "$BASE/package.finished"
