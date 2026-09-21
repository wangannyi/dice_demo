# K3 USB CAN 内核包

本目录为可选内核构建工具，通常已有 can0 的设备无需安装。Git 分发不包含内核源码压缩包或构建产物；需按 build.sh 的输入准备源码并构建后，才能运行下文安装命令。主流程见 [顶层 README](../README.md)。

原验证目标设备：`spacemit@10.0.90.160`，Bianbu 4.0.6、RISC-V 64 位。

## 配置

- 原内核：`6.18.3-generic`，软件包版本 `6.18.3-1.0.7.4`。
- 源码：[SpacemiT linux-6.18，提交 4158237f35b8fd62ba198c1627e5a66e5a34c50f](https://github.com/spacemit-com/linux-6.18/tree/4158237f35b8fd62ba198c1627e5a66e5a34c50f)。
- 新内核：`6.18.3-generic-usbcan`，软件包版本 `6.18.3-1.0.7.4+usbcan1`。
- 使用原 `/boot/config-6.18.3-generic`，启用以下模块：`CAN_GS_USB`、`CAN_8DEV_USB`、`CAN_EMS_USB`、`CAN_ESD_USB`、`CAN_ETAS_ES58X`、`CAN_F81604`、`CAN_KVASER_USB`、`CAN_MCBA_USB`、`CAN_PEAK_USB`、`CAN_UCAN`。

`artifacts/config.diff` 记录配置差异，`kernel.config` 为完整配置。安装包包含内核镜像、内核模块和对应设备树。使用厂商源码的安装脚本调用 Bianbu 的 initramfs、设备树及启动配置钩子。

## 安装

本目录的编译脚本不安装内核、不重启设备。以下命令由操作者在板端执行。

```bash
cd /home/spacemit/dice_demo/kernel_usbcan_20260921/artifacts
sha256sum -c SHA256SUMS
sudo test -e /boot/env_k3.txt.before-usbcan || sudo cp -a /boot/env_k3.txt /boot/env_k3.txt.before-usbcan
sudo dpkg -i linux-image-6.18.3-generic-usbcan_6.18.3-1.0.7.4+usbcan1_riscv64.deb
cat /boot/env_k3.txt
```

安装成功后，核对 `knl_name`、`ramdisk_name` 和 `dtb_dir` 指向 `6.18.3-generic-usbcan`，再执行：

```bash
sudo reboot
```

## 验证 USB CAN

```bash
uname -r
modinfo gs_usb
sudo modprobe gs_usb
lsusb
ip -details link show type can
```

`uname -r` 应为 `6.18.3-generic-usbcan`。连接兼容的 USB CAN 适配器后应出现 CAN 网络接口；未连接设备时不会凭空出现 `can0`。适配器型号决定实际使用哪个驱动，专有协议设备不一定兼容这些驱动。

例如设备为 `can0`、总线约定为 1 Mbit/s 时：

```bash
sudo ip link set can0 up type can bitrate 1000000
ip -details -statistics link show can0
```

编译和包内容验证不等于已完成启动或 CAN 收发验证；后两项需要安装、重启并连接适配器后测试。

## 恢复原内核

原内核包保持安装。需要恢复时，通过可用终端恢复启动配置并重启：

```bash
sudo cp -a /boot/env_k3.txt.before-usbcan /boot/env_k3.txt
sudo reboot
```

## 重新编译

源码、编译依赖和产物位于 `/home/spacemit/dice_demo/kernel_usbcan_20260921`。依赖通过发行版软件包提取到 `tools/`，编译无需 root。保留 `downloads/`、`config.original` 和源码目录，执行：

```bash
cd /home/spacemit/dice_demo/kernel_usbcan_20260921
bash build.sh > build.log 2>&1
```

脚本使用六路并行编译，完成后在 `artifacts/` 生成 `.deb` 和 `SHA256SUMS`。
