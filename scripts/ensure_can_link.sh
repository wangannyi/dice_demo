#!/usr/bin/env bash
# Prepare the Linux CAN interface after a USB-CAN adapter has been replugged.
# This changes only the network interface state; it sends no arm commands.
set -euo pipefail

channel="${DICE_CAN_INTERFACE:-can0}"
if ! ip -o link show dev "$channel" >/dev/null 2>&1; then
  echo "CAN 接口 $channel 不存在；检查 USB-CAN 适配器和 lsusb -t。" >&2
  exit 1
fi

can_is_up() {
  ip -o link show dev "$channel" | grep -Eq '(^|[,<])UP([,>])'
}

if can_is_up; then
  exit 0
fi

echo "CAN 接口 $channel 为 DOWN；恢复为 1 Mbps 后继续 Pipeline。" >&2
if ! sudo -n ip link set "$channel" up type can bitrate 1000000 2>/dev/null; then
  if [[ -t 0 ]]; then
    sudo ip link set "$channel" up type can bitrate 1000000
  else
    echo "无法在无交互终端获得 sudo 权限；请运行：sudo ip link set $channel up type can bitrate 1000000" >&2
    exit 1
  fi
fi

if ! can_is_up; then
  echo "CAN 接口 $channel 仍未 UP；检查适配器和系统日志。" >&2
  exit 1
fi
echo "CAN 接口 $channel 已恢复；继续 Pipeline。" >&2
