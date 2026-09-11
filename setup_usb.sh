#!/usr/bin/env bash
# setup_usb.sh — Linux USB prerequisites for the STWIN.box sensor fleet.
#
# Installs exactly what the production board runs:
#   1. libusb-1.0 (the SDK's native libhs_datalog_v2.so links against it),
#   2. the hsdatalog udev rules (ST vendor 0483, DATALOG2 products 5743/5744
#      owned by group "hsdatalog") at /etc/udev/rules.d/30-hsdatalog.rules,
#   3. the "hsdatalog" group with the current user as a member.
#
# Needs sudo for apt/udev/group changes. Idempotent. Log out and back in
# (or reboot) after the first run so the new group membership applies.
set -euo pipefail

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

echo "== [1/3] libusb-1.0"
if ! ldconfig -p | grep -F libusb-1.0 >/dev/null; then
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq libusb-1.0-0
else
    echo "   already present"
fi

echo "== [2/3] udev rules (/etc/udev/rules.d/30-hsdatalog.rules)"
RULES='SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5743", ACTION=="add", GROUP="hsdatalog"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5744", ACTION=="add", GROUP="hsdatalog"'
if [ ! -f /etc/udev/rules.d/30-hsdatalog.rules ] || \
        [ "$(cat /etc/udev/rules.d/30-hsdatalog.rules)" != "$RULES" ]; then
    printf '%s\n' "$RULES" | $SUDO tee /etc/udev/rules.d/30-hsdatalog.rules >/dev/null
    $SUDO udevadm control --reload-rules
    $SUDO udevadm trigger
else
    echo "   already in place"
fi

echo "== [3/3] hsdatalog group membership"
if ! getent group hsdatalog >/dev/null; then
    $SUDO groupadd hsdatalog
fi
TARGET_USER="${SUDO_USER:-$USER}"
if id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx hsdatalog; then
    echo "   $TARGET_USER already in hsdatalog"
else
    $SUDO usermod -aG hsdatalog "$TARGET_USER"
    echo "   added $TARGET_USER to hsdatalog — log out and back in before using the boards"
fi

echo "== USB setup complete"
