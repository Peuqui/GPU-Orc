#!/bin/bash
# Switch GPU-Orc from the systemd USER unit to the SYSTEM unit gpu-orc@<user>.service, which runs at
# boot without a login. Run as root: sudo deploy/switch-to-system-unit.sh [user]
# Same port as before, so the two units must never run together: the user unit is stopped first.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Needs root: sudo $0 [user]" >&2
    exit 1
fi
SERVICE_USER=${1:-$SUDO_USER}
USER_ID=$(id -u "$SERVICE_USER")
HOME_DIR=$(getent passwd "$SERVICE_USER" | cut -d: -f6)
REPO=$(cd "$(dirname "$0")/.." && pwd)
UNIT=gpu-orc@$SERVICE_USER.service

[[ -x $HOME_DIR/.local/share/gpu-orc/venv/bin/gpu-orc ]] || { echo "No gpu-orc in $HOME_DIR/.local/share/gpu-orc/venv: run deploy/install.sh first." >&2; exit 1; }
[[ -f $HOME_DIR/.config/gpu-orc/config.yaml ]] || { echo "No $HOME_DIR/.config/gpu-orc/config.yaml." >&2; exit 1; }

echo "== 1/3 Stopping and disabling the user unit of $SERVICE_USER"
sudo -u "$SERVICE_USER" XDG_RUNTIME_DIR="/run/user/$USER_ID" systemctl --user disable --now gpu-orc || echo "(no user unit was running)"

echo "== 2/3 Installing the system unit"
install -v -m 644 "$REPO/deploy/system/gpu-orc@.service" /etc/systemd/system/gpu-orc@.service
systemctl daemon-reload

echo "== 3/3 Starting $UNIT"
systemctl enable --now "$UNIT"
sleep 2
systemctl is-active "$UNIT"
