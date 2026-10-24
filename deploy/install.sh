#!/bin/bash
# Install GPU-Orc for the current user: its own venv with the code of this clone (editable), the
# command `gpu-orc` in ~/.local/bin, a systemd user unit, and, if there is none yet, a first config
# (~/.config/gpu-orc/config.yaml from deploy/config.example.yaml: adapt the consumers to this machine).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=${XDG_DATA_HOME:-$HOME/.local/share}/gpu-orc/venv
BIN=$HOME/.local/bin
UNITS=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
CONFIG_FILE=${XDG_CONFIG_HOME:-$HOME/.config}/gpu-orc/config.yaml

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
    echo "Python 3.12 or newer is needed; python3 here is $(python3 --version 2>&1)." >&2
    exit 1
fi
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet -e "$REPO"
mkdir -p "$BIN" "$UNITS" "$(dirname "$CONFIG_FILE")"
ln -sfn "$VENV/bin/gpu-orc" "$BIN/gpu-orc"
cp "$REPO/deploy/gpu-orc.service" "$UNITS/gpu-orc.service"
systemctl --user daemon-reload
if [ ! -e "$CONFIG_FILE" ]; then
    cp "$REPO/deploy/config.example.yaml" "$CONFIG_FILE"
    echo "Wrote $CONFIG_FILE from the example: check its consumers."
fi
echo "Installed $REPO to $VENV."
# The system unit and the user unit share the port: once the system unit is set up, never start the user unit.
if systemctl is-enabled --quiet "gpu-orc@$USER" 2>/dev/null; then
    echo "Next: systemctl restart gpu-orc@$USER   (the system unit runs this code after a restart)"
else
    echo "Next: systemctl --user enable --now gpu-orc   (or, to run at boot: sudo $REPO/deploy/switch-to-system-unit.sh)"
fi
