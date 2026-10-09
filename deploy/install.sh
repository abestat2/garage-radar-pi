#!/usr/bin/env bash
# Install the garage radar service on a Raspberry Pi (Zero 2 W, Raspberry Pi OS Bookworm, 64-bit or 32-bit Lite).
#
#   git clone <this repo>  (or copy the garage_radar_tester folder to the Pi)
#   cd garage_radar_tester
#   bash deploy/install.sh         # installs and starts the service
#
# Re-running is safe (it updates the code and keeps the data directory).
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/garage-radar}"
USER_NAME="$(id -un)"

echo "== app: $APP_DIR   data: $DATA_DIR   user: $USER_NAME"

echo "== system packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-numpy python3-serial python3-gpiozero python3-lgpio python3-spidev

echo "== SPI for the NeoPixel bar (GPIO10 / MOSI)"
if command -v raspi-config >/dev/null; then sudo raspi-config nonint do_spi 0 || true; fi

echo "== serial / GPIO / SPI permissions"
sudo usermod -a -G dialout,gpio,spi "$USER_NAME" || true

echo "== python environment"
python3 -m venv --system-site-packages "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/deploy/requirements-pi.txt"

echo "== data directory (seeded with the laptop's tested calibration)"
"$APP_DIR/.venv/bin/python" - <<EOF
import sys; sys.path.insert(0, "$APP_DIR")
from garage_pi.config import init_data_dir
cfg = init_data_dir("$DATA_DIR")
print("data dir ready:", cfg.data_dir)
EOF

echo "== systemd service"
sed -e "s|@USER@|$USER_NAME|g" -e "s|@APP_DIR@|$APP_DIR|g" -e "s|@DATA_DIR@|$DATA_DIR|g" \
    "$APP_DIR/deploy/garage-radar.service" | sudo tee /etc/systemd/system/garage-radar.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable garage-radar.service
sudo systemctl restart garage-radar.service

IP="$(hostname -I | awk '{print $1}')"
echo
echo "Installed. Web page: http://$IP:8080   (or http://$(hostname).local:8080)"
echo "Logs:     journalctl -u garage-radar -f"
echo "If this was the first install, reboot once so the group and SPI changes take effect: sudo reboot"
