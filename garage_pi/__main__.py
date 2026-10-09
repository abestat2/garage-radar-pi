"""Entry point:  python -m garage_pi [--data-dir DIR] [--replay RECORDING [--speed X] [--loop]]"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from garage_pi.config import init_data_dir  # noqa: E402
from garage_pi.radar_link import ReplayRadarLink, SerialRadarLink  # noqa: E402
from garage_pi.service import GarageService  # noqa: E402
from garage_pi.web import WebServer  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Garage radar headless service")
    ap.add_argument("--data-dir", default="~/garage-radar", help="settings, calibration and recordings")
    ap.add_argument("--replay", help="play a recording folder instead of the live radar (testing)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed factor")
    ap.add_argument("--loop", action="store_true", help="loop the replay")
    ap.add_argument("--no-hardware", action="store_true", help="never touch GPIO/SPI (mock outputs)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    cfg = init_data_dir(args.data_dir)
    if args.replay:
        factory = lambda settings, radar_cfg: ReplayRadarLink(args.replay, args.speed, args.loop)  # noqa: E731
    else:
        factory = lambda settings, radar_cfg: SerialRadarLink(settings, radar_cfg)  # noqa: E731
    service = GarageService(cfg, factory, use_hardware=not args.no_hardware)
    web = WebServer(service, int(cfg["web"]["port"])) if cfg["web"]["enabled"] else None

    def stop(*_):
        service.stop()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        service.run()
    finally:
        if web:
            web.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
