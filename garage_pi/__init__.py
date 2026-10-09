"""Headless garage-radar service for a Raspberry Pi (Zero 2 W).

Runs the same detection code as the laptop app (app/radar, app/detection, app/recording) without
the Qt UI, and drives the garage outputs:

* one GPIO per stall to an ESP32 (HIGH = car in the stall) plus a "system OK" GPIO,
* a WS2812/NeoPixel bar on SPI0 MOSI (GPIO10),
* MQTT with Home Assistant discovery,
* a web page for status, configuration and calibration,
* a TCP re-broadcast of the raw radar byte stream for the laptop app.

    python -m garage_pi --data-dir ~/garage-radar            # live radar
    python -m garage_pi --data-dir /tmp/gr --replay recordings/<name>   # no hardware
"""
