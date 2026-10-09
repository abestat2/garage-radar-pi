# Garage Radar (Raspberry Pi)

Car-entry and parked-car detection for a three-stall garage, running headless on a
**Raspberry Pi Zero 2 W** with a **TI IWR6843LEVM** mmWave radar. Each stall gets a digital output
for an ESP32 (HIGH = car in the stall), plus a NeoPixel status bar, MQTT / Home Assistant entities,
and a web page for status, configuration and calibration.

This is the deployment repository. The detection algorithms were developed and validated with the
laptop test bench, **[mmWave_car_detect_test](https://github.com/abestat2/mmWave_car_detect_test)**
(Qt app, simulator, recordings, analysis tools). Its `app/` modules are copied here unchanged, so
fixes to the detection code should be made there first and copied across.

```
garage_pi/        the service: radar link, outputs (GPIO, LEDs, MQTT), web page, raw stream
app/              detection code shared with the test bench (radar parsing, geometry, detection, recording)
configs/          tested radar profile, empty-garage reference and calibration (seed the Pi's data dir)
deploy/           install script and systemd unit
tests/            service tests, incl. a full replay of a real garage recording (tests/data)
```

Run the tests on any PC: `pip install numpy pyserial paho-mqtt pytest && python -m pytest`.

The Pi runs the same detection code as the laptop app, headless, and drives the garage outputs:
one GPIO per stall to an ESP32, a "system OK" GPIO, a NeoPixel bar, MQTT / Home Assistant,
a web page, and a raw-data stream that the laptop app can watch over Wi-Fi.

## 1. Install

On the Pi (Raspberry Pi OS Bookworm Lite, Wi-Fi configured, SSH enabled):

```bash
git clone https://github.com/abestat2/garage-radar-pi.git && cd garage-radar-pi
bash deploy/install.sh
sudo reboot            # first install only (groups + SPI)
```

The script:
* installs numpy, gpiozero, lgpio and spidev from apt (fast on a Zero);
* creates `.venv` with pyserial and paho-mqtt;
* enables SPI and adds you to `dialout`, `gpio` and `spi`;
* seeds `~/garage-radar` with the laptop's tested radar config, empty-garage reference and
  calibration;
* installs and starts `garage-radar.service`, which starts at boot and restarts on failure.

| Task | Command |
|---|---|
| Web page | `http://<pi>:8080` (or `http://<hostname>.local:8080`) |
| Logs | `journalctl -u garage-radar -f` |
| Restart | `sudo systemctl restart garage-radar` |
| Test without the radar | `.venv/bin/python -m garage_pi --data-dir /tmp/gr --replay <recording folder> --no-hardware` |

## 2. Wiring

**Radar:** the IWR6843LEVM goes into the Zero's **USB** micro-USB port (the inner one, not PWR)
through an OTG adapter or cable. It is found automatically by its CP2105 bridge.
Use a good 5 V supply for the Pi (≥ 2.5 A): it also powers the radar through USB. Measure the
radar's actual draw with a USB meter.

| Function | BCM GPIO | Header pin | Connect to |
|---|---|---|---|
| Stall 1 (far side of double door) | 17 | 11 | ESP32 input |
| Stall 2 (centre) | 27 | 13 | ESP32 input |
| Stall 3 (single door) | 22 | 15 | ESP32 input |
| System OK | 23 | 16 | ESP32 input |
| NeoPixel data (SPI0 MOSI) | 10 | 19 | 74AHCT125 → LED DIN |
| Ground | — | 6, 9, 14, 20, 25 | ESP32 GND, LED GND, 5 V supply GND |

* **Logic levels:** 3.3 V, the same as the ESP32. Connect directly, with a common ground. A 330 Ω
  series resistor per line is cheap protection.
* **NeoPixels:**
  * Power the strip from 5 V, not the Pi's 3.3 V pin. Budget up to 60 mA per LED at full white;
    the default brightness is 25%.
  * A 74AHCT125 (or similar) level shifter on the data line is recommended. Many WS2812B accept
    3.3 V data, but not reliably.
  * Add the usual 300–500 Ω resistor in series with DIN, and a 1000 µF capacitor across the strip
    supply.
* **Configuration:** pins, LED count, which LEDs belong to which stall, colours and brightness are
  all set on the web page (Settings), then restart the service.

## 3. Contract for the ESP32

| Pin | HIGH | LOW |
|---|---|---|
| Stall N | car in the stall (OCCUPIED, or VEHICLE EXITING while it is still in) | stall empty |
| System OK | radar running and parked-car detection ready: stall pins are trustworthy | radar offline, starting, or not ready: treat the stall pins as "last known" |

* **Edges:** a **rising edge** is a confirmed car entry, a **falling edge** a confirmed exit. Act
  on the edges.
* **Timing:**
  * Entry: the rising edge comes after the car has stopped and its echo has been stable for the
    occupancy persistence time, typically **5–10 s** after it stops.
  * Exit: the falling edge comes when the stall has looked empty for the empty persistence
    time (5 s).
  * Optional earlier edge: setting `gpio.entering_is_present = true` raises the pin already at
    VEHICLE ENTERING, from moving points only. It is earlier but less certain, and a person
    walking in the stall can cause it. Slow cars at wide angles may produce no moving points at
    all; stall 3's events came from the heatmap alone.
* **No glitches on errors:** UNKNOWN or SENSOR ERROR **hold** each stall pin at its last level and
  drop System OK.
* **At boot:** all pins are LOW until the first confirmed state, and System OK goes HIGH after
  about 2 s of valid data.
* **Shutdown:** a clean shutdown sets every pin LOW.
* **Inverting:** `gpio.active_high = false` inverts every pin.

**Sleeping ESP-NOW receivers:** ESP-NOW does not store messages. A battery device that sleeps and
wakes on a timer misses anything sent while it was asleep. Two common patterns:
1. The receiver, when it wakes, sends a short "state?" request. The mains-powered ESP next to the
   Pi answers with the current pin levels and a counter of edges since the last request.
2. The sender repeats the latest state every second for longer than the receiver's sleep
   interval, and the receiver stays awake briefly to listen.

Both need the sender ESP to keep the latest state and edges in memory, which is why the Pi
provides levels as well as edges.

## 4. Home Assistant (MQTT)

Settings → mqtt: set `enabled`, `host` (your broker, e.g. the Mosquitto add-on), username and
password, then restart the service. With MQTT discovery on (the HA default), a **Garage Radar**
device appears with:

| Entity | Values |
|---|---|
| `binary_sensor.stall_N_occupied` | ON / OFF (occupancy) |
| `sensor.stall_N_state` | EMPTY / OCCUPIED / VEHICLE ENTERING / VEHICLE EXITING / UNKNOWN / SENSOR/CONFIGURATION ERROR, with confidence and readings as attributes |
| `event.stall_N` | event types `ENTERED` / `EXITED`, for automations |
| `sensor.radar_status` | ok / starting / radar offline (diagnostic) |

Example automation trigger: *Event* → entity `event.stall_2` → event type `ENTERED`.

Raw topics are under `garage_radar/<device_id>/…`, documented in `garage_pi/mqtt_ha.py`.

## 5. Calibration without the laptop (web page → Calibrate)

The Pi ships with the laptop's calibration. Recalibrate on the Pi if the radar is moved or
re-aimed, the radar configuration changes, or the readings drift (the stall cards warn when an
empty stall's reading creeps toward its threshold).

1. **Empty garage, doors open:** record an empty reference.
2. **Empty garage, doors closed:** record an empty reference (combined with step 1).
3. **Cars parked:** choose which stalls have a car and record. Include each stall **empty while
   its neighbours are parked**; the page lists which of these are still missing.
4. **Fit thresholds** (about a minute). The report shows margins and decoding errors, and the new
   model is loaded immediately.

**Self-updating reference:**
* It is saved every 15 minutes and on shutdown, and reloaded at boot.
* "Revert to captured reference" (Status page) discards it.
* Recording a new empty reference resets it.

## 6. Debugging from the laptop

* **Live view:** in the laptop app, go to Connection → **Remote radar (garage Raspberry Pi)**,
  enter the Pi's host name or IP, then **Connect to Pi**.
  * The app loads the Pi's `radar.cfg` and watches the raw radar stream (TCP 7777).
  * All views, recording and replay work. The Pi stays in control of the radar.
* **Recordings:** the Files tab downloads recordings made on the Pi as zip files. Unzip and open
  them in the app (Record / replay → Open recording).
* **Geometry and zones:** edit them in the laptop app (Garage Geometry Mode → Save as…), then
  upload on Files → Garage geometry.

## 7. Not yet verified

This service has been tested on a PC: replays of the real garage recordings through the full
service, with mock GPIO and LEDs, a local MQTT payload check, the web API, and the raw stream
into the laptop app. It has **not yet run on a Pi Zero 2 W**. Things to check on first install:
* CPU load (`top`): the expectation is well under one core at 4 frames/s;
* the CP2105's names under Linux (the auto-detect has a fallback by interface order);
* the WS2812 timing with your LED strip;
* the GPIO levels at the ESP.
