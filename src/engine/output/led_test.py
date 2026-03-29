from __future__ import annotations

import argparse
import json
import sys
import time

import tinytuya


DEVICE_ID = "bf0057a91523c3c56bxtc4"
IP_ADDRESS = "192.168.0.211"
LOCAL_KEY = "G!sq8!*Kp(D;veY1"

VERSIONS_TO_TRY = [3.3, 3.4, 3.5]

# DPS mapping for this LED strip/controller
DPS_SWITCH = "20"   # switch_led
DPS_MODE = "21"     # work_mode
DPS_BRIGHT = "22"   # bright_value
DPS_TEMP = "23"     # temp_value
DPS_COLOUR = "24"   # colour_data


def rgb_to_tuya_hsv_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB 0..255 to Tuya HSV hex format HHHHSSSSVVVV.
    H: 0..360
    S: 0..1000
    V: 0..1000
    """
    r_f = max(0, min(255, int(r))) / 255.0
    g_f = max(0, min(255, int(g))) / 255.0
    b_f = max(0, min(255, int(b))) / 255.0

    mx = max(r_f, g_f, b_f)
    mn = min(r_f, g_f, b_f)
    diff = mx - mn

    if diff == 0:
        h = 0
    elif mx == r_f:
        h = (60 * ((g_f - b_f) / diff) + 360) % 360
    elif mx == g_f:
        h = (60 * ((b_f - r_f) / diff) + 120) % 360
    else:
        h = (60 * ((r_f - g_f) / diff) + 240) % 360

    s = 0 if mx == 0 else int(round((diff / mx) * 1000))
    v = int(round(mx * 1000))

    h_i = max(0, min(360, int(round(h))))
    s_i = max(0, min(1000, s))
    v_i = max(0, min(1000, v))

    return f"{h_i:04x}{s_i:04x}{v_i:04x}"


def connect(version: float) -> tinytuya.Device:
    d = tinytuya.Device(DEVICE_ID, IP_ADDRESS, LOCAL_KEY)
    d.set_version(version)

    try:
        d.set_socketPersistent(True)
    except Exception:
        pass

    return d


def print_status(version: float) -> bool:
    print(f"\n=== Testing version {version:.1f} ===")
    try:
        d = connect(version)
        status = d.status()
        print("STATUS:")
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return True
    except Exception as exc:
        print(f"FAILED on version {version:.1f}: {exc}")
        return False


def find_working_device() -> tuple[tinytuya.Device, float]:
    last_error: Exception | None = None

    for version in VERSIONS_TO_TRY:
        try:
            print(f"Trying version {version:.1f} ...")
            d = connect(version)
            status = d.status()
            print("Connected OK.")
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return d, version
        except Exception as exc:
            print(f"Version {version:.1f} failed: {exc}")
            last_error = exc

    raise RuntimeError(f"Could not connect with any version. Last error: {last_error}")


def set_on(d: tinytuya.Device) -> None:
    d.set_value(DPS_SWITCH, True)


def set_off(d: tinytuya.Device) -> None:
    d.set_value(DPS_SWITCH, False)


def set_colour(d: tinytuya.Device, r: int, g: int, b: int) -> None:
    colour_data = rgb_to_tuya_hsv_hex(r, g, b)
    print(f"Sending colour_data={colour_data}")
    d.set_value(DPS_SWITCH, True)
    d.set_value(DPS_MODE, "colour")
    d.set_value(DPS_COLOUR, colour_data)


def set_white(d: tinytuya.Device, brightness: int = 1000, temperature: int = 500) -> None:
    brightness = max(10, min(1000, int(brightness)))
    temperature = max(0, min(1000, int(temperature)))

    print(f"Sending white brightness={brightness} temperature={temperature}")
    d.set_value(DPS_SWITCH, True)
    d.set_value(DPS_MODE, "white")
    d.set_value(DPS_BRIGHT, brightness)
    d.set_value(DPS_TEMP, temperature)


def blink_test(d: tinytuya.Device) -> None:
    print("Blink test: red -> green -> blue -> white -> off")
    set_colour(d, 255, 0, 0)
    time.sleep(1.0)
    set_colour(d, 0, 255, 0)
    time.sleep(1.0)
    set_colour(d, 0, 0, 255)
    time.sleep(1.0)
    set_white(d, 1000, 500)
    time.sleep(1.0)
    set_off(d)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Deltaco SH-LS3M via TinyTuya")

    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=["status", "red", "green", "blue", "white", "off", "on", "blink"],
        help="Command to send",
    )
    parser.add_argument("--version", type=float, default=None, help="Force a specific version")
    parser.add_argument("--brightness", type=int, default=1000, help="White brightness 10..1000")
    parser.add_argument("--temperature", type=int, default=500, help="White temperature 0..1000")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "status":
        if args.version is not None:
            ok = print_status(args.version)
            return 0 if ok else 1

        any_ok = False
        for version in VERSIONS_TO_TRY:
            ok = print_status(version)
            any_ok = any_ok or ok
        return 0 if any_ok else 1

    try:
        if args.version is not None:
            d = connect(args.version)
            status = d.status()
            print("Connected with forced version:")
            print(json.dumps(status, indent=2, ensure_ascii=False))
            used_version = args.version
        else:
            d, used_version = find_working_device()

        print(f"Using version {used_version:.1f}")

        if args.command == "on":
            set_on(d)
        elif args.command == "off":
            set_off(d)
        elif args.command == "red":
            set_colour(d, 255, 0, 0)
        elif args.command == "green":
            set_colour(d, 0, 255, 0)
        elif args.command == "blue":
            set_colour(d, 0, 0, 255)
        elif args.command == "white":
            set_white(d, args.brightness, args.temperature)
        elif args.command == "blink":
            blink_test(d)

        print("Done.")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())