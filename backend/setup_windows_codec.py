"""
One-time local setup for Windows: downloads Cisco's OpenH264 DLL so OpenCV
can encode H.264 (avc1) video that plays inline in browsers via st.video().

Without this, OpenCV falls back to the mp4v codec, which most browsers will
let users download but won't preview inline. Not needed on Linux deploy
targets (Streamlit Cloud / Hugging Face Spaces) — packages.txt installs
ffmpeg there, which already provides working H.264 support.

Usage (from the project root, with the venv active):
    python -m backend.setup_windows_codec
"""
from __future__ import annotations

import bz2
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

OPENH264_URL = "https://github.com/cisco/openh264/releases/download/v1.8.0/openh264-1.8.0-win64.dll.bz2"
DLL_NAME = "openh264-1.8.0-win64.dll"


def main():
    if not sys.platform.startswith("win"):
        print("This setup step is only needed on Windows. Skipping.")
        return

    import cv2
    cv2_dir = Path(cv2.__file__).resolve().parent
    dest = cv2_dir / DLL_NAME

    if dest.exists():
        print(f"Already present: {dest}")
        return

    print(f"Downloading {OPENH264_URL} ...")
    with urlopen(OPENH264_URL) as resp:
        compressed = resp.read()

    data = bz2.decompress(compressed)
    with open(dest, "wb") as fh:
        fh.write(data)

    print(f"Installed OpenH264 DLL to {dest}")
    print("Browser-playable (H.264) video export is now enabled.")


if __name__ == "__main__":
    main()
