"""
build_sidecar.py
-----------------
Packages backend/main.py into a single-file executable using PyInstaller,
named per Tauri's sidecar convention: <name>-<target-triple>[.exe]

Usage:
    python build_sidecar.py

Output:
    src-tauri/binaries/tts-backend-<target-triple>

Run this BEFORE `tauri build` / `tauri dev` so the sidecar binary exists.
Requires: pip install pyinstaller (in addition to backend/requirements.txt)
"""

import platform
import subprocess
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
OUTPUT_DIR = ROOT / "src-tauri" / "binaries"


def get_target_triple() -> str:
    """
    Returns the Rust target triple for the current platform, matching
    what `rustc -vV` would report — required by Tauri's sidecar naming.
    """
    system = platform.system()
    machine = platform.machine().lower()

    arch = "x86_64" if machine in ("x86_64", "amd64") else "aarch64"

    if system == "Linux":
        return f"{arch}-unknown-linux-gnu"
    elif system == "Darwin":
        return f"{arch}-apple-darwin"
    elif system == "Windows":
        return f"{arch}-pc-windows-msvc"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def build():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_triple = get_target_triple()
    is_windows = platform.system() == "Windows"

    print(f"Building sidecar for target: {target_triple}")

    subprocess.run(
        [
            "pyinstaller",
            "--onefile",
            "--name", "tts-backend",
            "--distpath", str(OUTPUT_DIR / "_dist"),
            "--workpath", str(OUTPUT_DIR / "_build"),
            "--specpath", str(OUTPUT_DIR / "_spec"),
            "--add-data", f"{BACKEND_DIR / 'curated_script.json'}{':' if not is_windows else ';'}.",
            str(BACKEND_DIR / "main.py"),
        ],
        check=True,
        cwd=str(ROOT),
    )

    suffix = ".exe" if is_windows else ""
    built_binary = OUTPUT_DIR / "_dist" / f"tts-backend{suffix}"
    final_binary = OUTPUT_DIR / f"tts-backend-{target_triple}{suffix}"

    shutil.copy2(str(built_binary), str(final_binary))
    if not is_windows:
        final_binary.chmod(0o755)

    print(f"\nSidecar binary ready: {final_binary}")
    print("You can now run `npm run tauri dev` or `npm run tauri build`.")


if __name__ == "__main__":
    build()
