import tempfile
import shutil
import subprocess
import os
import urllib.request
import json
from pathlib import Path
from packaging.version import parse

PACKAGE = "py-subtitle-extractor"
VENDOR_DIR = Path("resources/lib/py_subtitle_extractor")
INIT_FILE = VENDOR_DIR / "__init__.py"

def get_latest_version_from_pypi():
    with urllib.request.urlopen(f"https://pypi.org/pypi/{PACKAGE}/json") as resp:
        data = json.load(resp)
        return data["info"]["version"]

def get_vendored_version():
    if not INIT_FILE.exists():
        return None
    for line in INIT_FILE.read_text().splitlines():
        if line.startswith("__version__"):
            return line.split("=")[-1].strip().strip('"\'')
    return None

def inject_version(init_path: Path, version: str):
    content = init_path.read_text()
    if "__version__" not in content:
        with init_path.open("a") as f:
            f.write(f'\n__version__ = "{version}"\n')
        print(f"ℹ️  Injected __version__ = \"{version}\"")

def vendor_package(version: str):
    print(f"📦 Vendoring {PACKAGE}=={version}")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.check_call([
            "pip", "install", f"{PACKAGE}=={version}", "--target", tmp
        ])
        src = Path(tmp) / PACKAGE.replace("-", "_")
        if not src.exists():
            raise RuntimeError(f"Package not found in {tmp}")

        if VENDOR_DIR.exists():
            shutil.rmtree(VENDOR_DIR)
        shutil.copytree(src, VENDOR_DIR)

    if INIT_FILE.exists():
        inject_version(INIT_FILE, version)
    else:
        print("⚠️  __init__.py not found, cannot inject version")
    print("✅ Vendoring complete")

def main():
    latest = get_latest_version_from_pypi()
    vendored = get_vendored_version()

    if vendored is None:
        print(f"Vendored version not found, installing {latest}")
        vendor_package(latest)
        return

    if parse(vendored) < parse(latest):
        print(f"⚠️  Outdated: vendored = {vendored}, latest = {latest}")
        vendor_package(latest)
    else:
        print(f"✅ Vendored {PACKAGE} is up to date ({vendored})")

if __name__ == "__main__":
    main()
