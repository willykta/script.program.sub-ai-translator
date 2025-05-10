import tempfile
import shutil
import subprocess
import os
import urllib.request
import json
from pathlib import Path
from packaging.version import parse

SUBTITLE_PACKAGE = "py-subtitle-extractor"
BABEL_PACKAGE = "babel"
SUBTITLE_DIR = Path("resources/lib/py_subtitle_extractor")
BABEL_DIR = Path("resources/lib/babel")
SUBTITLE_INIT = SUBTITLE_DIR / "__init__.py"

def get_latest_version_from_pypi(package):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json") as resp:
        data = json.load(resp)
        return data["info"]["version"]

def get_vendored_version(init_path: Path):
    if not init_path.exists():
        return None
    for line in init_path.read_text().splitlines():
        if line.startswith("__version__"):
            return line.split("=")[-1].strip().strip('"\'')
    return None

def inject_version(init_path: Path, version: str):
    content = init_path.read_text()
    if "__version__" not in content:
        with init_path.open("a") as f:
            f.write(f'\n__version__ = "{version}"\n')
        print(f"ℹ️  Injected __version__ = \"{version}\"")

def vendor_package(package: str, target_dir: Path, subpaths: list[str] = None, version: str = None):
    version_str = f"=={version}" if version else ""
    print(f"📦 Vendoring {package}{version_str}")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.check_call([
            "pip", "install", f"{package}{version_str}", "--target", tmp
        ])
        src = Path(tmp) / package.replace("-", "_")
        if not src.exists():
            raise RuntimeError(f"Package not found in {tmp}")

        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True)

        if subpaths:
            for sub in subpaths:
                src_path = src / sub
                dst_path = target_dir / sub
                if src_path.is_dir():
                    shutil.copytree(src_path, dst_path)
                elif src_path.is_file():
                    shutil.copy2(src_path, dst_path)
                else:
                    print(f"⚠️  Skipped missing path: {src_path}")
        else:
            shutil.copytree(src, target_dir, dirs_exist_ok=True)

    print(f"✅ Vendored {package} into {target_dir}")

def main():
    latest_sub = get_latest_version_from_pypi(SUBTITLE_PACKAGE)
    vendored_sub = get_vendored_version(SUBTITLE_INIT)

    if vendored_sub is None or parse(vendored_sub) < parse(latest_sub):
        print(f"🔄 Updating {SUBTITLE_PACKAGE}: {vendored_sub or 'none'} → {latest_sub}")
        vendor_package(SUBTITLE_PACKAGE, SUBTITLE_DIR, version=latest_sub)
        inject_version(SUBTITLE_INIT, latest_sub)
    else:
        print(f"✅ {SUBTITLE_PACKAGE} is up to date ({vendored_sub})")

    vendor_package(BABEL_PACKAGE, BABEL_DIR) 

if __name__ == "__main__":
    main()
