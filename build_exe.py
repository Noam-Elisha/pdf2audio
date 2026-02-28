"""Build pdf2audio as a standalone Windows exe using PyInstaller."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    project_dir = Path(__file__).parent
    spec_file = project_dir / "pdf2audio.spec"
    dist_dir = project_dir / "dist" / "pdf2audio"

    if not spec_file.exists():
        print(f"Error: {spec_file} not found")
        sys.exit(1)

    print("=" * 50)
    print("  Building pdf2audio.exe")
    print("=" * 50)
    print()

    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(spec_file),
        "--distpath", str(project_dir / "dist"),
        "--workpath", str(project_dir / "build"),
        "--noconfirm",
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(project_dir))

    if result.returncode != 0:
        print(f"\nBuild failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    exe_path = dist_dir / "pdf2audio.exe"
    if exe_path.exists():
        # Calculate total size of dist folder
        total_size = sum(f.stat().st_size for f in dist_dir.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)

        print()
        print("=" * 50)
        print(f"  Build complete!")
        print(f"  Output: {dist_dir}")
        print(f"  Exe:    {exe_path}")
        print(f"  Size:   {size_mb:.0f} MB")
        print("=" * 50)
    else:
        print(f"\nWarning: {exe_path} not found after build")
        sys.exit(1)


if __name__ == "__main__":
    main()
