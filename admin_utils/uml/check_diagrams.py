"""
Check that all labs and addons with the flag "need_uml" have up-to-date
UML diagrams by comparing SHA256 hashes
of the committed PNG file and a freshly generated PNG in isolation.

This ensures binary identity of diagram images — any difference in rendering,
Graphviz version, or layout will cause the check to fail.

Workflow:
1. For each lab or addon with the flag "need_uml" in project_config.json:
   - Copy the corresponding package to a temporary directory.
   - Generate a fresh description.png using the current code.
   - Compute SHA256 hash of both the committed and generated PNG.
   - Compare the hashes.
2. Exit with code 0 if all match, 1 otherwise.
"""

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

from quality_control.project_config import Addon, Lab, ProjectConfig

from admin_utils.constants import PROJECT_CONFIG_PATH
from admin_utils.uml.uml_diagrams_builder import (
    generate_lab_main_diagram,
    generate_package_diagram,
    subdirs_to_list,
)


def compute_png_hash(png_path: Path) -> str:
    """
    Compute a deterministic SHA256 hash from PNG.

    Args:
        png_path (Path): Path to the PNG file.

    Returns:
        str: SHA256 hex digest of the PNG file contents.
    """
    return hashlib.sha256(png_path.read_bytes()).hexdigest()


def check_lab_diagram(lab_info: Lab, root_dir: Path) -> bool:
    """
    Check a single lab's diagram by comparing PNG hashes.

    1. Locates the lab directory based on config info.
    2. Copies it to a temporary location to avoid side effects.
    3. Generates a fresh description.png from current code.
    4. Compares SHA256 hash of the committed PNG with the generated one.

    Args:
        lab_info (Lab): Lab entry from project_config.json.
        root_dir (Path): Root directory of the project (parent of lab folders).

    Returns:
        bool: True if hashes match, False if PNG is missing, generation fails,
            or hashes differ.
    """
    lab_name = lab_info.name
    lab_path = root_dir / lab_name

    committed_png = lab_path / "assets" / "description.png"
    if not committed_png.is_file():
        print(f"Missing committed diagram: {committed_png}")
        return False

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_lab = Path(tmp_dir) / lab_name
        shutil.copytree(lab_path, tmp_lab, dirs_exist_ok=True)

        if not generate_lab_main_diagram(tmp_lab):
            print(f"Failed to generate diagram for {lab_name}")
            return False

        generated_png = tmp_lab / "assets" / "description.png"
        if not generated_png.exists():
            print(f"Generated PNG not found: {generated_png}")
            return False

        committed_hash = compute_png_hash(committed_png)
        generated_hash = compute_png_hash(generated_png)

        if committed_hash != generated_hash:
            print(f"Diagram image differs: {committed_png}")
            print(f"  Committed hash: {committed_hash}")
            print(f"  Generated hash: {generated_hash}")
            return False

        print(f"Diagram image is up-to-date: {lab_name}")
        return True


def check_addon_diagram(addon: Addon, root_dir: Path) -> bool:
    """
    Check UML diagrams for an addon and its subpackages if they exist.

    Args:
        addon (Addon): Addon entry from project_config.json.
        root_dir (Path): Root directory of the project (parent of lab folders).

    Returns:
        bool: True if hashes match, False if PNG is missing, generation fails,
            or hashes differ.
    """
    addon_path = root_dir / addon.name

    if not addon_path.is_dir():
        print(f"Addon directory not found: {addon_path}")
        return False

    subdirs = subdirs_to_list(addon_path)
    if subdirs:
        all_ok = True
        for subdir in subdirs:
            png_path = subdir / "assets" / "description.png"
            if not png_path.is_file():
                print(f"Missing committed diagram: {png_path}")
                all_ok = False
                continue

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_subdir = Path(tmp_dir) / subdir.name
                shutil.copytree(subdir, tmp_subdir, dirs_exist_ok=True)
                generated_png = tmp_subdir / "assets" / "description.png"

                if not generate_package_diagram(tmp_subdir, generated_png):
                    print(f"Failed to generate diagram for {subdir.relative_to(root_dir)}")
                    all_ok = False
                    continue

                if compute_png_hash(png_path) != compute_png_hash(generated_png):
                    print(f"Diagram image differs: {png_path}")
                    all_ok = False

                print(f"Diagram image is up-to-date: {subdir.relative_to(root_dir)}")

        return all_ok

    committed_png = addon_path / "assets" / "description.png"
    if not committed_png.is_file():
        print(f"Missing committed diagram: {committed_png}")
        return False

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_addon = Path(tmp_dir) / addon.name
        shutil.copytree(addon_path, tmp_addon, dirs_exist_ok=True)
        generated_png = tmp_addon / "assets" / "description.png"

        if not generate_package_diagram(tmp_addon, generated_png):
            print(f"Failed to generate diagram for addon {addon.name}")
            return False

        if compute_png_hash(committed_png) != compute_png_hash(generated_png):
            print(f"Diagram image differs: {committed_png}")
            return False

        print(f"Diagram image is up-to-date: {addon.name}")
        return True


def main() -> None:
    """
    Entry point for the UML diagram consistency checker.

    Reads the project configuration from project_config.json,
    iterates over all registered labs, and verifies that each lab's
    UML diagram (represented by assets/description.png)
    is up-to-date with the current source code.

    Exits with code:
        0 — if all diagrams are present and up-to-date,
        1 — if any diagram is missing, invalid, or outdated.
    """
    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    root_dir = PROJECT_CONFIG_PATH.parent

    all_ok = True

    # Check labs
    target_labs = list(project_config.get_labs())
    for lab in target_labs:
        if not check_lab_diagram(lab, root_dir):
            all_ok = False

    # Check addons
    for addon in project_config.get_addons():
        if not addon.need_uml:
            continue
        if not check_addon_diagram(addon, root_dir):
            all_ok = False

    if not all_ok:
        print("\nTip: Run the UML generator locally and commit updated assets/description.png")
        print("Run: python -m admin_utils.uml.uml_diagrams_builder")
        sys.exit(1)

    print("\nAll diagrams are present and up-to-date")
    sys.exit(0)


if __name__ == "__main__":
    main()
