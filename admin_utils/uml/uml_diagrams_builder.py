"""
UML Diagram Generator for Labs

Generates structural diagrams based exclusively on main.py in each lab:
- Class diagrams (via AST) if main.py contains class definitions.
- Function diagrams (via AST) if main.py contains only functions.

Workflow:
1. Reads only main.py from the lab folder.
2. Uses AST to detect presence of classes.
3. Generates a deterministic DOT representation:
   - For classes: shows class name, fields, methods.
   - For functions: shows function names as nodes linked from main.py.
4. Renders DOT to assets/description.png using Graphviz (dot).

Requirements:
- Graphviz must be installed and available in PATH.
"""

import ast
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# pylint: disable=wrong-import-position
from quality_control.cli_unifier import _run_console_tool, handles_console_error

from admin_utils.constants import PROJECT_CONFIG_PATH

# pylint: enable=wrong-import-position


@handles_console_error()
def _run_dot(input_path: Path, output_path: Path) -> tuple[str, str, int]:
    """
    Render a DOT file to PNG using Graphviz dot command with deterministic layout.

    Args:
        input_path (Path): Path to the input DOT file.
        output_path (Path): Path where the resulting PNG image will be saved.

    Returns:
        tuple[str, str, int]: stdout, stderr, and exit code of the dot process.
    """
    return _run_console_tool(
        "dot",
        [
            "-Tpng",
            "-Gid=uml_diagram",
            "-Gdpi=96",
            str(input_path),
            "-o",
            str(output_path),
        ],
        env={**os.environ, "GVDETERMINISTIC": "1"},
    )


def extract_functions(py_file: Path) -> list[str]:
    """
    Parses the given Python file and collects names of all top-level
    function definitions (excluding nested or lambda functions and dunder methods).

    Args:
        py_file (Path): Path to the Python source file.

    Returns:
        list[str]: Sorted list of function names defined in the file.
    """
    functions = []
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not (node.name.startswith("__") and node.name.endswith("__")):
                    functions.append(node.name)
    except (SyntaxError, UnicodeDecodeError):
        pass
    return sorted(functions)


def generate_function_diagram_dot_from_main(lab_folder: Path) -> str | None:
    """
    Generate DOT content for function-level diagram based only on main.py.

    Args:
        lab_folder (Path): Path to the lab directory.

    Returns:
        str | None: DOT content as string, or None if main.py is missing or has no functions.
    """
    main_py = lab_folder / "main.py"
    if not main_py.exists():
        return None

    functions = extract_functions(main_py)
    if not functions:
        return None

    lines = [
        "digraph Functions {",
        "  graph [ordering=out, rankdir=LR, nodesep=0.4, ranksep=0.6,"
        'bgcolor=white, size="10,5!", overlap=false, splines=true];',
        '  node [shape=box, style=filled, fillcolor="#E0F0FF", fontname="Arial"];',
        '  main [label="main.py", shape=folder, fillcolor="#FFE0E0"];',
    ]

    for func in functions:
        lines.append(f'  "{func}" [label="{func}()"];')
        lines.append(f'  main -> "{func}";')

    lines.append("}")
    return "\n".join(lines) + "\n"


def generate_module_diagram(lab_folder: Path, output_dir: Path) -> bool:
    """
    Generate function-level diagram PNG from main.py.

    Args:
        lab_folder (Path): Path to the lab directory.
        output_dir (Path): Directory to save description.png.

    Returns:
        bool: True if PNG was saved successfully.
    """
    dot_content = generate_function_diagram_dot_from_main(lab_folder)
    if dot_content is None:
        return False

    dot_path = output_dir / "temp.dot"
    png_path = output_dir / "description.png"

    try:
        dot_path.write_text(dot_content, encoding="utf-8")
        _, _, exit_code = _run_dot(dot_path, png_path)
        return exit_code == 0 and png_path.exists()
    finally:
        dot_path.unlink(missing_ok=True)


def has_classes_in_main(lab_folder: Path) -> bool:
    """
    Check if main.py in the lab contains any class definitions.

    Args:
        lab_folder (Path): Path to the lab directory.

    Returns:
        bool: True if main.py exists and contains at least one class, False otherwise.
    """
    main_py = lab_folder / "main.py"
    if not main_py.exists():
        return False
    try:
        tree = ast.parse(main_py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                return True
    except (SyntaxError, UnicodeDecodeError):
        pass
    return False


def _extract_class_members(class_node: ast.ClassDef) -> tuple[list[str], list[str]]:
    """
    Extract field and method names from a ClassDef AST node.

    Args:
        class_node (ast.ClassDef): The AST node representing a class definition.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two sorted lists
        field names (attributes assigned in class body)
        and method names (excluding dunder methods)
    """
    fields = set()
    methods = set()

    for item in class_node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            fields.add(item.target.id)
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    fields.add(target.id)
        elif isinstance(item, ast.FunctionDef):
            if not (item.name.startswith("__") and item.name.endswith("__")):
                methods.add(item.name)

    return sorted(fields), sorted(methods)


def extract_classes_from_main(main_py: Path) -> list[dict]:
    """
    Extract class definitions from main.py for UML diagram.

    Each class dict contains:
    - 'name': str
    - 'fields': sorted list of field names
    - 'methods': sorted list of method names (excluding __dunder__)

    Args:
        main_py (Path): Path to main.py.

    Returns:
        list[dict]: List of class info dictionaries.
    """
    classes = []
    try:
        tree = ast.parse(main_py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                fields, methods = _extract_class_members(node)
                classes.append(
                    {
                        "name": node.name,
                        "fields": fields,
                        "methods": methods,
                    }
                )
    except (SyntaxError, UnicodeDecodeError):
        pass
    return sorted(classes, key=lambda c: c["name"])


def generate_class_diagram_dot_from_main(lab_folder: Path) -> str | None:
    """
    Generate deterministic DOT content for class diagram based only on main.py.

    Args:
        lab_folder (Path): Path to the lab directory.

    Returns:
        str | None: DOT content as string, or None if main.py is missing or has no classes.
    """
    main_py = lab_folder / "main.py"
    if not main_py.exists():
        return None

    classes = extract_classes_from_main(main_py)
    if not classes:
        return None

    lines = [
        "digraph UML {",
        "  graph [ordering=out, rankdir=BT, nodesep=0.5, ranksep=0.75, bgcolor=white,"
        'size="12,8!", overlap=false, splines=true];',
        '  node [shape=record, style=filled, fillcolor=white, fontname="Arial"];',
    ]

    relations = set()
    try:
        tree = ast.parse(main_py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        parent_name = base.id
                        relations.add((node.name, parent_name))
    except (SyntaxError, UnicodeDecodeError):
        pass

    for cls in classes:
        fields_str = "\\n".join(f"\u200b{f}" for f in cls["fields"]) if cls["fields"] else ""
        methods_str = "\\n".join(f"\u200b{m}()" for m in cls["methods"]) if cls["methods"] else ""

        if not cls["fields"]:
            label = f"{{{cls['name']}||{methods_str}}}"
        else:
            label = f"{{{cls['name']}|{fields_str}|{methods_str}}}"

        lines.append(f'  "{cls["name"]}" [label="{label}"];')

    for child, parent in sorted(relations):
        lines.append(f'  "{child}" -> "{parent}" [arrowhead=empty];')

    lines.append("}")
    return "\n".join(lines) + "\n"


def generate_class_diagram(lab_folder: Path, output_dir: Path) -> bool:
    """
    Generate UML class diagram PNG from main.py.

    Args:
        lab_folder (Path): Path to the lab directory.
        output_dir (Path): Directory to save description.png.

    Returns:
        bool: True if PNG was saved successfully.
    """
    dot_content = generate_class_diagram_dot_from_main(lab_folder)
    if dot_content is None:
        return False

    dot_path = output_dir / "temp.dot"
    png_path = output_dir / "description.png"

    try:
        dot_path.write_text(dot_content, encoding="utf-8")
        _, _, exit_code = _run_dot(dot_path, png_path)
        return exit_code == 0 and png_path.exists()
    finally:
        dot_path.unlink(missing_ok=True)


def generate_uml_diagrams(lab_folder: Path) -> bool:
    """
    Generate appropriate UML diagram for a lab based on main.py content.

    Args:
        lab_folder (Path): Path to the lab directory.

    Returns:
        bool: True if diagram generation succeeded, otherwise False.
    """
    lab_folder = Path(lab_folder).resolve()
    assets_dir = lab_folder / "assets"
    output_dir = assets_dir if assets_dir.is_dir() else lab_folder
    output_dir.mkdir(exist_ok=True)

    if has_classes_in_main(lab_folder):
        return generate_class_diagram(lab_folder, output_dir)
    return generate_module_diagram(lab_folder, output_dir)


def main() -> None:
    """
    Generate diagrams for all labs in project_config.json.

    Reads the project configuration, iterates through all registered labs,
    and triggers diagram generation for each one. Skips missing lab folders.
    """
    if not PROJECT_CONFIG_PATH.exists():
        print(f"Config file not found: {PROJECT_CONFIG_PATH}")
        return

    with open(PROJECT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    root_dir = PROJECT_CONFIG_PATH.parent
    labs = config.get("labs", [])

    print(f"Found {len(labs)} labs in config")

    for lab_info in labs:
        lab_name = lab_info["name"]

        if not lab_name.startswith("lab_"):
            continue

        lab_path = root_dir / lab_name
        if not lab_path.exists():
            print(f"Lab folder not found: {lab_path}")
            continue

        print(f"\nProcessing {lab_name}...")
        if generate_uml_diagrams(lab_path):
            print("Diagram generated successfully")
        else:
            print("Failed to generate diagram")


if __name__ == "__main__":
    main()
