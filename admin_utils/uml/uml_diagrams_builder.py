"""
UML Diagram Generator for Labs and Core Utils Packages

Generates structural diagrams:
- For labs: based exclusively on main.py in each lab folder.
- For addons:
    - If the addon directory contains immediate subdirectories (excluding __pycache__,
      private (_*) and hidden (.*)), a diagram is generated for each such subdirectory.
    - Otherwise, a single diagram is generated for the addon directory itself.

Diagrams are either:
- Class diagram (if any .py file in the scope contains class definitions).
- Function diagram (otherwise).

All rendering is deterministic via Graphviz.

Requirements:
- Graphviz must be installed and available in PATH.
"""

import ast
import os
from pathlib import Path

from quality_control.cli_unifier import _run_console_tool, handles_console_error
from quality_control.project_config import Addon, Lab, ProjectConfig

from admin_utils.constants import PROJECT_CONFIG_PATH, PROJECT_ROOT


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


def get_python_files_in_package(package_dir: Path) -> list[Path]:
    """
    Return sorted list of all .py files in package_dir (recursive).

    Args:
        package_dir (Path): Path to the root directory of a Python package.

    Returns:
        list[Path]: Sorted list of Path objects pointing to all .py files
                    found recursively under package_dir.
    """
    return sorted(package_dir.rglob("*.py"))


def has_classes_in_files(py_files: list[Path]) -> bool:
    """
    Check if any of the given Python files contains a class definition.

    Args:
        py_files (list[Path]): List of paths to Python source files.

    Returns:
        bool: True if at least one file contains a top-level or nested class
              definition, False otherwise or if all files are
              unreadable or syntactically invalid.
    """
    for py_file in py_files:
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    return True
        except (SyntaxError, UnicodeDecodeError):
            continue
    return False


def extract_functions(py_file: Path) -> list[str]:
    """
    Extract all function names from a Python file.

    Args:
        py_file (Path): Path to a Python source file.

    Returns:
        list[str]: Sorted list of function names (instances of ast.FunctionDef).
                   Returns empty list on parse error.
    """
    functions = []
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
    except (SyntaxError, UnicodeDecodeError):
        pass
    return sorted(functions)


def extract_class_members(class_node: ast.ClassDef) -> tuple[list[str], list[str]]:
    """
    Extract field and method names from a ClassDef AST node.

    Args:
        class_node (ast.ClassDef): An AST node representing a Python class.

    Returns:
        tuple[list[str], list[str]]: A 2-tuple where:
            - First element is a sorted list of field names (from AnnAssign
              and Assign statements targeting simple names).
            - Second element is a sorted list of method names (from FunctionDef
              nodes directly in the class body).
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
            methods.add(item.name)

    return sorted(fields), sorted(methods)


def extract_classes_from_file(py_file: Path) -> list[dict]:
    """
    Extract class info (name, fields, methods) from a Python file.

    Args:
        py_file (Path): Path to a Python source file.

    Returns:
        list[dict]: List of dictionaries, each representing a class with keys:
            - "name" (str): Class name.
            - "fields" (list[str]): Sorted list of field names.
            - "methods" (list[str]): Sorted list of method names.
            The list is sorted by class name. Returns empty list on parse error.
    """
    classes = []
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                fields, methods = extract_class_members(node)
                classes.append({"name": node.name, "fields": fields, "methods": methods})
    except (SyntaxError, UnicodeDecodeError):
        pass
    return sorted(classes, key=lambda c: c["name"])


def generate_function_diagram_dot(
    py_files: list[Path], root_label: str, root_shape: str = "folder"
) -> str | None:
    """
    Generate function diagram DOT from a list of Python files.

    Args:
        py_files (list[Path]): List of paths to Python files to analyze.
        root_label (str): Label for the root node in the diagram.
        root_shape (str): Graphviz shape for the root node (default: "folder").

    Returns:
        str | None: Valid Graphviz DOT string describing a function diagram,
                    or None if no functions were found in any of the files.
    """
    functions = set()
    for f in py_files:
        functions.update(extract_functions(f))
    function_names = sorted(functions)
    if not function_names:
        return None

    lines = [
        "digraph Functions {",
        "  graph [ordering=out, rankdir=LR, nodesep=0.4, ranksep=0.6,"
        'bgcolor=white, size="10,5!", overlap=false, splines=true];',
        '  node [shape=box, style=filled, fillcolor="#E0F0FF", fontname="Arial"];',
        f'  main [label="{root_label}", shape={root_shape}, fillcolor="#FFE0E0"];',
    ]
    for func in function_names:
        lines.append(f'  "{func}" [label="{func}()"];')
        lines.append(f'  main -> "{func}";')
    lines.append("}")
    return "\n".join(lines) + "\n"


def collect_classes_and_inheritance(
    py_files: list[Path], include_inheritance: bool
) -> tuple[list[dict[str, str | list[str]]], set[tuple[str, str]]]:
    """
    Extract class info and inheritance relations from Python files.

    Args:
        py_files (list[Path]): List of paths to Python files to collect classes from.
        include_inheritance (bool): Whether to extract base class names for inheritance edges.

    Returns:
        tuple[list[dict[str, str | list[str]]], set[tuple[str, str]]]:
            - First element: list of class descriptors, each with keys:
                - "name" (str): class name,
                - "fields" (list[str]): field names defined at class level,
                - "methods" (list[str]): method names.
            - Second element: set of (child_class_name, parent_class_name) tuples
              representing direct inheritance relationships found in the same files.
    """
    classes: list[dict[str, str | list[str]]] = []
    inheritance_relations: set[tuple[str, str]] = set()

    for py_file in py_files:
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        class_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        for node in class_nodes:
            fields, methods = extract_class_members(node)
            classes.append({"name": node.name, "fields": fields, "methods": methods})

        if include_inheritance:
            for node in class_nodes:
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        inheritance_relations.add((node.name, base.id))

    return classes, inheritance_relations


def generate_class_diagram_dot(
    py_files: list[Path], include_inheritance: bool = False
) -> str | None:
    """
    Generate class diagram DOT from a list of Python files.

    Inheritance only works if the base class is defined
    in the same file or its name is available as a simple name (ast.Name).

    Args:
        py_files (list[Path]): List of paths to Python files to analyze.
        include_inheritance (bool): Whether to include inheritance edges
                                    between classes (default: False).

    Returns:
        str | None: Valid Graphviz DOT string describing a UML-like class diagram,
                    or None if no classes were found in any of the files.
    """
    all_classes, inheritance_relations = collect_classes_and_inheritance(
        py_files, include_inheritance
    )

    if not all_classes:
        return None

    all_classes = sorted(all_classes, key=lambda c: c["name"])
    lines = [
        "digraph UML {",
        "  graph [ordering=out, rankdir=BT, nodesep=0.5, ranksep=0.75, bgcolor=white,"
        'size="12,8!", overlap=false, splines=true];',
        '  node [shape=record, style=filled, fillcolor=white, fontname="Arial"];',
    ]

    for cls in all_classes:
        fields_str = "\\n".join(f"\u200b{f}" for f in cls["fields"]) if cls["fields"] else ""
        methods_str = "\\n".join(f"\u200b{m}()" for m in cls["methods"]) if cls["methods"] else ""
        label = (
            f"{{{cls['name']}|{fields_str}|{methods_str}}}"
            if cls["fields"]
            else f"{{{cls['name']}||{methods_str}}}"
        )
        lines.append(f'  "{cls["name"]}" [label="{label}"];')

    if include_inheritance:
        for child, parent in sorted(inheritance_relations):
            lines.append(f'  "{child}" -> "{parent}" [arrowhead=empty];')

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_dot_to_png(dot_content: str | None, output_png: Path) -> bool:
    """
    Render DOT content to PNG; return True on success.

    Args:
        dot_content (str | None): Graphviz DOT source as a string, or None.
        output_png (Path): Path where the resulting PNG image will be saved.

    Returns:
        bool: True if dot_content is not None, the temporary DOT file was written,
              the Graphviz `dot` command succeeded, and the output PNG exists;
              False otherwise.
    """
    if dot_content is None:
        return False

    output_dir = output_png.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    dot_path = output_dir / "temp.dot"

    try:
        dot_path.write_text(dot_content, encoding="utf-8")
        _, _, exit_code = _run_dot(dot_path, output_png)
        return exit_code == 0 and output_png.exists()
    finally:
        dot_path.unlink(missing_ok=True)


def generate_lab_main_diagram(lab_folder: Path) -> bool:
    """
    Generate UML diagram for a lab based solely on its main.py.

    Args:
        lab_folder (Path): Path to the lab directory.

    Returns:
        bool: True if diagram was successfully generated.
    """
    main_py = lab_folder / "main.py"
    if not main_py.exists():
        return False

    py_files = [main_py]
    assets_dir = lab_folder / "assets"
    png_path = assets_dir / "description.png"

    if has_classes_in_files(py_files):
        dot_content = generate_class_diagram_dot(py_files, include_inheritance=True)
    else:
        dot_content = generate_function_diagram_dot(py_files, root_label="main.py")

    return render_dot_to_png(dot_content, png_path)


def generate_package_diagram(package_dir: Path, output_png: Path) -> bool:
    """
    Generate UML diagram for a package (all .py files recursively).

    Inheritance across files is NOT analyzed.

    Args:
        package_dir (Path): Path to the package (e.g., core_utils/llm).
        output_png (Path): Full path to output PNG file.

    Returns:
        bool: True if diagram was successfully generated.
    """
    py_files = get_python_files_in_package(package_dir)
    if not py_files:
        return False

    if has_classes_in_files(py_files):
        dot_content = generate_class_diagram_dot(py_files, include_inheritance=True)
    else:
        dot_content = generate_function_diagram_dot(py_files, root_label=f"{package_dir.name}/")

    return render_dot_to_png(dot_content, output_png)


def process_lab(lab: Lab, root_dir: Path) -> None:
    """
    Process a single lab from config.

    Args:
        lab_info (Lab): Lab entry from project_config.json.
        root_dir (Path): The root directory of the project, used to resolve
                         the absolute path to the lab folder.
    """
    lab_name = lab.name
    lab_path = root_dir / lab_name
    if not lab_path.exists():
        print(f"Lab folder not found: {lab_path}")
        return
    print(f"\nProcessing {lab_name}...")
    success = generate_lab_main_diagram(lab_path)
    status = "generated successfully" if success else "failed"
    print(status)


def subdirs_to_list(addon_path: Path) -> list[Path]:
    """
    Make a list of subpackages' paths that are not private or hidden.

    Args:
        addon_path(Path): Path to the addon with subpackages to consider.

    Returns:
        list(str): The list of subpackages' paths.
    """
    return [
        item
        for item in addon_path.iterdir()
        if item.is_dir() and not item.name.startswith(("_", "."))
    ]


def process_addon(addon: Addon, root_dir: Path) -> None:
    """
    Process a single addon from config.

    Args:
        addon_info (dict): Addon entry from project_config.json.
        root_dir (Path): The root directory of the project, used to resolve
                         the absolute path to the addon folder.
    """
    if not addon.need_uml:
        return
    addon_name = addon.name
    addon_path = root_dir / addon_name
    if not addon_path.is_dir():
        print(f"Addon folder not found: {addon_path}")
        return

    subdirs = subdirs_to_list(addon_path)
    if subdirs:
        for subdir in subdirs:
            output_png = subdir / "assets" / "description.png"
            print(f"\nProcessing subpackage: {subdir.relative_to(root_dir)}...")
            success = generate_package_diagram(subdir, output_png)
            status = "generated successfully" if success else "failed"
            print(status)
    else:
        output_png = addon_path / "assets" / "description.png"
        print(f"\nProcessing leaf addon: {addon_name}...")
        success = generate_package_diagram(addon_path, output_png)
        status = "generated successfully" if success else "failed"
        print(status)


def main() -> None:
    """
    Main entry point: process labs and addons.
    """
    if not PROJECT_CONFIG_PATH.exists():
        print(f"Config file not found: {PROJECT_CONFIG_PATH}")
        return

    project_config = ProjectConfig(PROJECT_CONFIG_PATH)

    for lab in project_config.get_labs():
        process_lab(lab, PROJECT_ROOT)

    for addon in project_config.get_addons():
        if addon.need_uml:
            process_addon(addon, PROJECT_ROOT)


if __name__ == "__main__":
    main()
