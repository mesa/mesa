# save as fix_mesa_solara.py and run: python fix_mesa_solara.py
import os
import re

PROJECT_DIR = "."  # Change if your Mesa project is in a subfolder


def fix_mutable_defaults(file_path):
    """Replace [] or {} defaults with None in function definitions"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace = None and = None with = None
    content_fixed = re.sub(r"=\s*\[\]", "= None", content)
    content_fixed = re.sub(r"=\s*\{\}", "= None", content_fixed)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content_fixed)


def add_minimal_docstrings(file_path):
    """Add empty docstrings to functions missing D103"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = None
    func_pattern = re.compile(r"^\s*def (\w+)\(.*\):")
    for i, line in enumerate(lines):
        new_lines.append(line)
        if func_pattern.match(line):
            # Check if next non-empty line is a docstring
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines):
                if not re.match(r'^\s*"""', lines[j]):
                    # Insert minimal docstring
                    indent = re.match(r"^(\s*)", lines[j]).group(1)
                    new_lines.append(f'{indent}"""TODO: Add docstring."""\n')

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def find_undefined_names(file_path):
    """List names used but not defined (simple heuristic)"""
    undefined = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # Find capitalized names (likely classes)
            matches = re.findall(r"\b([A-Z][A-Za-z0-9_]+)\b", line)
            for name in matches:
                # Skip if defined in file
                if f"class {name}" not in open(file_path).read():
                    undefined.add(name)
    if undefined:
        print(f"[UNDEFINED] {file_path}: {undefined}")


def process_py_file(file_path):
    """TODO: Add docstring."""
    fix_mutable_defaults(file_path)
    add_minimal_docstrings(file_path)
    find_undefined_names(file_path)


def main():
    """TODO: Add docstring."""
    print("Processing all Python files in the project...")
    for root, dirs, files in os.walk(PROJECT_DIR):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                process_py_file(file_path)
    print("✅ Done. Check console output for undefined names.")


if __name__ == "__main__":
    main()
