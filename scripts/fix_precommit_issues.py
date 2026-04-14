"""Fix common pre-commit issues for this repository.

This script:
1. Hoists module-level imports to the top of Python files when Ruff E402 is triggered.
2. Rewrites Ruff config sections from nested ``[tool.ruff.lint]`` form to top-level
   ``[tool.ruff]`` keys for compatibility with older hook versions.
3. Removes invalid Vale style references when the configured style is not available.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
VALE_CONFIG = ROOT / ".vale.ini"
PYTHON_ROOTS = (ROOT / "mesa", ROOT / "tests")


def fix_pyproject() -> bool:
    """Rewrite Ruff config headers to the flat layout expected by old Ruff hooks."""
    text = PYPROJECT.read_text(encoding="utf-8")
    updated = text.replace("[tool.ruff.lint.pydocstyle]", "[tool.ruff.pydocstyle]")
    updated = updated.replace("[tool.ruff.lint]", "[tool.ruff]")

    if updated == text:
        return False

    PYPROJECT.write_text(updated, encoding="utf-8", newline="\n")
    return True


def fix_vale() -> bool:
    """Remove invalid Google style references when only local Mesa styles exist."""
    text = VALE_CONFIG.read_text(encoding="utf-8")
    lines = text.splitlines()

    fixed_lines: list[str] = []
    changed = False

    for line in lines:
        stripped = line.strip()
        if stripped == "Packages = Google":
            changed = True
            continue
        if stripped.startswith("BasedOnStyles ="):
            styles = [style.strip() for style in stripped.split("=", 1)[1].split(",")]
            styles = [style for style in styles if style and style != "Google"]
            replacement = ", ".join(styles) or "Mesa"
            new_line = f"BasedOnStyles = {replacement}"
            fixed_lines.append(new_line)
            changed |= new_line != line
            continue
        if stripped.startswith("Google."):
            changed = True
            continue
        fixed_lines.append(line)

    if not changed:
        return False

    VALE_CONFIG.write_text(
        "\n".join(fixed_lines) + "\n", encoding="utf-8", newline="\n"
    )
    return True


def _module_docstring_end(tree: ast.Module) -> int:
    if not tree.body:
        return 0
    first = tree.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(getattr(first, "value", None), ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.end_lineno or first.lineno
    return 0


def _header_end_line(tree: ast.Module) -> int:
    """Find the insertion point for top-level imports."""
    docstring_end = _module_docstring_end(tree)
    header_end = docstring_end

    for node in tree.body:
        if node.lineno <= docstring_end:
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            header_end = node.end_lineno or node.lineno
            continue
        if (
            isinstance(node, ast.Import | ast.ImportFrom)
            and node.lineno <= header_end + 2
        ):
            header_end = node.end_lineno or node.lineno
            continue
        break

    return header_end


def _offending_imports(tree: ast.Module) -> list[ast.stmt]:
    """Return imports that appear after other top-level statements."""
    seen_non_import = False
    offenders: list[ast.stmt] = []

    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.Import | ast.ImportFrom):
            if seen_non_import:
                offenders.append(node)
        else:
            seen_non_import = True

    return offenders


def fix_python_imports(path: Path) -> bool:
    """Move module-level imports to the top of the file."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    offenders = _offending_imports(tree)
    if not offenders:
        return False

    lines = source.splitlines()
    import_blocks: list[str] = []
    skip_lines: set[int] = set()

    for node in offenders:
        start = node.lineno - 1
        end = node.end_lineno or node.lineno
        import_blocks.append("\n".join(lines[start:end]).strip("\n"))
        skip_lines.update(range(start, end))

    kept_lines = [line for idx, line in enumerate(lines) if idx not in skip_lines]
    insertion_line = _header_end_line(tree)
    insertion_index = insertion_line

    while (
        insertion_index < len(kept_lines) and kept_lines[insertion_index].strip() == ""
    ):
        insertion_index += 1

    block = "\n\n".join(part for part in import_blocks if part).strip()
    if not block:
        return False

    new_lines = kept_lines[:insertion_index]
    if new_lines and new_lines[-1].strip():
        new_lines.append("")
    new_lines.extend(block.splitlines())
    new_lines.append("")
    new_lines.extend(kept_lines[insertion_index:])

    updated = "\n".join(new_lines).rstrip() + "\n"
    if updated == source:
        return False

    path.write_text(updated, encoding="utf-8", newline="\n")
    return True


def fix_python_tree() -> list[Path]:
    """Handle fix python tree."""
    changed: list[Path] = []
    for root in PYTHON_ROOTS:
        for path in root.rglob("*.py"):
            if fix_python_imports(path):
                changed.append(path.relative_to(ROOT))
    return changed


def main() -> None:
    """Handle main."""
    changed = []
    if fix_pyproject():
        changed.append("pyproject.toml")
    if fix_vale():
        changed.append(".vale.ini")
    changed.extend(str(path) for path in fix_python_tree())

    if changed:
        print("Updated files:")
        for path in changed:
            print(f"- {path}")
    else:
        print("No changes needed.")

    print("\nNext steps:")
    print("1. pre-commit run --all-files")
    print("2. If Ruff is installed locally, run: ruff format .")


if __name__ == "__main__":
    main()
