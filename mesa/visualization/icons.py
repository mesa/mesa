"""Bundled icon access helpers for Mesa visualization.

Provides functions to access bundled icon files (SVG and PNG).
"""

from __future__ import annotations

import importlib.resources

ICONS_SUBDIR = "icons"
SVG_EXT = ".svg"
PNG_EXT = ".png"


def _icons_package_root():
    """Return path to the icons subdirectory."""
    return importlib.resources.files(__package__).joinpath(ICONS_SUBDIR)


def list_icons() -> list[str]:
    """Return sorted list of available icon basenames (without extension)."""
    root = _icons_package_root()
    names = set()
    for item in root.iterdir():
        if item.is_file():
            name = item.name.lower()
            if name.endswith(SVG_EXT):
                names.add(item.name[: -len(SVG_EXT)])
            elif name.endswith(PNG_EXT):
                base = item.name[: -len(PNG_EXT)]
                names.add(base.rsplit("_", 1)[0] if "_" in base else base)
    return sorted(names)


def get_icon_svg(name: str) -> str:
    """Return SVG text for a bundled icon.

    Args:
        name: Icon name (e.g., "smiley" or "mesa:smiley")

    Returns:
        SVG content as string
    Raises:
        FileNotFoundError: If icon not found
    """
    name = name.split(":", 1)[-1]
    svg_path = _icons_package_root().joinpath(f"{name}{SVG_EXT}")
    if not svg_path.exists():
        raise FileNotFoundError(f"Icon not found: {name}")
    return svg_path.read_text(encoding="utf-8")


def get_icon_png(name: str, size: int) -> bytes:
    """Return pre-rendered PNG bytes for a bundled icon.

    Tries: exact size match, unsized file, then any size variant.

    Args:
        name: Icon name (e.g., "smiley" or "mesa:smiley")
        size: Desired icon size in pixels

    Returns:
        PNG image bytes

    Raises:
        FileNotFoundError: If no PNG found
    """
    name = name.split(":", 1)[-1]  # Remove optional "mesa:" prefix
    root = _icons_package_root()

    # Try exact size, simple name, then any variant
    candidates = [
        f"{name}_{size}{PNG_EXT}",
        f"{name}{PNG_EXT}",
    ]

    for candidate in candidates:
        path = root.joinpath(candidate)
        if path.exists():
            return path.read_bytes()

    # Try any size variant
    for item in root.iterdir():
        if (
            item.is_file()
            and item.name.startswith(f"{name}_")
            and item.name.endswith(PNG_EXT)
        ):
            return item.read_bytes()

    raise FileNotFoundError(f"PNG icon not found: {name} (size {size})")
