"""Icon cache for Mesa visualizations."""

from __future__ import annotations

import base64
import io
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image, ImageDraw

try:
    from reportlab.graphics import renderPM
    from svglib.svglib import svg2rlg

    SVG_SUPPORT = True
except ImportError:
    SVG_SUPPORT = False

from mesa.visualization.icons import get_icon_png, get_icon_svg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class IconCache:
    """Cache for rasterized icon images.

    Stores pre-rendered icons as data URLs or numpy arrays to avoid
    redundant rendering operations.
    For bundled icons, pre-rendered PNGs are used (no dependencies required).
    For custom SVG icons, svglib+reportlab can be used (optional, pip-installable).
    Falls back to colored circles if icons cannot be loaded.
    """

    def __init__(self, backend: Literal["matplotlib", "altair"] = "matplotlib"):
        """Initialize the icon cache.

        Args:
            backend: The visualization backend ("matplotlib" or "altair")
        """
        self.backend = backend
        self._cache = {}

    def get(self, icon_name: str | None, size: int) -> str | np.ndarray | None:
        """Get cached icon or return None if not cached.

        Args:
            icon_name: Name of the icon
            size: Size of the icon in pixels
        Returns:
            Cached icon data (data URL for altair, numpy array for matplotlib),
            or None if not cached
        """
        if icon_name is None:
            return None
        key = (icon_name, size)
        return self._cache.get(key)

    def get_or_create(self, icon_name: str, size: int) -> str | np.ndarray | None:
        """Get cached icon or create and cache it.

        Args:
            icon_name: Name of the icon to retrieve/create
            size: Size of the icon in pixels
        Returns:
            Icon data (data URL for altair, numpy array for matplotlib),
            or None if icon cannot be created
        """
        if icon_name is None:
            return None

        cached = self.get(icon_name, size)
        if cached is not None:
            return cached

        raster = self._rasterize_icon(icon_name, size)
        if raster is not None:
            self._cache[(icon_name, size)] = raster
        return raster

    def _rasterize_icon(self, icon_name: str, size: int) -> str | np.ndarray | None:
        """Rasterize icon to appropriate format for backend.

        Tries in order:
        1. Load pre-rendered PNG (for bundled icons, no dependencies)
        2. Convert SVG to PNG using svglib+reportlab (for custom SVGs, optional)
        3. Fall back to colored circle marker
        Args:
            icon_name: Name of the icon to rasterize
            size: Size of the icon in pixels
        Returns:
            Rasterized icon (data URL for altair, numpy array for matplotlib),
            or None if icon cannot be rasterized
        """
        # Strategy 1: Try to load pre-rendered PNG
        try:
            png_bytes = get_icon_png(icon_name, size)
            if png_bytes:
                img = Image.open(BytesIO(png_bytes))
                if img.size != (size, size):
                    img = img.resize((size, size), Image.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    png_bytes = buffer.getvalue()
                return self.from_png_bytes(png_bytes, self.backend)
        except (FileNotFoundError, ValueError, OSError):
            pass  # Try next strategy

        # Strategy 2: Try to convert SVG to PNG
        if SVG_SUPPORT:
            try:
                svg_text = get_icon_svg(icon_name)
                drawing = svg2rlg(BytesIO(svg_text.encode("utf-8")))

                if not drawing:
                    raise ValueError(f"Could not parse SVG: {icon_name}")

                scale = min(size / drawing.width, size / drawing.height)
                drawing.width = size
                drawing.height = size
                drawing.scale(scale, scale)

                png_bytes = renderPM.drawToString(drawing, fmt="PNG")
                return self.from_png_bytes(png_bytes, self.backend)
            except (FileNotFoundError, ValueError) as e:
                logger.debug(
                    f"SVG conversion failed for {icon_name}: {e}. Trying fallback."
                )
            except Exception as e:
                logger.warning(f"Unexpected error converting SVG {icon_name}: {e}")

        # Strategy 3: Fallback to colored circles
        return self._create_fallback_circle(icon_name, size)

    def _create_fallback_circle(self, icon_name: str, size: int) -> str | np.ndarray:
        """Create a colored circle fallback when icon cannot be loaded.

        Args:
            icon_name: Icon name (used to choose color)
            size: Size in pixels
        Returns:
            Fallback circle in appropriate format for backend
        """
        img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        color_map = {
            "smiley": (255, 215, 0, 255),
            "sad_face": (100, 149, 237, 255),
            "neutral_face": (169, 169, 169, 255),
        }
        color = color_map.get(icon_name, (100, 150, 255, 255))

        margin = size // 8
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=color,
            outline=None,
        )

        if self.backend == "altair":
            return self._to_data_url(img)
        return self._to_numpy(img)

    def _to_data_url(self, img: Image.Image) -> str:
        """Convert PIL Image to data URL for Altair.

        Args:
            img: PIL Image to convert
        Returns:
            Base64-encoded data URL string
        """
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _to_numpy(self, img: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array for Matplotlib.

        Args:
            img: PIL Image to convert
        Returns:
            RGBA numpy array
        """
        return np.asarray(img.convert("RGBA"))

    @classmethod
    def from_png_bytes(cls, png_bytes: bytes, backend: str) -> str | np.ndarray:
        """Convert PNG bytes to appropriate format for backend.

        Args:
            png_bytes: Raw PNG image bytes
            backend: Target backend ("matplotlib" or "altair")

        Returns:
            Converted image (data URL for altair, numpy array for matplotlib)
        """
        if backend == "altair":
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        elif backend == "matplotlib":
            img = Image.open(BytesIO(png_bytes)).convert("RGBA")
            return np.asarray(img)
        raise ValueError(f"Unsupported backend: {backend}")

    def clear(self):
        """Clear all cached icons."""
        self._cache.clear()

    def __len__(self):
        """Get number of cached icons.

        Returns:
            Number of cached icon entries
        """
        return len(self._cache)
