# Mesa Agent Icon Library

A collection of minimal, performance-optimized icons for agent-based model visualization.

## Overview

This directory contains bundled icons (both SVG and pre-rendered PNG) that can be used to represent agents in Mesa visualizations. The icons are designed to be lightweight, customizable, and work out of the box with no extra dependencies.

## Usage

### Python

```python
from mesa.visualization import icons

# List all available icons
icon_names = icons.list_icons()
print(icon_names)  # ['smiley', 'sad_face', 'neutral_face', ...]

# Get pre-rendered PNG bytes (no dependencies required)
png_bytes = icons.get_icon_png("smiley", size=32)

# Get SVG content as string (for custom styling)
svg_content = icons.get_icon_svg("smiley")

# Use with namespace prefix (optional)
svg_content = icons.get_icon_svg("mesa:smiley")
```

### Integration with Visualization

**Bundled Icons (Recommended):**
- Pre-rendered PNG files are included and work with **no extra dependencies**
- Simply use `icon="smiley"` in your agent portrayal
- Icons are automatically loaded and cached

**Custom SVG Icons (Optional):**
- If you have custom SVG icons, you can use `svglib` and `reportlab` (pip-installable, pure Python)
- Install with: `pip install svglib reportlab`
- The system will automatically fall back to colored circle markers if SVG conversion is not available

## Design Guidelines

### File Naming
- **Lowercase with underscores**: `person.svg`, `happy_face.svg`
- **Icon name = filename without extension**: `person.svg` → `"person"`
- **Descriptive and concise**: Prefer `arrow_up` over `arr_u`

### SVG Standards
- **ViewBox**: Use `0 0 32 32` for consistency
- **Dynamic coloring**: Use `fill="currentColor"` to enable programmatic color control
- **Minimal paths**: Keep SVG markup simple for performance
- **No embedded styles**: Avoid `<style>` tags or inline CSS

### Example Icon

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <circle cx="16" cy="16" r="14" fill="currentColor"/>
</svg>
```

## Performance Considerations

- Icons are loaded via `importlib.resources` for efficient bundling
- Pre-rendered PNGs are cached for fast access
- Small file sizes (<2KB recommended) ensure fast loading
- No system dependencies required for bundled icons

## Adding New Icons

1. Create an SVG file following the guidelines above
2. Place it in this directory
3. Icon automatically becomes available via `icons.get_icon_svg()`
4. No code changes required - icons are discovered at runtime

## License

Icons in this directory are part of the Mesa project and follow the same Apache 2.0 license.
