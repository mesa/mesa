# Mesa Visualization Icons

This directory contains bundled SVG and PNG icons for agent visualization in Mesa.

## Available Icons

### smiley
- **Files**: `smiley.svg`, `smiley.png`
- **Color**: Gold (#FFD700)
- **Use Case**: Represent happy/positive agents or high values
- **Example**: Wealthy agents in economic models, successful agents in learning models

### sad_face
- **Files**: `sad_face.svg`, `sad_face.png`
- **Color**: Royal Blue (#28B4E7)
- **Use Case**: Represent unhappy/negative agents or low values
- **Example**: Poor agents in economic models, failed agents in learning models

### neutral_face
- **Files**: `neutral_face.svg`, `neutral_face.png`
- **Color**: Gray (#BCB3B3)
- **Use Case**: Represent neutral/baseline agents or medium values
- **Example**: Average agents in economic models, undecided agents in opinion dynamics

## Using Icons in Your Model

Icons are automatically rendered when you specify them in your agent portrayal function:

```python
def agent_portrayal(agent):
    return {
        "size": 24,
        "color": "blue",
        "icon": "smiley",           # Optional: icon name
        "icon_size": 24,            # Optional: defaults to "size"
    }

# Enable icon rendering
renderer = SpaceRenderer(model, backend="altair", icon_mode="force")
renderer.draw_agents(agent_portrayal=agent_portrayal)

Icon Rendering
Supported Backends
Altair: Interactive web-based visualization (recommended for icons)
Matplotlib: Static plotting (limited icon support)
Dependencies
Core: Pillow (PIL) - automatically included with Mesa
Optional: svglib + reportlab - for custom SVG icon support

pip install svglib reportlab