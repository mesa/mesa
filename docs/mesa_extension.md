# Mesa Extensions Overview

This contains an overview of Mesa Extensions. Mesa's extensibility is a key feature that allows users to enhance functionality, improve scalability, and foster innovation in agent-based modeling.


## Mesa-Geo 🌍

**Field:** Geographic Information Systems (GIS)

---
**Description:**
Mesa-Geo is an extension of the Mesa framework designed to facilitate working with geographic data in agent-based modeling. It introduces a **GeoSpace** to host **GeoAgents**, which are enhanced agents that include a `geometry` attribute ([a Shapely object](https://shapely.readthedocs.io/en/latest/manual.html)) and a `crs` attribute (Coordinate Reference System). These attributes enable the integration of geographic and spatial data into simulations. Geometries can be defined manually using Shapely or imported from various sources, such as vector data files (e.g., shapefiles), GeoJSON objects, or GeoPandas GeoDataFrames.

---
**Key Features:**
- **Spatial Reference Systems Support:** Mesa-Geo handles coordinate reference systems (CRS), which is essential for working with geographic data in various projections.
- **Geometric Operations Support:** Mesa-Geo utilizes Shapely, which provides robust tools for creating and manipulating geometric shapes like points, polygons, and lines.
- **Topological Operations Support:** Functions for analyzing spatial relationships between geometries.

---
**Author(s):** Wang Boyu

---
**Additional Resources:**
For more information, visit the official [Mesa-Geo repository](https://github.com/mesa/mesa-geo?tab=readme-ov-file).

---

## Mesa Examples 📊

**Description:**
Mesa Examples provide a collection of models and use cases demonstrating the features and capabilities of the Mesa framework for agent-based modeling. These examples include core and user-submitted models covering a variety of domains like grid spaces, networks, visualization, and GIS.

---

**Key Features:**
- **Core Examples:** Fully tested and updated models included directly with the Mesa framework.
- **User Examples:** Community-contributed models showcasing advanced and diverse use cases.
- **Extensive Coverage:** Examples for grid spaces, GIS integration, networks, visualization, and more.
- **Easy Access:** Available directly from the Mesa package or via installation from the repository.

---

**Author(s):** Contributions from the Mesa developer community.

---

**Examples Include:**
- **Grid Space:** Models like Bank Reserves, Conway’s Game of Life, and Forest Fire.
- **GIS:** GeoSchelling Models, Urban Growth, and Population Models.
- **Network:** Boltzmann Wealth Model and Ant System for the Traveling Salesman Problem.
- **Visualization:** Charting tools and grid displays.

---

**For More Information:**
For more Detail, Visit the [Mesa Examples Repository](https://github.com/mesa/mesa/tree/main/mesa/examples).

---

## **Mesa-Frames** 🚀

**Description:**
Mesa-Frames is an extension of the Mesa framework designed to handle complex simulations with thousands of agents. By utilizing DataFrames (pandas or Polars), it enhances scalability and performance while maintaining a syntax similar to Mesa.

---

**Key Features:**
- **Enhanced Performance:** Uses DataFrames for SIMD processing and vectorized functions to speed up simulations.
- **Backend Support:** Supports `pandas` (ease of use) and `Polars` (performance innovations with Rust-based backend).
- **Seamless Integration:** Maintains a similar API and functionality as the base Mesa framework for easier adoption.
- **In-Place Operations:** Functional programming and fast memory-efficient copy methods.
- **Future Plans:** GPU functionality, automatic model vectorization, and backend-independent AgentSet class.

---

**Usage:**
- Define agents using `AgentSetPandas` or `AgentSetPolars`.
- Implement models by subclassing `ModelDF`.
- Perform vectorized operations to enhance simulation performance.

---

**Author(s):**
Developed and maintained by the Mesa development community.

---

**License:**
Distributed under the MIT License.

---

**More Information:**
Visit the [GitHub Repository](https://github.com/mesa/mesa-frames).

---
