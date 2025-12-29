Software Renderer (Python)

A from-scratch 3D software renderer written in Python that implements the core stages of a real-time graphics pipeline without relying on OpenGL, DirectX, or engine-level rendering APIs.

The project is focused on understanding rendering fundamentals, system design, and performance tradeoffs, rather than visual fidelity or hardware acceleration.

Overview

This renderer manually performs all major steps required to display 3D geometry on a 2D screen, including camera transforms, perspective projection, triangle rasterization, and basic lighting.

All math, data handling, and rendering logic is implemented explicitly in Python using NumPy for numerical operations and Pygame for windowing and input.

Features

Custom FPS-style camera

Mouse-controlled yaw and pitch

Horizontal-only forward movement

World → camera space transformations

Perspective projection

Configurable field of view and aspect ratio

Proper perspective divide

Rendering pipeline

Vertex transformation and projection

Back-face culling

Triangle sorting using the painter’s algorithm

Screen-space triangle drawing

OBJ model support

Custom OBJ parser

Vertex normalization

Degenerate triangle detection and removal

UV coordinate handling

Texture mapping

Raw byte-based texture loading

UV sampling and per-face color lookup

Lighting

Simple diffuse lighting using surface normals and dot products

Ambient term for baseline illumination

Profiling & debugging

Built-in profiler for timing pipeline stages

Accumulated performance reports

Debug visualization utilities using Matplotlib

.
├── Camera.py              # Camera movement, orientation, and projection
├── renderable_object.py   # Mesh representation and OBJ loading
├── texture.py             # Texture loading and UV sampling
├── profiler.py            # Lightweight profiling utilities
├── debug.py               # Debug visualization helpers
├── transform.py           # General-purpose 3D transform system
├── kept.py                # Main render loop and scene setup
└── resources/             # OBJ models and texture data
