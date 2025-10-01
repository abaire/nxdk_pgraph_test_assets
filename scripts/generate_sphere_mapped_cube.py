#!/usr/bin/env python3

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

# ruff: noqa: T201 `print` found


def power_of_two(value: str) -> int:
    """Custom argparse type for power-of-two integers."""
    try:
        ivalue = int(value)
    except ValueError:
        msg = f"'{value}' is not an integer."
        raise argparse.ArgumentTypeError(msg) from None
    if ivalue <= 0 or (ivalue & (ivalue - 1)) != 0:
        msg = f"'{value}' is not a positive power of two."
        raise argparse.ArgumentTypeError(msg)
    return ivalue


def generate_cube_data(
    size: float = 2.0, *, ccw: bool = True, triangles: bool = False
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[int]]:
    """
    Generates vertex, UV, index, and normal data for a cube.
    Winding order is efficiently handled by reordering indices.
    """
    s = size / 2.0  # Half-size for coordinates from -s to +s

    # Define a single, canonical set of vertices in CCW order.
    vertices: list[list[float]] = [
        # +Z (Front) face
        [-s, -s, s],
        [s, -s, s],
        [s, s, s],
        [-s, s, s],
        # -Z (Back) face
        [s, -s, -s],
        [-s, -s, -s],
        [-s, s, -s],
        [s, s, -s],
        # +X (Right) face
        [s, -s, s],
        [s, -s, -s],
        [s, s, -s],
        [s, s, s],
        # -X (Left) face
        [-s, -s, -s],
        [-s, -s, s],
        [-s, s, s],
        [-s, s, -s],
        # +Y (Top) face
        [-s, s, s],
        [s, s, s],
        [s, s, -s],
        [-s, s, -s],
        # -Y (Bottom) face
        [-s, -s, -s],
        [s, -s, -s],
        [s, -s, s],
        [-s, -s, s],
    ]

    # Define the conceptual 4x3 UV layout.
    base_uvs: list[list[float]] = [
        # +Z (Front)
        [0.25, 1 / 3],
        [0.50, 1 / 3],
        [0.50, 2 / 3],
        [0.25, 2 / 3],
        # -Z (Back)
        [0.75, 1 / 3],
        [1.00, 1 / 3],
        [1.00, 2 / 3],
        [0.75, 2 / 3],
        # +X (Right)
        [0.50, 1 / 3],
        [0.75, 1 / 3],
        [0.75, 2 / 3],
        [0.50, 2 / 3],
        # -X (Left)
        [0.00, 1 / 3],
        [0.25, 1 / 3],
        [0.25, 2 / 3],
        [0.00, 2 / 3],
        # +Y (Top)
        [0.25, 2 / 3],
        [0.50, 2 / 3],
        [0.50, 1.0],
        [0.25, 1.0],
        # -Y (Bottom)
        [0.25, 0.0],
        [0.50, 0.0],
        [0.50, 1 / 3],
        [0.25, 1 / 3],
    ]

    # Remap the V-coordinate to fit the 4:3 layout into a 1:1 square texture space.
    uvs: list[list[float]] = [[u, (v * 0.75) + 0.125] for u, v in base_uvs]

    base_normals: list[list[float]] = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
    normals: list[list[float]] = [n for n in base_normals for _ in range(4)]

    indices: list[int] = []
    for i in range(6):
        base = i * 4
        if triangles:
            if ccw:  # Counter-Clockwise Triangles (Default)
                indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
            else:  # Clockwise Triangles
                indices.extend([base, base + 2, base + 1, base, base + 3, base + 2])
        elif ccw:  # Counter-Clockwise Quad (Default)
            indices.extend([base, base + 1, base + 2, base + 3])
        else:  # Clockwise Quad
            indices.extend([base, base + 3, base + 2, base + 1])

    return vertices, normals, uvs, indices


def generate_normal_map(filename: str = "cube_normal_map.png", *, texture_size: int = 256, size: float = 2.0) -> None:
    """Generates and saves an object-relative normal map for a cube."""
    width, height = texture_size, texture_size
    s = size / 2.0
    image_data = np.zeros((height, width, 3), dtype=np.uint8)

    face_definitions = [
        {"uv_min": (0.25, 1 / 3), "uv_max": (0.5, 2 / 3), "map": lambda u, v: [u * size - s, v * size - s, s]},
        {"uv_min": (0.75, 1 / 3), "uv_max": (1.0, 2 / 3), "map": lambda u, v: [s - u * size, v * size - s, -s]},
        {"uv_min": (0.5, 1 / 3), "uv_max": (0.75, 2 / 3), "map": lambda u, v: [s, v * size - s, s - u * size]},
        {"uv_min": (0.0, 1 / 3), "uv_max": (0.25, 2 / 3), "map": lambda u, v: [-s, v * size - s, u * size - s]},
        {"uv_min": (0.25, 2 / 3), "uv_max": (0.5, 1.0), "map": lambda u, v: [u * size - s, s, s - v * size]},
        {"uv_min": (0.25, 0.0), "uv_max": (0.5, 1 / 3), "map": lambda u, v: [u * size - s, -s, v * size - s]},
    ]
    for y in range(height):
        for x in range(width):
            u_texture = x / (width - 1)
            v_texture = 1.0 - (y / (height - 1))

            # Remap the square texture's v-coordinate to the 4:3 layout space.
            v_layout = (v_texture - 0.125) / 0.75

            pos_on_cube = None
            if 0.0 <= v_layout <= 1.0:
                for face in face_definitions:
                    min_u, min_v = face["uv_min"]
                    max_u, max_v = face["uv_max"]
                    if min_u <= u_texture < max_u and min_v <= v_layout < max_v:
                        norm_u = (u_texture - min_u) / (max_u - min_u)
                        norm_v = (v_layout - min_v) / (max_v - min_v)
                        pos_on_cube = np.array(face["map"](norm_u, norm_v))
                        break
            if pos_on_cube is not None:
                norm = pos_on_cube / np.linalg.norm(pos_on_cube)
                color = (norm * 0.5 + 0.5) * 255
                image_data[y, x] = color.astype(np.uint8)

    img = Image.fromarray(image_data, "RGB")
    img.save(filename)
    print(f"✅ Normal map saved as '{filename}'")


def write_cpp_header(
    filename: str,
    vertices: list[list[float]],
    normals: list[list[float]],
    uvs: list[list[float]],
    indices: list[int],
    size: float,
    *,
    ccw: bool,
    triangles: bool,
) -> None:
    """Writes the geometry data to a C++ header file using raw arrays."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("// Generated by a Python script. Do not edit manually.\n")
        f.write(f"// Geometry for a cube of size {size}x{size}x{size}, centered at the origin.\n")
        f.write(f"// Winding order for front faces: {'Counter-Clockwise' if ccw else 'Clockwise'}.\n\n")
        f.write("#pragma once\n\n")
        f.write("#include <cstdint>\n\n")

        f.write(f"static constexpr float CUBE_SIZE = {size:.2f}f;\n\n")
        f.write("static constexpr float CUBE_VERTICES[24][3] = {\n")
        for v in vertices:
            f.write(f"    {{ {v[0]:.2f}f, {v[1]:.2f}f, {v[2]:.2f}f }},\n")
        f.write("};\n\n")
        f.write("static constexpr float CUBE_NORMALS[24][3] = {\n")
        for n in normals:
            f.write(f"    {{ {n[0]:.2f}f, {n[1]:.2f}f, {n[2]:.2f}f }},\n")
        f.write("};\n\n")
        f.write("static constexpr float CUBE_UVS[24][2] = {\n")
        for uv in uvs:
            f.write(f"    {{ {uv[0]:.4f}f, {uv[1]:.4f}f }},\n")
        f.write("};\n\n")

        if triangles:
            f.write("// 12 triangles, 3 indices per triangle\n")
            f.write("static constexpr uint32_t CUBE_INDICES[36] = {\n")
            for i in range(0, len(indices), 3):
                f.write(f"    {indices[i]}, {indices[i+1]}, {indices[i+2]},\n")
        else:
            f.write("// 6 quads, 4 indices per quad\n")
            f.write("static constexpr uint32_t CUBE_INDICES[24] = {\n")
            for i in range(0, len(indices), 4):
                f.write(f"    {indices[i]}, {indices[i+1]}, {indices[i+2]}, {indices[i+3]},\n")
        f.write("};\n")

    print(f"✅ C++ header saved as '{filename}'")


def main() -> None:
    """Main function to generate data, normal map, and C++ header."""
    parser = argparse.ArgumentParser(description="Generate cube geometry and a normal map.")
    parser.add_argument(
        "--size",
        "-s",
        default=2.0,
        type=float,
        help="The total width, height, and depth of the cube. Defaults to 2.0.",
    )
    parser.add_argument(
        "--texture-size",
        default=256,
        type=power_of_two,
        help="Width and height of the output texture (must be a power of two). Defaults to 256.",
    )
    parser.add_argument(
        "--wind-cw",
        action="store_true",
        help="Use clockwise winding for front faces (default is counter-clockwise).",
    )
    parser.add_argument(
        "--generate-triangles",
        "-T",
        action="store_true",
        help="Generate triangle indices (2 per face) instead of quad indices (1 per face).",
    )
    args = parser.parse_args()

    generate_normal_map(size=args.size, texture_size=args.texture_size)

    # Set ccw to True (default) unless the --wind-cw flag is specified.
    use_ccw = not args.wind_cw

    vertices, normals, uvs, indices = generate_cube_data(size=args.size, ccw=use_ccw, triangles=args.generate_triangles)

    write_cpp_header(
        "cube_geometry.hpp",
        vertices,
        normals,
        uvs,
        indices,
        size=args.size,
        ccw=use_ccw,
        triangles=args.generate_triangles,
    )


if __name__ == "__main__":
    main()
