# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal binary PLY reader/writer — no external dependencies beyond numpy."""

from __future__ import annotations

import struct
from collections import OrderedDict

import numpy as np

# PLY type name → (struct format char, numpy dtype)
_PLY_TYPES = {
    "float": ("f", np.float32),
    "float32": ("f", np.float32),
    "double": ("d", np.float64),
    "float64": ("d", np.float64),
    "uchar": ("B", np.uint8),
    "uint8": ("B", np.uint8),
    "char": ("b", np.int8),
    "int8": ("b", np.int8),
    "ushort": ("H", np.uint16),
    "uint16": ("H", np.uint16),
    "short": ("h", np.int16),
    "int16": ("h", np.int16),
    "uint": ("I", np.uint32),
    "uint32": ("I", np.uint32),
    "int": ("i", np.int32),
    "int32": ("i", np.int32),
}


def read_ply(path: str) -> dict[str, np.ndarray]:
    """Read a binary little-endian PLY file and return vertex properties.

    Returns:
        OrderedDict mapping property name → 1-D numpy array of length N.
    """
    with open(path, "rb") as f:
        # --- parse header ---
        line = f.readline().strip()
        if line != b"ply":
            raise ValueError(f"Not a PLY file: {path}")

        fmt = None
        vertex_count = 0
        properties: list[tuple[str, str]] = []  # (name, ply_type)
        in_vertex_element = False

        while True:
            line = f.readline().strip()
            if line == b"end_header":
                break
            parts = line.decode("ascii").split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex_element = parts[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex_element:
                if parts[1] == "list":
                    raise ValueError("List properties are not supported")
                properties.append((parts[2], parts[1]))

        if fmt != "binary_little_endian":
            raise ValueError(f"Only binary_little_endian is supported, got: {fmt}")
        if vertex_count == 0:
            raise ValueError("No vertices found in PLY header")

        # --- build struct format for one vertex ---
        struct_fmt = "<"
        for _, ply_type in properties:
            if ply_type not in _PLY_TYPES:
                raise ValueError(f"Unsupported PLY type: {ply_type}")
            struct_fmt += _PLY_TYPES[ply_type][0]

        vertex_size = struct.calcsize(struct_fmt)
        raw = f.read(vertex_count * vertex_size)
        if len(raw) != vertex_count * vertex_size:
            raise ValueError(f"Expected {vertex_count * vertex_size} bytes, got {len(raw)}")

    # --- unpack into per-property arrays ---
    result: dict[str, np.ndarray] = OrderedDict()
    for i, (name, ply_type) in enumerate(properties):
        dtype = _PLY_TYPES[ply_type][1]
        result[name] = np.empty(vertex_count, dtype=dtype)

    for row_idx in range(vertex_count):
        offset = row_idx * vertex_size
        values = struct.unpack_from(struct_fmt, raw, offset)
        for i, (name, _) in enumerate(properties):
            result[name][row_idx] = values[i]

    return result


def write_ply(
    path: str,
    count: int,
    map_to_tensors: dict[str, np.ndarray],
) -> None:
    """Write vertex data as a binary little-endian PLY file.

    Args:
        path: Output file path.
        count: Number of vertices.
        map_to_tensors: OrderedDict of property name → 1-D numpy array.
    """
    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {count}\n".encode())
        for key, tensor in map_to_tensors.items():
            data_type = "float" if tensor.dtype.kind == "f" else "uchar"
            f.write(f"property {data_type} {key}\n".encode())
        f.write(b"end_header\n")

        for i in range(count):
            for tensor in map_to_tensors.values():
                value = tensor[i]
                if tensor.dtype.kind == "f":
                    f.write(np.float32(value).tobytes())
                elif tensor.dtype == np.uint8:
                    f.write(value.tobytes())
