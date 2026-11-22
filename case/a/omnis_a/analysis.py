"""形から計算で確かめること。重心と倒れにくさ、印刷の向きでの張り出し。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import assembly, geom, parts, shell
from .params import Params

PLA = 1.24e-3        # g/mm³
SWITCH_G = 1.8       # Choc V2 1個のおおよその重さ
BALL_DENSITY = 1.35e-3
# 印刷物は中が粗く埋まるので、体積から出した重さより軽い。BambuStudio の見積もり
# （0.20mm Standard、壁 2 本、充填 15%）と体積の比から決めた。
FILL = {"plate": 0.67, "base": 0.87, "foot": 0.87}


def centroid(manifold) -> tuple[np.ndarray, float]:
    """重心と体積。"""
    mesh = manifold.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    f = np.asarray(mesh.tri_verts)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    return ((a + b + c) / 4.0 * vol[:, None]).sum(0) / vol.sum(), float(vol.sum())


@dataclass(frozen=True)
class Stability:
    mass: float                 # g
    cg: np.ndarray              # 机の座標
    foot_x: tuple[float, float]
    tip_force: dict             # キーの名前 -> 倒れ始める力（N）。inf は倒れない
    landing: dict               # キーの名前 -> 押す力の向きが机に届く点が、足の外に出る量（mm）


def stability(p: Params, side: str) -> Stability:
    """キーを、キー面に垂直に押したときの倒れにくさ。

    押す力の向きを机まで延ばした点が足の内側なら、どれだけ強く押しても倒れない。
    外側なら、足の縁を支点にして、力のモーメントが自重のモーメントを超えると倒れ始める。
    """
    lay = assembly.layout(p, side)
    half = assembly.half(p, side, knurl=False)
    items = []
    for name, density in (("plate", PLA * FILL["plate"]), ("base", PLA * FILL["base"]),
                          ("foot", PLA * FILL["foot"]), ("caps", PLA),
                          ("voice_cap", PLA), ("wheel", PLA), ("ball", BALL_DENSITY)):
        if name in half:
            c, vol = centroid(half[name])
            items.append((vol * density, c))
    c, _ = centroid(half["switch_upper"])
    items.append((SWITCH_G * len(lay.keys), c))
    mass = sum(m for m, _ in items)
    cg = sum(m * c for m, c in items) / mass

    foot = geom.polygons(shell.foot_outline(p, lay))[0]
    rot, trans = shell.frame(p, lay)
    push = rot @ np.array([0.0, 0.0, -1.0])
    tip, landing = {}, {}
    for key in lay.keys:
        _z0, z1 = parts.cap_heights(p, key)
        point = rot @ np.array([key.u, key.v, z1 - key.sink]) + trans
        ground = point + push * (-point[2] / push[2])
        best, beyond = math.inf, 0.0
        for i in range(len(foot)):
            a, b = foot[i], foot[(i + 1) % len(foot)]
            edge = b - a
            out = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
            over = float((ground[:2] - a) @ out)
            if over > 1e-9:
                arm = float(-((cg[:2] - a) @ out))
                force = mass * 9.81e-3 * arm / (abs(push[2]) * over)
                if force < best:
                    best, beyond = force, over
        tip[key.name] = best
        landing[key.name] = beyond
    return Stability(mass, cg, (float(foot[:, 0].min()), float(foot[:, 0].max())), tip, landing)


@dataclass(frozen=True)
class Overhang:
    steepest: float        # 下を向いた斜面の、垂直からの角度の最大（度）。天井は含めない
    over_limit: float      # limit を超える斜面の面積（mm²）。天井は含めない
    ceiling: float         # 天井（85° 以上）の面積（mm²）
    width: float           # 天井ひとつの幅（面積 ÷ 縁の長さの半分）の最大（mm）


def overhang(mesh, limit: float = 47.5, min_area: float = 0.02) -> Overhang:
    """印刷する向きのメッシュ（trimesh）で、下に支えがない面を調べる。"""
    normals = mesh.face_normals
    areas = mesh.area_faces
    tri = mesh.triangles
    z_min = float(mesh.bounds[0][2])
    on_bed = tri[:, :, 2].max(axis=1) < z_min + 0.01
    down = (-normals[:, 2]).clip(0.0, 1.0)
    angle = np.degrees(np.arcsin(down))
    faces = (~on_bed) & (areas > min_area) & (angle > 0.5)
    slope_all = faces & (angle < 85.0)
    steepest = float(angle[slope_all].max()) if slope_all.any() else 0.0
    slope = slope_all & (angle > limit)
    roof = (~on_bed) & (angle >= 85.0)

    width = 0.0
    if roof.any():
        idx = np.flatnonzero(roof)
        parent = {int(i): int(i) for i in idx}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b in mesh.face_adjacency:
            a, b = int(a), int(b)
            if a in parent and b in parent:
                parent[find(a)] = find(b)
        groups: dict[int, list[int]] = {}
        for i in parent:
            groups.setdefault(find(i), []).append(i)
        for members in groups.values():
            total = float(areas[members].sum())
            if total < 0.5:
                continue
            edges = np.sort(mesh.faces[members][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
            uniq, counts = np.unique(edges, axis=0, return_counts=True)
            rim = mesh.vertices[uniq[counts == 1]]
            length = float(np.linalg.norm(rim[:, 0] - rim[:, 1], axis=1).sum())
            width = max(width, total / (length / 2.0))
    return Overhang(steepest, float(areas[slope].sum()), float(areas[roof].sum()), width)
