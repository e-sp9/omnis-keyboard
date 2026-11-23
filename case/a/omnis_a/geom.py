"""形を作るための小さな道具。manifold3d の上に載せている。"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from manifold3d import CrossSection, JoinType, Manifold

BIG = 600.0   # 「十分に大きい」箱の半分の大きさ
EPS = 0.02    # 面をぴったり重ねないための食い込み
TINY = 0.01   # これより短い辺は輪郭から落とす


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class Plane:
    """半空間 n·p <= d。n は外向きの単位ベクトル。"""

    __slots__ = ("n", "d")

    def __init__(self, n, d: float):
        self.n = unit(n)
        self.d = float(d)

    @classmethod
    def through(cls, n, point) -> "Plane":
        n = unit(n)
        return cls(n, float(n @ np.asarray(point, dtype=float)))

    def shifted(self, delta: float) -> "Plane":
        """外向きに delta だけ動かす（負なら内へ）。"""
        return Plane(self.n, self.d + delta)

    def moved(self, rot: np.ndarray, trans: np.ndarray) -> "Plane":
        """p' = rot·p + trans で写した半空間。"""
        n2 = rot @ self.n
        return Plane(n2, self.d + float(n2 @ trans))

    def value(self, point) -> float:
        """点から面までの符号つき距離（外が正）。"""
        return float(self.n @ np.asarray(point, dtype=float)) - self.d


def chamfer(a: Plane, b: Plane, c: float) -> Plane:
    """a と b の稜を、両方の面で稜から c のところを通る面で落とす。"""
    s = a.n + b.n
    length = float(np.linalg.norm(s))
    cos_half = length / 2.0
    sin_half = math.sqrt(max(0.0, 1.0 - cos_half * cos_half))
    return Plane(s / length, (a.d + b.d) / length - c * sin_half)


def convex(planes: Iterable[Plane], bound: float = BIG) -> Manifold:
    """半空間の共通部分（凸多面体）。"""
    m = Manifold.cube((2 * bound, 2 * bound, 2 * bound), center=True)
    for pl in planes:
        m = m.trim_by_plane(tuple(-pl.n), -pl.d)
    return m


def affine(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    m = np.zeros((3, 4))
    m[:, :3] = rot
    m[:, 3] = trans
    return m


def move(m: Manifold, rot: np.ndarray, trans: np.ndarray) -> Manifold:
    return m.transform(affine(rot, trans))


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# ---- 2次元の輪郭 ---------------------------------------------------------------

def section(points: Sequence[Sequence[float]]) -> CrossSection:
    return CrossSection([np.asarray(points, dtype=float)])


def rect(cx: float, cy: float, w: float, h: float, rot_deg: float = 0.0) -> CrossSection:
    """中心 (cx, cy)、幅 w、奥行き h の長方形。rot_deg は中心まわりの回転。"""
    hw, hh = w / 2.0, h / 2.0
    pts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    a = math.radians(rot_deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return section(pts @ r.T + np.array([cx, cy]))


def rect_corners(cx: float, cy: float, w: float, h: float, rot_deg: float = 0.0) -> np.ndarray:
    return np.asarray(rect(cx, cy, w, h, rot_deg).to_polygons()[0])


def grow(cs: CrossSection, delta: float, round_: bool = False) -> CrossSection:
    """輪郭を delta だけ太らせる（負なら痩せさせる）。角はとがらせたまま。"""
    if abs(delta) < 1e-9:
        return cs
    if round_:
        return cs.offset(delta, JoinType.Round, 2.0, 64)
    return cs.offset(delta, JoinType.Miter, 4.0).simplify(TINY)


def close_gaps(cs: CrossSection, gap: float) -> CrossSection:
    """幅が 2·gap より細いすき間を埋める。"""
    return grow(grow(cs, gap), -gap)


def union_all(items: Iterable):
    out = None
    for it in items:
        out = it if out is None else out + it
    return out


def prism(cs: CrossSection, z0: float, z1: float) -> Manifold:
    """輪郭を z0 から z1 まで押し出す。"""
    return cs.extrude(z1 - z0).translate((0.0, 0.0, z0))


def loft(cs_low: CrossSection, z_low: float, cs_high: CrossSection, z_high: float) -> Manifold:
    """2枚の凸な輪郭を結ぶ錐台。"""
    pts = []
    for cs, z in ((cs_low, z_low), (cs_high, z_high)):
        for poly in cs.to_polygons():
            for x, y in poly:
                pts.append((x, y, z))
    return Manifold.hull_points(np.asarray(pts, dtype=float))


def box(x0, x1, y0, y1, z0, z1) -> Manifold:
    return Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def polygons(cs: CrossSection) -> list[np.ndarray]:
    return [np.asarray(p, dtype=float) for p in cs.to_polygons()]
