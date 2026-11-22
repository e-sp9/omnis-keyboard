"""プレートと傾斜台のほかの部品。キーキャップ、ホイール、球、確認用のスイッチ。"""

from __future__ import annotations

import math

import numpy as np
from manifold3d import CrossSection, Manifold

from . import geom, layout as lay_mod
from .geom import EPS
from .layout import Key, Layout
from .params import Params


# ---- キーキャップ ---------------------------------------------------------------

def _rounded(w: float, d: float, r: float) -> CrossSection:
    r = min(r, w / 2.0 - 0.05, d / 2.0 - 0.05)
    return geom.grow(geom.rect(0.0, 0.0, w - 2 * r, d - 2 * r), r, round_=True)


def _stack(levels) -> Manifold:
    """(輪郭, z) をいくつか重ねた凸な形。"""
    pts = []
    for cs, z in levels:
        for poly in cs.to_polygons():
            pts += [(x, y, z) for x, y in poly]
    return Manifold.hull_points(np.asarray(pts, dtype=float))


def cap_heights(p: Params, key: Key) -> tuple[float, float]:
    """キャップの下端と上面の高さ（そのキーのスイッチが載る面から）。"""
    if key.kind == "voice":
        skirt = p.sw_top + p.sw_travel + 0.25   # 小さいキャップはハウジングをまたげない
        return skirt, skirt + p.voice_cap_h
    return p.cap_skirt, p.cap_top


def keycap(p: Params, key: Key) -> Manifold:
    """キーキャップ。原点はスイッチが載る面の、キーの中心。上が +z。"""
    w, d = key.cap_w, key.cap_d
    z0, z1 = cap_heights(p, key)
    taper = p.cap_taper if key.kind != "voice" else 0.9
    edge = 0.5
    r = p.cap_round
    band = min(1.0, (z1 - z0) / 3.0)     # 下端のまっすぐな部分
    outer = _stack([
        (_rounded(w, d, r), z0),
        (_rounded(w, d, r), z0 + band),
        (_rounded(w - 2 * taper, d - 2 * taper, r), z1 - edge),
        (_rounded(w - 2 * (taper + edge), d - 2 * (taper + edge), max(0.3, r - edge)), z1),
    ])
    t = p.cap_wall
    r_in = max(0.4, r - t)               # 中の角は小さく丸める。ハウジングの角に当てないため
    inner = _stack([
        (_rounded(w - 2 * t, d - 2 * t, r_in), z0 - 1.0),
        (_rounded(w - 2 * t, d - 2 * t, r_in), z0 + band),
        (_rounded(w - 2 * (taper + t), d - 2 * (taper + t), r_in), z1 - p.cap_roof),
    ])
    z_post = p.sw_stem - p.stem_depth
    post = Manifold.cylinder(z1 - p.cap_roof + 0.2 - z_post, p.stem_post / 2.0, p.stem_post / 2.0, 48)
    post = post.translate((0.0, 0.0, z_post))
    a, b = p.stem_slot
    slot = geom.box(-a / 2, a / 2, -b / 2, b / 2, z_post - 1.0, p.sw_stem)
    slot += geom.box(-b / 2, b / 2, -a / 2, a / 2, z_post - 1.0, p.sw_stem)
    return (outer - inner) + (post - slot)


def keycap_printable(p: Params, key: Key) -> Manifold:
    """印刷する向き（上面を下）。"""
    _z0, z1 = cap_heights(p, key)
    return geom.move(keycap(p, key), geom.rot_x(180.0), np.array([0.0, 0.0, z1]))


def place_on_key(p: Params, key: Key, m: Manifold, press: float = 0.0) -> Manifold:
    """キーの位置へ置く（キー面の座標）。press は押し込み量。"""
    return geom.move(m, geom.rot_z(key.rot), np.array([key.u, key.v, -key.sink - press]))


def switch(p: Params) -> tuple[Manifold, Manifold]:
    """Choc V2 のおおよその形。プレートより上と下に分けて返す。原点はスイッチが載る面。

    上は、つば（15 mm 角）、その上のまっすぐな部分（下ハウジングと同じ幅）、上端へすぼまる部分。
    軸は入れていない（キーキャップと一緒に動くため）。
    """
    f, b = p.sw_flange, p.sw_body
    upper = geom.prism(geom.rect(0, 0, f, f), 0.0, p.sw_flange_t)
    upper += geom.prism(geom.rect(0, 0, b, b), p.sw_flange_t - EPS, p.sw_shoulder)
    upper += geom.loft(geom.rect(0, 0, b, b), p.sw_shoulder - EPS,
                       geom.rect(0, 0, p.sw_crown, p.sw_crown), p.sw_top)
    lower = geom.prism(geom.rect(0, 0, b, b), -p.sw_below, EPS)
    lower += Manifold.cylinder(p.sw_post + 0.2, 2.4, 2.4, 32).translate((0, 0, -p.sw_below - p.sw_post))
    for px, py in ((0.0, 5.9), (5.0, 3.8)):
        lower += Manifold.cylinder(p.sw_pin + 0.2, 0.6, 0.6, 12).translate((px, py, -p.sw_below - p.sw_pin))
    return upper, lower


def stem(p: Params, press: float = 0.0) -> Manifold:
    """軸（まわりの筒を含む外径 6.5 mm）。キーキャップと一緒に動く。"""
    return Manifold.cylinder(p.sw_stem + 2.0, 3.25, 3.25, 48).translate((0, 0, -2.0 - press))


# ---- ホイール -------------------------------------------------------------------

def wheel(p: Params, knurl: bool = True) -> Manifold:
    """スクロールホイール。軸は z（印刷する向き）。両端を面取りし、表面にあや目を付ける。"""
    r, half, e = p.wheel_d / 2.0, p.wheel_len / 2.0, p.wheel_end
    bore = (p.axle_d + 0.35) / 2.0
    starts, depth = 18, 0.45
    lead = 2.0 * math.pi * r / starts          # 45° のあや目
    flat = half - e
    # 外周に沿って細かく点を打っておき、回してできる格子の点を半径の向きに動かす
    steps = max(2, round(2 * flat / (lead / 6.0))) if knurl else 1
    side = [(r, -flat + 2 * flat * i / steps) for i in range(steps + 1)]
    profile = geom.section([(bore, -half), (r - e, -half)] + side + [(r - e, half), (bore, half)])
    body = profile.revolve(starts * 6)
    if not knurl:
        return body

    def warp(v):
        x, y, z = v
        rad = math.hypot(x, y)
        if rad < r - 0.02 or abs(z) > flat - 1e-6:
            return (x, y, z)
        ang = math.atan2(y, x) * starts / (2.0 * math.pi)
        zz = z / lead
        a = abs(((ang + zz) % 1.0) - 0.5) * 2.0     # 0（溝の底）〜1
        b = abs(((ang - zz) % 1.0) - 0.5) * 2.0
        fade = min(1.0, (flat - abs(z)) / 0.6)
        k = (rad - depth * (1.0 - min(a, b)) * fade) / rad
        return (x * k, y * k, z)

    return body.warp(warp)


def wheel_in_place(p: Params, lay: Layout, knurl: bool = True) -> Manifold:
    """窓に入れた位置のホイール（キー面の座標）。"""
    u0, u1, v0, v1 = lay.wheel
    centre = np.array([(u0 + u1) / 2.0, (v0 + v1) / 2.0, lay_mod.wheel_axis_w(p)])
    return geom.move(wheel(p, knurl), geom.rot_y(90.0), centre)


def axle_in_place(p: Params, lay: Layout) -> Manifold:
    u0, u1, v0, v1 = lay.wheel
    a0, a1 = u0 - lay_mod.CRADLE + 0.3, u1 + lay_mod.CRADLE - 0.3
    rod = Manifold.cylinder(a1 - a0, p.axle_d / 2.0, p.axle_d / 2.0, 24)
    return geom.move(rod, geom.rot_y(90.0), np.array([a0, (v0 + v1) / 2.0, lay_mod.wheel_axis_w(p)]))


# ---- 球 -------------------------------------------------------------------------

def ball_in_place(p: Params, lay: Layout, segments: int = 128) -> Manifold:
    c = (lay.ball[0], lay.ball[1], lay_mod.ball_centre_w(p))
    return Manifold.sphere(p.ball_d / 2.0, segments).translate(c)


# ---- スイッチ穴のふち、試し刷り ---------------------------------------------------

def flare_cut(hole: CrossSection, wide: CrossSection, z_low: float, z_wide: float, z_land: float) -> Manifold:
    """穴のふちの下を広げる形。z_land で穴の大きさ、z_wide から下は wide の大きさ。"""
    return _stack([(wide, z_low), (wide, z_wide), (hole, z_land)])



def fit_test(p: Params) -> Manifold:
    """スイッチ穴の試し刷り。穴は 3 通りで、横の刻みの数（1〜3）で見分ける。"""
    sizes = (p.hole - 0.1, p.hole, p.hole + 0.1)
    pitch = 19.0
    length = pitch * len(sizes) + 2.0
    m = geom.box(0.0, length, 0.0, 21.0, 0.0, p.plate_t)
    for i, size in enumerate(sizes):
        cx, cy = 1.0 + pitch * (i + 0.5), 10.5
        key = Key(f"t{i}", "alpha", cx, cy)
        z_land = p.plate_t - p.land
        cut = geom.prism(key.square(size), -1.0, p.plate_t + 1.0)
        cut += flare_cut(key.square(size), key.square(size + 2 * p.flare), -1.0, z_land - p.flare, z_land)
        m -= cut
        for n in range(i + 1):
            m -= geom.box(cx - 3.0 + 2.2 * n, cx - 2.0 + 2.2 * n, -1.0, 0.9, -1.0, p.plate_t + 1.0)
    return m
