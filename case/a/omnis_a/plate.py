"""プレート。キー面の座標で作る。印刷する向きも同じ（下面を造形面に置く）。

上から、見えている上面（w = bezel）、キーキャップが入るくぼみ、スイッチが載る面（w = 0）、
下面（w = -plate_t）。スイッチ穴のふちは land の厚みで、その下を 45° で広げてある。
こうすると、下に支えのない張り出しを作らずに、穴の間を深いリブにできる。
"""

from __future__ import annotations

import math

import numpy as np
from manifold3d import Manifold

from . import geom, layout as lay_mod, parts, shell
from .geom import EPS
from .layout import CRADLE, Key, Layout
from .params import Params


def blank(p: Params, lay: Layout) -> Manifold:
    """穴をあける前のプレート。外形のうち、合わせ目より上と差し込み部。"""
    region = shell.seat_solid(p, lay) + shell.plug_solid(p, lay)
    return shell.exterior_solid(p, lay) ^ region


def switch_hole(p: Params, key: Key) -> Manifold:
    """スイッチ穴（まっすぐ抜いた部分）。"""
    return geom.prism(key.square(p.hole), p.bottom - 1.0, -key.sink + EPS)


def switch_flare(p: Params, key: Key) -> Manifold:
    """スイッチ穴のふちの下を 45° で広げる部分。"""
    flare = min(p.flare, p.plate_t - key.sink - p.land)
    w_land = -key.sink - p.land
    return parts.flare_cut(key.square(p.hole), key.square(p.hole + 2 * flare),
                           p.bottom - 1.0, w_land - flare, w_land)


def wheel_axle_span(p: Params, lay: Layout) -> tuple[float, float]:
    u0, u1, _v0, _v1 = lay.wheel
    return u0 - CRADLE, u1 + CRADLE


def wheel_cut(p: Params, lay: Layout) -> Manifold:
    """ホイールの窓と、軸を上から落とし込む溝。"""
    u0, u1, v0, v1 = lay.wheel
    vc = (v0 + v1) / 2.0
    w_axis = lay_mod.wheel_axis_w(p)
    w_flat = lay_mod.wheel_flat_w(p)
    low, high = p.bottom - 1.0, p.top + 5.0
    pts = []
    for v in (v0, v1):
        pts += [(u0, v, low), (u0, v, high), (u1, v, high), (u1, v, w_flat),
                (u1 - (w_flat - low), v, low)]
    window = Manifold.hull_points(np.asarray(pts, dtype=float))
    slot_w = p.axle_d + 0.25
    a0, a1 = wheel_axle_span(p, lay)
    slot = geom.box(a0, a1, vc - slot_w / 2.0, vc + slot_w / 2.0, w_axis, high)
    trough = Manifold.cylinder(a1 - a0, slot_w / 2.0, slot_w / 2.0, 48)
    trough = geom.move(trough, geom.rot_y(90.0), np.array([a0, vc, w_axis]))
    return window + slot + trough


def lip(p: Params, lay: Layout) -> Manifold:
    """Space とホイールの間の低い仕切り。"""
    u0, u1, v0, v1 = lay.lip
    foot = geom.section([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
    crown = geom.grow(foot, -min(0.6, (u1 - u0) / 2.0 - 0.3))
    return geom.loft(foot, p.top - 0.5, crown, p.top + p.lip_h)


def ball_seat_points(p: Params, lay: Layout) -> list[np.ndarray]:
    """球が載る3点（キー面の座標）。1点は傾きの低い側（-u）に置く。"""
    r = p.ball_d / 2.0
    c = np.array([lay.ball[0], lay.ball[1], lay_mod.ball_centre_w(p)])
    phi = math.radians(p.ball_seat_deg)
    out = []
    for az in (180.0, 60.0, -60.0):
        a = math.radians(az)
        d = np.array([math.sin(phi) * math.cos(a), math.sin(phi) * math.sin(a), -math.cos(phi)])
        out.append(c + r * d)
    return out


def ball_cut(p: Params, lay: Layout) -> Manifold:
    """球の穴。中心より下は球に沿ったすり鉢、上はまっすぐ。上面のまわりを面取りする。"""
    r = p.ball_d / 2.0 + p.ball_gap
    cu, cv = lay.ball
    cw = lay_mod.ball_centre_w(p)
    bowl = Manifold.sphere(r, 128).translate((cu, cv, cw))
    shaft = Manifold.cylinder(p.top + 5.0 - cw, r, r, 128).translate((cu, cv, cw))
    ring = Manifold.cylinder(p.ball_ring + 5.0, r, r + p.ball_ring + 5.0, 128)
    ring = ring.translate((cu, cv, p.top - p.ball_ring))
    return bowl + shaft + ring


def ball_nubs(p: Params, lay: Layout) -> Manifold:
    """球を受ける3つの突起。下面（造形面）から立ち上がる。"""
    r = p.ball_d / 2.0
    nub_r = 2.2
    c = np.array([lay.ball[0], lay.ball[1], lay_mod.ball_centre_w(p)])
    nubs = None
    for pt in ball_seat_points(p, lay):
        d = (pt - c) / r
        s = Manifold.sphere(nub_r, 48).translate(tuple(c + (r + nub_r) * d))
        nubs = s if nubs is None else nubs + s
    keep = geom.box(-geom.BIG, geom.BIG, -geom.BIG, geom.BIG, p.bottom, p.top)
    return nubs ^ keep


PLUG_SKIN = 1.0   # 差し込み部の側面に必ず残す肉


def plate(p: Params, lay: Layout) -> Manifold:
    """プレート（キー面の座標）。"""
    through = geom.prism(lay_mod.wells(p, lay), 0.0, p.top + 5.0)
    vw = lay_mod.voice_well(p, lay)
    if vw is not None:
        through += geom.prism(vw, -p.voice_sink, p.top + 5.0)
    flares = None
    for key in lay.keys:
        through += switch_hole(p, key)
        fl = switch_flare(p, key)
        flares = fl if flares is None else flares + fl
    if lay.wheel is not None:
        through += wheel_cut(p, lay)
    else:
        through += ball_cut(p, lay)

    m = blank(p, lay)
    if lay.lip is not None:
        m += lip(p, lay)
    m -= flares
    # 穴のふちを広げても、差し込み部の側面には薄い壁を残す。
    m += shell.plug_solid(p, lay) - shell.plug_solid(p, lay, shrink=PLUG_SKIN)
    m -= through
    if lay.wheel is None:
        m += ball_nubs(p, lay)
    return m


def printable(p: Params, lay: Layout) -> Manifold:
    """印刷する向き。下面を z = 0 に置き、右手は左右を反転する。"""
    m = plate(p, lay).translate((0.0, 0.0, p.plate_t))
    if lay.side == "right":
        m = m.mirror((1.0, 0.0, 0.0))
    return m
