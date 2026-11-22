"""プレート。キー面の座標で作る。印刷する向きも同じ（下面を造形面に置く）。

上から、見えている上面（w = bezel）、キーキャップが入るくぼみ、スイッチが載る面（w = 0）、
下面（w = -plate_t）。スイッチ穴のふちは厚み clip_t の平らな板で、Choc V2 の爪はこの板の
下面に掛かる。板の下は穴の間を深いリブにし、爪が開く所だけを外へ広げてある（爪の逃げ）。
"""

from __future__ import annotations

import math

import numpy as np
from manifold3d import Manifold

from . import geom, layout as lay_mod, parts, shell
from .geom import EPS
from .layout import CRADLE, Key, Layout
from .params import Params

MIN_WALL = 0.8   # 爪の逃げと差し込み部の側面の間に、これだけの肉が残らないなら外へ抜く


def blank(p: Params, lay: Layout) -> Manifold:
    """穴をあける前のプレート。外形のうち、合わせ目より上と差し込み部。"""
    region = shell.seat_solid(p, lay) + shell.plug_solid(p, lay)
    return shell.exterior_solid(p, lay) ^ region


def switch_hole(p: Params, key: Key) -> Manifold:
    """スイッチ穴（まっすぐ抜いた部分）。"""
    return geom.prism(key.square(p.hole), p.bottom - 1.0, -key.sink + EPS)


def open_sides(p: Params, key: Key, plug: Manifold) -> tuple[int, ...]:
    """爪の逃げのうち、差し込み部の側面に近すぎて薄い壁しか残らない辺。

    内側の壁は寝ているので、内側の列では差し込み部の側面が穴のすぐ脇まで来る。そこに
    薄い壁を残すと印刷が乱れるので、その辺の逃げは側面まで抜いてしまう。抜けた先は
    傾斜台の壁の内側で、外からは見えない。
    """
    z_ledge = -key.sink - parts.ledge(p, key)
    if z_ledge - p.bottom < 0.05:
        return ()
    h = p.hole / 2.0
    out = []
    for i in range(4):
        probe = geom.box(h, h + p.clip_d + MIN_WALL, -p.clip_w / 2.0, p.clip_w / 2.0, p.bottom, z_ledge)
        probe = geom.move(probe, geom.rot_z(90.0 * i + key.rot), np.array([key.u, key.v, 0.0]))
        if (probe - plug).volume() > 1e-6:
            out.append(i)
    return tuple(out)


def clip_reliefs(p: Params, key: Key, plug: Manifold) -> Manifold | None:
    """そのキーの爪の逃げ（板が薄くて要らないときは None）。"""
    z_ledge = -key.sink - parts.ledge(p, key)
    return parts.clip_relief(p, key, p.bottom - 1.0, z_ledge, open_sides(p, key, plug))


def wheel_axle_span(p: Params, lay: Layout) -> tuple[float, float]:
    u0, u1, _v0, _v1 = lay.wheel
    return u0 - CRADLE, u1 + CRADLE


def wheel_window(p: Params, lay: Layout, shrink: float = 0.0) -> Manifold:
    """ホイールの窓。shrink を渡すと、軸の向きの両端をそのぶん内へ寄せた形になる。

    内側の端は、ホイールの面取りより下を 45° で引っ込める（寝ている内側の壁との間に肉を残す）。
    """
    u0, u1, v0, v1 = lay.wheel
    u0, u1 = u0 + shrink, u1 - shrink
    w_flat = lay_mod.wheel_flat_w(p)
    low, high = p.bottom - 1.0, p.top + 5.0
    pts = []
    for v in (v0, v1):
        pts += [(u0, v, low), (u0, v, high), (u1, v, high), (u1, v, w_flat),
                (u1 - (w_flat - low), v, low)]
    return Manifold.hull_points(np.asarray(pts, dtype=float))


def wheel_axle_cut(p: Params, lay: Layout) -> Manifold:
    """軸を上から落とし込む溝。窓の両端から CRADLE だけ先まで伸びる。"""
    _u0, _u1, v0, v1 = lay.wheel
    vc = (v0 + v1) / 2.0
    w_axis = lay_mod.wheel_axis_w(p)
    slot_w = p.axle_d + 0.25
    a0, a1 = wheel_axle_span(p, lay)
    slot = geom.box(a0, a1, vc - slot_w / 2.0, vc + slot_w / 2.0, w_axis, p.top + 5.0)
    trough = Manifold.cylinder(a1 - a0, slot_w / 2.0, slot_w / 2.0, 48)
    trough = geom.move(trough, geom.rot_y(90.0), np.array([a0, vc, w_axis]))
    return slot + trough


def wheel_pads(p: Params, lay: Layout) -> Manifold:
    """窓の両端の壁で、軸の下だけを wheel_pad だけ窓の中へ出した当たり。

    キー面と一緒に軸も傾くので、ホイールは低い側へ寄る。寄ったホイールの端面は、軸に近い
    この当たりに触れて止まり、窓の壁には届かない。下面から立ち上がる形なので、支えなしで
    印刷できる。壁の中へも少し食い込ませて、壁とひと続きにする。
    """
    _u0, _u1, v0, v1 = lay.wheel
    vc = (v0 + v1) / 2.0
    u_lo, u_hi = wheel_axle_span(p, lay)
    top = lay_mod.wheel_axis_w(p) - 0.3      # 軸受けの溝の底と面をそろえないよう、軸の少し下で止める
    strip = geom.box(u_lo, u_hi, vc - p.wheel_pad_r, vc + p.wheel_pad_r, p.bottom, top)
    return (strip ^ wheel_window(p, lay, shrink=-0.3)) - wheel_window(p, lay, shrink=p.wheel_pad)


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


def plate(p: Params, lay: Layout) -> Manifold:
    """プレート（キー面の座標）。"""
    through = geom.prism(lay_mod.wells(p, lay), 0.0, p.top + 5.0)
    vw = lay_mod.voice_well(p, lay)
    if vw is not None:
        through += geom.prism(vw, -p.voice_sink, p.top + 5.0)
    plug = shell.plug_solid(p, lay)
    for key in lay.keys:
        through += switch_hole(p, key)
        relief = clip_reliefs(p, key, plug)
        if relief is not None:
            through += relief
    if lay.wheel is not None:
        through += wheel_window(p, lay) + wheel_axle_cut(p, lay)
    else:
        through += ball_cut(p, lay)

    m = blank(p, lay)
    if lay.lip is not None:
        m += lip(p, lay)
    m -= through
    if lay.wheel is not None:
        m += wheel_pads(p, lay)
        m -= wheel_axle_cut(p, lay)
    else:
        m += ball_nubs(p, lay)
    return m


def printable(p: Params, lay: Layout) -> Manifold:
    """印刷する向き。下面を z = 0 に置き、右手は左右を反転する。"""
    m = plate(p, lay).translate((0.0, 0.0, p.plate_t))
    if lay.side == "right":
        m = m.mirror((1.0, 0.0, 0.0))
    return m
