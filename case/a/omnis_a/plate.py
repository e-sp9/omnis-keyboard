"""プレート。キー面の座標で作る。印刷する向きも同じ（下面を造形面に置く）。

上から、見えている上面（w = bezel）、キーキャップが入るくぼみ、スイッチが載る面（w = 0）、
下面（w = -plate_t）。スイッチ穴のふちは厚み clip_t の平らな板で、Choc V2 の爪はこの板の
下面に掛かる。板の下は穴の間を深いリブにし、爪が開く所だけを外へ広げてある（爪の逃げ）。

上面は、参考画像に合わせて縁をやわらげてある。外周は面取りの帯、キーのくぼみの縁は細い
面取り、球の穴とホイールの窓はふちをなだらかに斜めに落としてある。
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


# ---- キーのくぼみとスイッチ穴 -----------------------------------------------------

def well_cut(p: Params, lay: Layout) -> Manifold:
    """キーキャップが入るくぼみ。上の縁を well_chamfer だけ面取りする。"""
    cut = geom.prism(lay_mod.wells(p, lay), 0.0, p.top + 5.0)
    c = p.well_chamfer
    rects = list(lay_mod.well_rects(p, lay))
    cut_voice = None
    for key in lay.keys:
        if key.kind == "voice":
            size = p.sw_flange + 2 * 0.3
            cut_voice = geom.prism(key.square(size), -key.sink, p.top + 5.0)
            rects.append((key.u, key.v, size, size, key.rot))
    if cut_voice is not None:
        cut += cut_voice
    if c > 0:
        for u, v, w, d, rot in rects:
            cut += geom.loft(geom.rect(u, v, w, d, rot), p.top - c,
                             geom.rect(u, v, w + 2 * c, d + 2 * c, rot), p.top + EPS)
    return cut


def switch_hole(p: Params, key: Key) -> Manifold:
    """スイッチ穴（まっすぐ抜いた部分）。"""
    return geom.prism(key.square(p.hole), p.bottom - 1.0, -key.sink + EPS)


def open_sides(p: Params, key: Key, plug: Manifold) -> tuple[int, ...]:
    """爪の逃げのうち、差し込み部の側面に近すぎて薄い壁しか残らない辺。

    差し込み部の内側の面は寝ているので、内側の列では穴のすぐ脇まで来る。そこに薄い壁を
    残すと印刷が乱れるので、その辺の逃げは側面まで抜いてしまう。抜けた先は傾斜台の壁の
    内側で、外からは見えない。
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


# ---- 球とホイールのまわり ---------------------------------------------------------

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
    """球の穴。中心より下は球に沿ったすり鉢、上はまっすぐ。上面のふちは、輪の形になだらかに落とす。"""
    r = p.ball_d / 2.0 + p.ball_gap
    cu, cv = lay.ball
    cw = lay_mod.ball_centre_w(p)
    bowl = Manifold.sphere(r, 128).translate((cu, cv, cw))
    shaft = Manifold.cylinder(p.top + 5.0 - cw, r, r, 128).translate((cu, cv, cw))
    ring = [(r - 0.5, p.top - p.ball_ring_depth), (r, p.top - p.ball_ring_depth), (r + p.ball_ring, p.top),
            (r + p.ball_ring, p.top + 5.0), (r - 0.5, p.top + 5.0)]
    return bowl + shaft + geom.section(ring).revolve(128).translate((cu, cv, 0.0))


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


def wheel_axle_span(p: Params, lay: Layout) -> tuple[float, float]:
    u0, u1, _v0, _v1 = lay.wheel
    return u0 - CRADLE, u1 + CRADLE


def wheel_window(p: Params, lay: Layout, shrink: float = 0.0) -> Manifold:
    """ホイールの窓（角を丸める前の形）。shrink を渡すと、軸の向きの両端をそのぶん内へ寄せる。

    内側の端は、ホイールの面取りより下を 45° で引っ込める（差し込み部の寝ている側面との間に
    肉を残す）。
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


def wheel_pocket(p: Params, lay: Layout) -> Manifold:
    """ホイールの窓。上から見た角を丸め、上面のふちを斜めに落とす。"""
    u0, u1, v0, v1 = lay.wheel
    r = min(p.pocket_r, (v1 - v0) / 2.0 - 0.1)
    core = geom.rect((u0 + u1) / 2.0, (v0 + v1) / 2.0, u1 - u0 - 2 * r, v1 - v0 - 2 * r)
    cut = wheel_window(p, lay) ^ geom.prism(geom.grow(core, r, round_=True), p.bottom - 2.0, p.top + 6.0)
    if p.pocket_ring > 0:
        cut += geom.loft(geom.grow(core, r, round_=True), p.top - p.pocket_ring_depth,
                         geom.grow(core, r + p.pocket_ring, round_=True), p.top + EPS)
    return cut


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
    """Space とホイールの間の低い仕切り（lip_h を 0 より大きくしたときだけ付く）。"""
    u0, u1, v0, v1 = lay.lip
    foot = geom.section([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
    crown = geom.grow(foot, -min(0.6, (u1 - u0) / 2.0 - 0.3))
    return geom.loft(foot, p.top - 0.5, crown, p.top + p.lip_h)


# ---- 全体 ----------------------------------------------------------------------

def plate(p: Params, lay: Layout) -> Manifold:
    """プレート（キー面の座標）。"""
    through = well_cut(p, lay)
    plug = shell.plug_solid(p, lay)
    for key in lay.keys:
        through += switch_hole(p, key)
        relief = clip_reliefs(p, key, plug)
        if relief is not None:
            through += relief
    if lay.wheel is not None:
        through += wheel_pocket(p, lay) + wheel_axle_cut(p, lay)
    else:
        through += ball_cut(p, lay)

    m = blank(p, lay)
    m -= through
    if lay.wheel is not None:
        m += wheel_pads(p, lay)
        if lay.lip is not None:
            m += lip(p, lay)
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
