"""外形と傾斜台。

外形は「平らな面の半空間の共通部分」として作る凸多面体。面取りも半空間を1枚足すだけなので、
どの稜も平らな面で落ちる。側面はキー面の座標で決めてあり、tilt を変えるとキー面ごと回る。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from manifold3d import CrossSection, Manifold

from . import geom
from .geom import EPS, Plane
from .layout import Layout, make_layout
from .params import Params

SIDE_ORDER = ("front", "cheek", "back", "diag", "outer")   # 上から見て反時計回り


# ---- 座標 ----------------------------------------------------------------------

@lru_cache(maxsize=32)
def outer_height(p: Params) -> float:
    """外側の端で、上面の縁が机から上がる高さ。左右で同じ値にする。

    h_outer を基本にし、傾きを小さくしてスイッチの柱やプレートの下の角が底に
    近づくときは、そのぶん持ち上げる。
    """
    th = math.radians(p.tilt)
    reach = p.sw_body / 2.0
    drop = 0.0
    for side in ("left", "right"):
        lay = make_layout(p, side)
        lows = [(lay.u_out - p.seam * math.tan(math.radians(p.tilt_nominal)) + p.wall, p.bottom)]
        for key in lay.keys:
            w = -key.sink - p.sw_below - p.sw_post
            lows += [(key.u - reach, w), (key.u + reach, w)]
        drop = min([drop] + [(u - lay.u_out) * math.sin(th) + (w - p.top) * math.cos(th) for u, w in lows])
    return max(p.h_outer, p.foot_h + 1.4 - drop)


def frame(p: Params, lay: Layout) -> tuple[np.ndarray, np.ndarray]:
    """キー面の座標から机の座標への回転と平行移動。

    机の座標は x が内側、y が奥、z が上。外側の上面の縁が (x, z) = (0, outer_height) に来る。
    """
    rot = geom.rot_y(-p.tilt)
    trans = rot @ np.array([-lay.u_out, 0.0, -p.top]) + np.array([0.0, 0.0, outer_height(p)])
    return rot, trans


def to_world(p: Params, lay: Layout, m: Manifold) -> Manifold:
    rot, trans = frame(p, lay)
    return geom.move(m, rot, trans)


def world_point(p: Params, lay: Layout, uvw) -> np.ndarray:
    rot, trans = frame(p, lay)
    return rot @ np.asarray(uvw, dtype=float) + trans


# ---- 外形（キー面の座標）---------------------------------------------------------

@dataclass(frozen=True)
class Exterior:
    top: Plane
    sides: dict            # 名前 -> Plane。SIDE_ORDER の順
    corners: tuple         # 縦の角の面取り
    top_chamfers: tuple

    @property
    def walls(self) -> tuple:
        """上面以外の、まわりの面（側面と縦の角の面取り）。"""
        return tuple(self.sides[k] for k in SIDE_ORDER) + self.corners

    @property
    def all(self) -> tuple:
        return (self.top,) + self.walls + self.top_chamfers


def exterior(p: Params, lay: Layout) -> Exterior:
    th0 = math.radians(p.tilt_nominal)
    kappa = math.radians(p.tilt_nominal - p.lean_nominal)
    top_w = p.top
    up = np.array([math.sin(th0), 0.0, math.cos(th0)])   # 基準の傾きでの鉛直（キー面の座標）
    centre = np.array([(lay.u_out + lay.u_in) / 2, (lay.v_front + lay.v_back) / 2, top_w - 5.0])

    sides = {
        "front": Plane.through((0.0, -1.0, 0.0), (0.0, lay.v_front, top_w)),
        "cheek": Plane.through((math.cos(kappa), 0.0, -math.sin(kappa)), (lay.u_in, 0.0, top_w)),
        "back": Plane.through((0.0, 1.0, 0.0), (0.0, lay.v_back, top_w)),
        "outer": Plane.through((-math.cos(th0), 0.0, math.sin(th0)), (lay.u_out, 0.0, top_w)),
    }
    a = np.array([lay.u_out + p.back_cut_u, lay.v_back, top_w])
    b = np.array([lay.u_out, lay.v_back - p.back_cut_v, top_w])
    n = np.cross(up, a - b)
    diag = Plane.through(n, a)
    if diag.value(centre) > 0:
        diag = Plane.through(-n, a)
    sides["diag"] = diag

    ordered = [sides[k] for k in SIDE_ORDER]
    corners = []
    for i, a_pl in enumerate(ordered):
        b_pl = ordered[(i + 1) % len(ordered)]
        if float(a_pl.n @ b_pl.n) < math.cos(math.radians(30.0)):   # ほぼ平らにつながる角は落とさない
            corners.append(geom.chamfer(a_pl, b_pl, p.chamfer_corner))
    corners = tuple(corners)
    top = Plane((0.0, 0.0, 1.0), top_w)
    top_chamfers = tuple(geom.chamfer(top, pl, p.chamfer_top) for pl in tuple(ordered) + corners)
    return Exterior(top, sides, corners, top_chamfers)


def exterior_solid(p: Params, lay: Layout) -> Manifold:
    """外形（キー面の座標。下は切っていない）。"""
    return geom.convex(exterior(p, lay).all)


def seam_outline(p: Params, lay: Layout) -> CrossSection:
    """合わせ目の高さでの外形の輪郭。"""
    return exterior_solid(p, lay).slice(p.top - p.seam).simplify(geom.TINY)


def seat_levels(p: Params) -> tuple[float, float]:
    """合わせ目の高さと、45° の当たり面が終わる高さ（w）。"""
    w_seam = p.top - p.seam
    return w_seam, w_seam - (p.wall - p.seat_land)


def seat_solid(p: Params, lay: Layout) -> Manifold:
    """合わせ目より上と、そこから 45° で内へ下がる当たり面より上（キー面の座標）。

    プレートはこの形に外形を掛けたもの、傾斜台は外形からこの形を引いたもの。
    """
    outline = seam_outline(p, lay)
    w_seam, w_low = seat_levels(p)
    above = geom.box(-geom.BIG, geom.BIG, -geom.BIG, geom.BIG, w_seam, w_seam + 60.0)
    # 当たり面は壁の内面より少し先まで延ばす。内面の上端とぴったり重なる稜を作らないため。
    over = 0.4
    cone = geom.loft(geom.grow(outline, -(p.wall + over)), w_low - over,
                     geom.grow(outline, -p.seat_land), w_seam + EPS)
    return above + cone


def plug_solid(p: Params, lay: Layout, shrink: float = 0.0) -> Manifold:
    """プレートの差し込み部（傾斜台の壁の内側に入る部分。キー面の座標）。

    合わせ目の輪郭をまっすぐ下ろした形を、壁の内面から plug_gap 離した面で切る。
    内側の壁はキー面に垂直な向きより寝ているので、そちらは下ほど狭くなる。
    shrink を渡すと、側面をそのぶん内へ寄せた形になる。
    """
    inset = p.wall + p.plug_gap + shrink
    _w_seam, w_low = seat_levels(p)
    # 角は丸める。差し込みやすくなり、当たり面の稜と頂点がぴったり重なるのも避けられる。
    corner_r = 1.2
    outline = geom.grow(geom.grow(seam_outline(p, lay), -(inset + corner_r)), corner_r, round_=True)
    straight = geom.prism(outline, p.bottom, w_low + EPS)
    return straight ^ geom.convex([pl.shifted(-inset) for pl in exterior(p, lay).walls])


# ---- 机の座標での外形 -----------------------------------------------------------

def _skirt(wall: Plane, floor: Plane, c: float) -> Plane:
    """壁と底の稜を落とす面。ふつうは、両方の面で稜から c のところを通る。

    傾きを基準より大きくすると、外側の壁は上へ行くほど外へ倒れ、この面は 45° より寝る。
    そのときは、底で稜から c のところを通る 45° の面にする（支えなしで印刷できる角度に収める）。
    """
    plane = geom.chamfer(wall, floor, c)
    if -plane.n[2] <= math.sin(math.radians(45.0)) + 1e-9:
        return plane
    flat = geom.unit((wall.n[0], wall.n[1], 0.0))        # 壁の向きの、水平な成分
    z = -floor.d
    reach = (wall.d - wall.n[2] * z) / float(wall.n @ flat)   # 底の高さで、壁の面までの距離
    return Plane.through(flat + np.array([0.0, 0.0, -1.0]), (reach - c) * flat + np.array([0.0, 0.0, z]))


def _world_planes(p: Params, lay: Layout):
    rot, trans = frame(p, lay)
    ext = exterior(p, lay)
    walls = [pl.moved(rot, trans) for pl in ext.walls]
    tops = [pl.moved(rot, trans) for pl in (ext.top,) + ext.top_chamfers]
    floor = Plane((0.0, 0.0, -1.0), -p.foot_h)          # z >= foot_h
    skirts = [_skirt(pl, floor, p.foot_inset) for pl in walls]
    return walls, tops, floor, skirts


def body(p: Params, lay: Layout) -> Manifold:
    """足を除いた外形（机の座標）。底のまわりは 45° で内へ絞る。"""
    walls, tops, floor, skirts = _world_planes(p, lay)
    return geom.convex(walls + tops + [floor] + skirts)


def foot_outline(p: Params, lay: Layout) -> CrossSection:
    under = body(p, lay).slice(p.foot_h + 0.01).simplify(geom.TINY)
    return geom.grow(under, -p.foot_step)


def foot(p: Params, lay: Layout) -> Manifold:
    return geom.prism(foot_outline(p, lay), 0.0, p.foot_h + EPS)


def cavity(p: Params, lay: Layout) -> Manifold:
    """傾斜台の中の空洞（机の座標）。壁の厚みぶん内側。上は開いている。"""
    walls, _tops, _floor, skirts = _world_planes(p, lay)
    inner = [pl.shifted(-p.wall) for pl in walls + skirts]
    inner.append(Plane((0.0, 0.0, -1.0), -p.foot_h))
    return geom.convex(inner)


CABLE_SHOULDER = 2.5   # ケーブルの窓の上の角を 45° で落とす量


def cable_window(p: Params, lay: Layout) -> Manifold:
    """奥の壁の窓。配線したあとのケーブルを外へ出す。底は閉じているので、開口はここだけ。

    下端は、底の上面から cable_sill だけ上（足のまわりの 45° の面取りより上）。上の角は
    45° で落とし、残る平らな天井を支えなしで渡せる幅にする。
    """
    x_in = world_point(p, lay, (lay.u_in, 0.0, p.top))[0]
    xc = x_in - 16.0
    half, c = p.cable_w / 2.0, CABLE_SHOULDER
    z0 = p.foot_h + p.cable_sill
    z1 = z0 + p.cable_h
    y0, y1 = lay.v_back - p.wall - 3.0, lay.v_back + 2.0
    pts = []
    for y in (y0, y1):
        pts += [(xc - half, y, z0), (xc + half, y, z0),
                (xc + half, y, z1 - c), (xc + half - c, y, z1),
                (xc - half + c, y, z1), (xc - half, y, z1 - c)]
    return Manifold.hull_points(np.asarray(pts, dtype=float))


def pad_centres(p: Params, lay: Layout) -> list[np.ndarray]:
    """ゴム足を貼る位置（机の座標の x, y）。足の四隅で、ゴム足の縁が足の縁から pad_edge 内側。

    順に、外側の手前、内側の手前、内側の奥、外側の奥。倒れにくさを決めるのは内側の 2 枚。
    """
    reach = geom.grow(foot_outline(p, lay), -(p.pad_d / 2.0 + p.pad_edge))
    pts = geom.polygons(reach)[0]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    out = []
    for corner in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])):
        d = np.linalg.norm(pts - np.asarray(corner), axis=1)
        out.append(pts[int(d.argmin())])
    return out


def pad_marks(p: Params, lay: Layout) -> Manifold:
    """ゴム足を貼る位置の目印。底に浅い輪の溝を彫る（輪の内側に貼る）。足の縁には肉を残す。"""
    keep = geom.grow(foot_outline(p, lay), -p.pad_edge)
    r0 = p.pad_d / 2.0 + 0.2
    marks = None
    for c in pad_centres(p, lay):
        ring = CrossSection.circle(r0 + p.pad_mark_w, 64) - CrossSection.circle(r0, 64)
        ring = ring.translate((float(c[0]), float(c[1]))) ^ keep
        marks = ring if marks is None else marks + ring
    return geom.prism(marks, -1.0, p.pad_mark_depth)


def base(p: Params, lay: Layout) -> Manifold:
    """傾斜台（机の座標。印刷する向きも同じ）。底は足と同じ厚みの板で閉じている。"""
    shell = body(p, lay) + foot(p, lay)
    shell -= cavity(p, lay)
    shell -= to_world(p, lay, seat_solid(p, lay))
    shell -= cable_window(p, lay)
    shell -= pad_marks(p, lay)
    return shell
