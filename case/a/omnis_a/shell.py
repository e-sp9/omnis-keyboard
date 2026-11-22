"""外形と傾斜台。

外形は「平らな面の半空間の共通部分」として作る凸多面体で、面取りも半空間を足して作る。

プレートが持つ面（上面、そのまわりの面取りの帯、合わせ目までの細い壁）は、キー面の座標で
決めてあり、姿勢を変えても変わらない。傾斜台の壁は、合わせ目の輪郭から机へ下ろす。
手前と外側の壁は机に垂直に立て、奥と内側の壁はキー面に垂直な向きまで寝かせる。こうすると、
プレートの下の差し込み部をキー面に垂直に抜き差しでき、キーを押す力の向きも足の内側に落ちる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from manifold3d import CrossSection, Manifold

from . import geom, layout as lay_mod
from .geom import EPS, Plane
from .layout import Layout, make_layout
from .params import Params

MAINS = ("front", "cheek", "back", "diag", "outer")   # 上から見て反時計回り
SEAT_OVER = 1.2        # 当たり面を、壁の内面より先まで延ばす量
CABLE_SHOULDER = 2.5   # ケーブルの窓の上の角を 45° で落とす量


# ---- 姿勢 ----------------------------------------------------------------------

def posture(tilt: float, slope: float) -> np.ndarray:
    """キー面の座標から机の座標への回転。列は、キーの行の向き、列の向き、キー面の法線。

    行の向き（外側から内側へ）は水平から tilt だけ上がり、列の向き（手前から奥へ）は slope だけ
    上がる。列の向きは、上から見るとまっすぐ奥を向く。机の座標は x が内側、y が奥、z が上。
    """
    a, b = math.radians(tilt), math.radians(slope)
    ev = np.array([0.0, math.cos(b), math.sin(b)])
    y = -math.sin(a) * math.tan(b)
    eu = np.array([math.sqrt(max(1.0 - y * y - math.sin(a) ** 2, 0.0)), y, math.sin(a)])
    return np.stack([eu, ev, np.cross(eu, ev)], axis=1)


def steepest(p: Params) -> float:
    """キー面のいちばん急な向きの傾き（左右と前後を合わせた傾き）。"""
    return math.degrees(math.acos(posture(p.tilt, p.slope)[2, 2]))


@lru_cache(maxsize=32)
def base_height(p: Params) -> float:
    """外側の手前の角で、上面の縁が机から上がる高さ。左右で同じ値にする。

    h_low を基本にし、スイッチの下端が底の板の上面から clear_floor 以上浮く高さまで上げる。
    プレートの下の角、球、ホイールも底の板に当たらない高さにする。
    """
    up = posture(p.tilt, p.slope)[2]
    steep = math.hypot(up[0], up[1])
    need = p.h_low
    for side in ("left", "right"):
        lay = make_layout(p, side)
        ref = np.array([lay.u_out, lay.v_front, p.top])

        def lift(point, gap: float) -> float:
            return p.floor_top + gap - float(up @ (np.asarray(point, dtype=float) - ref))

        inset = p.wall + p.plug_gap
        for u, v in ((lay.u_out + inset, lay.v_front + inset), (lay.u_in - inset, lay.v_front + inset),
                     (lay.u_in - inset, lay.v_back - inset), (lay.u_out + inset, lay.v_back - p.back_cut_v)):
            need = max(need, lift((u, v, p.bottom), 0.6))
        for key in lay.keys:
            w = -key.sink - p.sw_below
            corner = p.sw_body / 2.0 * (abs(up[0]) + abs(up[1]))     # ハウジングの底の、いちばん低い角
            need = max(need,
                       lift((key.u, key.v, w), p.clear_floor) + corner,
                       lift((key.u, key.v, w - p.sw_post), p.clear_floor) + 2.4 * steep,
                       lift((key.u, key.v, w - p.sw_pin), p.clear_floor) + 6.9 * steep)   # 端子。どの向きに挿しても
        if lay.wheel is None:
            centre = (lay.ball[0], lay.ball[1], lay_mod.ball_centre_w(p))
            need = max(need, lift(centre, 0.6) + p.ball_d / 2.0)
        else:
            u0, u1, v0, v1 = lay.wheel
            for u in (u0, u1):
                need = max(need, lift((u, (v0 + v1) / 2.0, lay_mod.wheel_axis_w(p)), 0.6) + p.wheel_d / 2.0)
    return need


def frame(p: Params, lay: Layout) -> tuple[np.ndarray, np.ndarray]:
    """キー面の座標から机の座標への回転と平行移動。

    外側の手前の上面の角が (x, y, z) = (0, v_front, base_height) に来る。
    """
    rot = posture(p.tilt, p.slope)
    ref = np.array([lay.u_out, lay.v_front, p.top])
    return rot, np.array([0.0, lay.v_front, base_height(p)]) - rot @ ref


def to_world(p: Params, lay: Layout, m: Manifold) -> Manifold:
    rot, trans = frame(p, lay)
    return geom.move(m, rot, trans)


def world_point(p: Params, lay: Layout, uvw) -> np.ndarray:
    rot, trans = frame(p, lay)
    return rot @ np.asarray(uvw, dtype=float) + trans


# ---- 壁（キー面の座標）-----------------------------------------------------------

def _edges(p: Params, lay: Layout) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """上面の輪郭の辺ごとに、名前、辺の上の点、外向きの単位ベクトル（キー面の中で）。"""
    diag = geom.unit((-p.back_cut_v, p.back_cut_u))
    return [
        ("front", np.array([0.0, lay.v_front]), np.array([0.0, -1.0])),
        ("cheek", np.array([lay.u_in, 0.0]), np.array([1.0, 0.0])),
        ("back", np.array([0.0, lay.v_back]), np.array([0.0, 1.0])),
        ("diag", np.array([lay.u_out + p.back_cut_u, lay.v_back]), diag),
        ("outer", np.array([lay.u_out, 0.0]), np.array([-1.0, 0.0])),
    ]


def _upright_slope(m2: np.ndarray, rot: np.ndarray, lean: float = 0.0) -> float:
    """m2 を外向きとする壁を机に垂直に立てたときの、キー面の座標での傾き。

    戻り値は、キー面に垂直に 1 下がるごとに、壁が外へ出る量（負なら内へ入る）。
    lean を渡すと、壁の上が内へ倒れる向きに寝かせ、壁の面と鉛直のなす角を lean にする。
    """
    along = rot @ np.array([-m2[1], m2[0], 0.0])           # 辺の向き
    flat = geom.unit(np.cross(along, [0.0, 0.0, 1.0]))     # 辺に垂直で水平な向き
    if flat @ (rot @ np.array([m2[0], m2[1], 0.0])) < 0:
        flat = -flat
    rise = geom.unit(np.cross(along, flat))
    if rise[2] < 0:
        rise = -rise
    tip = math.asin(min(1.0, math.sin(math.radians(lean)) / rise[2]))
    n = rot.T @ (math.cos(tip) * flat + math.sin(tip) * rise)
    return float(n[2] / math.hypot(n[0], n[1]))


def _plug_slope(p: Params, name: str) -> float:
    """プレートの差し込み部の側面の傾き。内側だけ、キー面に垂直な向きから plug_kappa 寝かせてある。"""
    return -math.tan(math.radians(p.plug_kappa)) if name == "cheek" else 0.0


def _slope(p: Params, name: str, m2: np.ndarray, rot: np.ndarray, plug: bool) -> float:
    """壁の傾き。机に垂直に立てるのが基本で、差し込み部の側面より内へは倒さない。"""
    if plug:
        return _plug_slope(p, name)
    lean = p.cheek_lean if name == "cheek" else 0.0
    return max(_upright_slope(m2, rot, lean), _plug_slope(p, name))


@lru_cache(maxsize=32)
def _nominal(p: Params, lay: Layout) -> tuple[dict, tuple]:
    """基準の姿勢での壁と、縦の角の面取り。プレートの端面はこれで決まる。"""
    rot = posture(p.tilt_nominal, p.slope_nominal)
    mains = {}
    for name, point, m2 in _edges(p, lay):
        s = _slope(p, name, m2, rot, plug=False)
        mains[name] = Plane.through((m2[0], m2[1], s), (point[0], point[1], p.top))
    corners = []
    for i, a in enumerate(MAINS):
        b = MAINS[(i + 1) % len(MAINS)]
        if float(mains[a].n @ mains[b].n) < math.cos(math.radians(30.0)):   # ほぼ平らにつながる角は落とさない
            corners.append((a, b, geom.chamfer(mains[a], mains[b], p.chamfer_corner)))
    return mains, tuple(corners)


def _trace(pl: Plane, w: float) -> tuple[np.ndarray, float]:
    """面を、キー面に平行な高さ w の面で切った直線。外向きの単位ベクトルと、原点からの距離。"""
    k = math.hypot(pl.n[0], pl.n[1])
    return np.array([pl.n[0], pl.n[1]]) / k, (pl.d - pl.n[2] * w) / k


def walls(p: Params, lay: Layout, tilt: float | None = None, slope: float | None = None,
          plug: bool = False) -> list[tuple[str, Plane]]:
    """ある姿勢での壁（キー面の座標）。縦の角の面取りも含む。

    どの姿勢でも、合わせ目の高さでは基準の姿勢の輪郭を通る。プレートと傾斜台の境目が
    ずれないようにするためで、姿勢で変わるのは、そこから下の向きだけ。
    plug を渡すと、プレートの差し込み部の側面の向きになる（姿勢によらない）。
    """
    rot = posture(p.tilt if tilt is None else tilt, p.slope if slope is None else slope)
    mains0, corners0 = _nominal(p, lay)
    w_seam = p.top - p.seam
    mains = {}
    for name in MAINS:
        m2, dist = _trace(mains0[name], w_seam)
        s = _slope(p, name, m2, rot, plug)
        mains[name] = Plane.through((m2[0], m2[1], s), (m2[0] * dist, m2[1] * dist, w_seam))
    out = [(name, mains[name]) for name in MAINS]
    for a, b, pl0 in corners0:
        m2, dist = _trace(pl0, w_seam)
        edge = np.cross(mains[a].n, mains[b].n)                 # 2 枚の壁が交わる稜の向き
        n = np.cross(np.array([-m2[1], m2[0], 0.0]), edge)
        if n[0] * m2[0] + n[1] * m2[1] < 0:
            n = -n
        out.append((f"{a}-{b}", Plane.through(n, (m2[0] * dist, m2[1] * dist, w_seam))))
    return out


# ---- プレートの外形（キー面の座標）-------------------------------------------------

@dataclass(frozen=True)
class Exterior:
    top: Plane
    walls: tuple           # (名前, Plane)。基準の姿勢での壁と、縦の角の面取り
    bands: tuple           # 上面と壁の間の、面取りの帯
    softs: tuple           # 帯の上下の縁を落とす細い面

    @property
    def all(self) -> tuple:
        return (self.top,) + tuple(pl for _n, pl in self.walls) + self.bands + self.softs


def exterior(p: Params, lay: Layout) -> Exterior:
    top = Plane((0.0, 0.0, 1.0), p.top)
    ws = tuple(walls(p, lay, p.tilt_nominal, p.slope_nominal))
    bands = tuple(geom.chamfer(top, pl, p.chamfer_top) for _n, pl in ws)
    softs = []
    for (_n, pl), band in zip(ws, bands):
        softs += [geom.chamfer(top, band, p.chamfer_soft), geom.chamfer(band, pl, p.chamfer_soft)]
    return Exterior(top, ws, bands, tuple(softs))


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
    # 当たり面は壁の内面より先まで延ばす。寝ている壁の内面にも届き、内面の上端とぴったり
    # 重なる稜も作らないため。
    cone = geom.loft(geom.grow(outline, -(p.wall + SEAT_OVER)), w_low - SEAT_OVER,
                     geom.grow(outline, -p.seat_land), w_seam + EPS)
    return above + cone


def plug_solid(p: Params, lay: Layout, shrink: float = 0.0) -> Manifold:
    """プレートの差し込み部（傾斜台の壁の内側に入る部分。キー面の座標）。

    合わせ目の輪郭をキー面に垂直に下ろした形を、壁の内面から plug_gap 離す。内側だけは、
    キー面に垂直な向きから plug_kappa 寝かせた面で切る。左右の傾きを大きくした傾斜台でも、
    内側の壁をそこまで立てられるようにするため。shrink を渡すと、側面をそのぶん内へ寄せる。
    """
    inset = p.wall + p.plug_gap + shrink
    _w_seam, w_low = seat_levels(p)
    # 角は丸める。差し込みやすくなり、当たり面の稜と頂点がぴったり重なるのも避けられる。
    corner_r = 1.2
    outline = geom.grow(geom.grow(seam_outline(p, lay), -(inset + corner_r)), corner_r, round_=True)
    straight = geom.prism(outline, p.bottom, w_low + EPS)
    leaning = [pl.shifted(-inset) for _n, pl in walls(p, lay, plug=True) if abs(pl.n[2]) > 1e-9]
    return straight ^ geom.convex(leaning) if leaning else straight


# ---- 机の座標での外形 -----------------------------------------------------------

def _skirt(wall: Plane, z1: float) -> Plane:
    """壁の下端（高さ z1）から、45° で内へ絞る面。底の台へつながる。"""
    flat = geom.unit((wall.n[0], wall.n[1], 0.0))
    reach = (wall.d - wall.n[2] * z1) / float(wall.n @ flat)     # 高さ z1 で、壁の面までの距離
    return Plane.through(flat + np.array([0.0, 0.0, -1.0]), reach * flat + np.array([0.0, 0.0, z1]))


def _world_planes(p: Params, lay: Layout):
    rot, trans = frame(p, lay)
    sides = [pl.moved(rot, trans) for _n, pl in walls(p, lay)]
    skirts = [_skirt(pl, p.floor_top) for pl in sides]
    lid = Plane((0.0, 0.0, 1.0), p.top - p.seam + 1.0).moved(rot, trans)   # 合わせ目の少し上まで
    return sides, skirts, lid


def wall_leans(p: Params, lay: Layout) -> dict[str, float]:
    """壁ごとの、垂直から寝ている角度（度）。上が内へ倒れる向きが正。"""
    rot, trans = frame(p, lay)
    return {name: math.degrees(math.asin(float(pl.moved(rot, trans).n[2])))
            for name, pl in walls(p, lay) if name in MAINS}


def body(p: Params, lay: Layout) -> Manifold:
    """台を除いた外形（机の座標）。壁の下端から 45° で内へ絞る。"""
    sides, skirts, lid = _world_planes(p, lay)
    return geom.convex(sides + skirts + [lid, Plane((0.0, 0.0, -1.0), -p.foot_h)])


def foot_outline(p: Params, lay: Layout) -> CrossSection:
    """底の台の輪郭。上の斜面の下端より、ごくわずかに内側（稜をぴったり重ねないため）。"""
    return geom.grow(body(p, lay).slice(p.foot_h + 0.01).simplify(geom.TINY), -2 * EPS)


def foot(p: Params, lay: Layout) -> Manifold:
    return geom.prism(foot_outline(p, lay), 0.0, p.foot_h + EPS)


def _wall_extra(name: str) -> float:
    """空洞を作るとき、壁の厚みに足すわずかな量。隣り合う面で必ず違う値になる。

    キー面に垂直な壁どうしは、内面の角の稜が、当たり面の角の稜とちょうど 1 点で交わる。
    4 枚の面が 1 点に集まると STL にしたときに閉じた形でなくなるので、厚みをずらして避ける。
    """
    if "-" in name:
        return 0.12
    return 0.06 if name == "diag" else 0.0


def cavity(p: Params, lay: Layout) -> Manifold:
    """傾斜台の中の空洞（机の座標）。壁の厚みぶん内側。上は開いていて、下は底の板の上面まで。"""
    sides, skirts, _lid = _world_planes(p, lay)
    names = [name for name, _pl in walls(p, lay)]
    inner = [pl.shifted(-(p.wall + _wall_extra(name))) for name, pl in zip(names, sides)]
    inner += [pl.shifted(-(p.wall + _wall_extra(name))) for name, pl in zip(names, skirts)]
    inner.append(Plane((0.0, 0.0, -1.0), -p.floor_top))
    return geom.convex(inner)


def cable_window(p: Params, lay: Layout) -> Manifold:
    """奥の壁の窓。配線したあとのケーブルを外へ出す。底は閉じているので、開口はここだけ。

    下端は、底の板の上面から cable_sill だけ上。上の角は 45° で落とし、残る平らな天井を
    支えなしで渡せる幅にする。
    """
    xc = world_point(p, lay, (lay.u_in - 16.0, lay.v_back, p.top))[0]
    half, c = p.cable_w / 2.0, CABLE_SHOULDER
    z0 = p.floor_top + p.cable_sill
    z1 = z0 + p.cable_h
    y_back = body(p, lay).bounding_box()[4]
    pts = []
    for y in (y_back - 16.0, y_back + 2.0):
        pts += [(xc - half, y, z0), (xc + half, y, z0),
                (xc + half, y, z1 - c), (xc + half - c, y, z1),
                (xc - half + c, y, z1), (xc - half, y, z1 - c)]
    return Manifold.hull_points(np.asarray(pts, dtype=float))


def pad_centres(p: Params, lay: Layout) -> list[np.ndarray]:
    """ゴム足を貼る位置（机の座標の x, y）。台の四隅で、ゴム足の縁が台の縁から pad_edge 内側。

    順に、外側の手前、内側の手前、内側の奥、外側の奥。
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
    """ゴム足を貼る位置の目印。底に浅い輪の溝を彫る（輪の内側に貼る）。台の縁には肉を残す。"""
    keep = geom.grow(foot_outline(p, lay), -p.pad_edge)
    r0 = p.pad_d / 2.0 + 0.2
    marks = None
    for c in pad_centres(p, lay):
        ring = CrossSection.circle(r0 + p.pad_mark_w, 64) - CrossSection.circle(r0, 64)
        ring = ring.translate((float(c[0]), float(c[1]))) ^ keep
        marks = ring if marks is None else marks + ring
    return geom.prism(marks, -1.0, p.pad_mark_depth)


def base(p: Params, lay: Layout) -> Manifold:
    """傾斜台（机の座標。印刷する向きも同じ）。底は板で閉じている。"""
    shell = body(p, lay) + foot(p, lay)
    shell -= cavity(p, lay)
    shell -= to_world(p, lay, seat_solid(p, lay))
    shell -= cable_window(p, lay)
    shell -= pad_marks(p, lay)
    return shell
