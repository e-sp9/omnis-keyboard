"""キー面の上の配置。キー面の座標 (u, v) で、左右とも内側を +u として決める。

文字キーは寸法から直接置く。親指キー、球、ホイール、音声入力キーは、
「キャップどうしが当たらない」「穴と外形の間に肉が残る」という条件から位置を求める。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from manifold3d import CrossSection

from . import geom
from .params import Params

EDGE_MIN = 1.0   # 抜き穴と、プレートの差し込み部の側面の間に残す肉
VOICE_EDGE = 1.4  # 音声入力キーの穴では、板が薄いぶん多めに残す


@dataclass(frozen=True)
class Key:
    name: str
    kind: str            # "alpha" | "thumb" | "voice"
    u: float
    v: float
    rot: float = 0.0
    cap_w: float = 16.5
    cap_d: float = 16.5
    sink: float = 0.0    # スイッチが載る面をどれだけ下げるか

    def cap_poly(self, grow: float = 0.0) -> np.ndarray:
        return geom.rect_corners(self.u, self.v, self.cap_w + 2 * grow, self.cap_d + 2 * grow, self.rot)

    def cap_section(self, grow: float = 0.0) -> CrossSection:
        return geom.rect(self.u, self.v, self.cap_w + 2 * grow, self.cap_d + 2 * grow, self.rot)

    def square(self, size: float) -> CrossSection:
        return geom.rect(self.u, self.v, size, size, self.rot)


@dataclass(frozen=True)
class Layout:
    side: str                       # "left" | "right"
    keys: tuple[Key, ...]
    u_out: float
    u_in: float
    v_front: float
    v_back: float
    outline: tuple[tuple[float, float], ...]   # 上面の輪郭。反時計回り
    ball: tuple[float, float]                  # 球の中心（左ではホイールの奥行きの基準）
    ball_hole_r: float
    wheel: tuple[float, float, float, float] | None   # 窓 (u0, u1, v0, v1)。左だけ
    lip: tuple[float, float, float, float] | None     # 仕切り (u0, u1, v0, v1)。左だけ

    def key(self, name: str) -> Key:
        for k in self.keys:
            if k.name == name:
                return k
        raise KeyError(name)

    @property
    def alphas(self) -> tuple[Key, ...]:
        return tuple(k for k in self.keys if k.kind == "alpha")

    @property
    def thumbs(self) -> tuple[Key, ...]:
        return tuple(k for k in self.keys if k.kind == "thumb")


# ---- 2次元の距離 ---------------------------------------------------------------

def _seg_point(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - p))


def _inside(poly: np.ndarray, p: np.ndarray) -> bool:
    n = len(poly)
    sign = 0.0
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cross) < 1e-12:
            continue
        if sign == 0.0:
            sign = math.copysign(1.0, cross)
        elif math.copysign(1.0, cross) != sign:
            return False
    return True


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def poly_point_dist(poly: np.ndarray, p) -> float:
    """凸多角形と点の距離。中なら 0。"""
    p = np.asarray(p, dtype=float)
    if _inside(poly, p):
        return 0.0
    n = len(poly)
    return min(_seg_point(poly[i], poly[(i + 1) % n], p) for i in range(n))


def poly_dist(a: np.ndarray, b: np.ndarray) -> float:
    """凸多角形どうしの距離。重なっていれば 0。"""
    na, nb = len(a), len(b)
    for i in range(na):
        for j in range(nb):
            if _segments_cross(a[i], a[(i + 1) % na], b[j], b[(j + 1) % nb]):
                return 0.0
    if _inside(a, b[0]) or _inside(b, a[0]):
        return 0.0
    best = math.inf
    for i in range(na):
        for q in b:
            best = min(best, _seg_point(a[i], a[(i + 1) % na], q))
    for j in range(nb):
        for q in a:
            best = min(best, _seg_point(b[j], b[(j + 1) % nb], q))
    return best


# ---- 配置 ----------------------------------------------------------------------

def _alpha_keys(p: Params) -> list[Key]:
    keys = []
    for col, shift in enumerate(p.stagger):
        for row in range(p.rows):
            keys.append(Key(f"c{col}r{row}", "alpha", p.pitch * col, shift + p.pitch * row,
                            cap_w=p.cap, cap_d=p.cap))
    return keys


def _plug_inset(p: Params) -> float:
    return p.wall + p.plug_gap


def _levels(p: Params, step: float = 0.1) -> list[float]:
    n = int(round((p.top - p.bottom) / step))
    return [p.bottom + i * step for i in range(n + 1)]


def side_offset(p: Params, w: float) -> float:
    """キー面に垂直な壁（手前など）で、高さ w に肉がある範囲が外形から引っ込む量。

    合わせ目より上は外形まで、そこから 45° の当たり面、その下は差し込み部の側面。
    """
    w_seam = p.top - p.seam
    w_low = w_seam - (p.wall - p.seat_land)
    if w >= w_seam:
        return 0.0
    if w >= w_low:
        return p.seat_land + (w_seam - w)
    return _plug_inset(p)


def inner_limit(p: Params, u_in: float, w: float) -> float:
    """内側で、高さ w にプレートの肉がある u の上限。

    内側の壁はキー面に垂直な向きより寝ているので、下へ行くほど外側（-u）へ逃げる。
    """
    kappa = math.radians(p.tilt_nominal - p.lean_nominal)
    w_seam = p.top - p.seam
    w_low = w_seam - (p.wall - p.seat_land)
    if w >= w_seam:
        return u_in - (p.top - w) * math.tan(kappa)
    if w >= w_low:
        return u_in - p.seam * math.tan(kappa) - p.seat_land - (w_seam - w)
    return u_in - (p.top - w) * math.tan(kappa) - _plug_inset(p) / math.cos(kappa)


def ball_centre_w(p: Params) -> float:
    return p.top + p.ball_rise - p.ball_d / 2.0


def ball_hole_radius(p: Params, w: float) -> float:
    """球の穴の半径。球の中心より下は球に沿ってすぼめる。"""
    r = p.ball_d / 2.0 + p.ball_gap
    dw = ball_centre_w(p) - w
    if dw <= 0:
        return r
    return math.sqrt(max(r * r - dw * dw, 0.0))


def wheel_axis_w(p: Params) -> float:
    return p.top + p.wheel_rise - p.wheel_d / 2.0


def wheel_flat_w(p: Params) -> float:
    """窓の内側の端が、ここより下で 45° に引っ込み始める高さ。ホイールの面取りより少し下。"""
    return wheel_axis_w(p) - (p.wheel_d / 2.0 - p.wheel_end) - 0.25


def wheel_end_offset(p: Params, w: float) -> float:
    """ホイールの端の面取りに合わせて、窓の内側の端を高さ w でどれだけ引っ込めるか。"""
    return max(0.0, wheel_flat_w(p) - w)


CRADLE = 1.6   # 軸受けの溝が窓の端から伸びる長さ


def _ball_centre(p: Params, alphas: list[Key], u_in: float) -> tuple[float, float, float]:
    seat_w = ball_centre_w(p) - p.ball_d / 2.0 * math.cos(math.radians(p.ball_seat_deg))
    if seat_w < p.bottom + 0.3:
        need = -(seat_w - 0.3)
        raise RuntimeError(f"球を受ける突起がプレートの厚みに収まりません。plate_t を {need:.1f} 以上にするか、"
                           "ball_rise を大きくしてください。")
    r_hole = p.ball_d / 2.0 + p.ball_gap
    u_b = min(inner_limit(p, u_in, w) - EDGE_MIN - ball_hole_radius(p, w) for w in _levels(p))
    u_b = math.floor(u_b * 20.0) / 20.0
    keep = r_hole + 0.8 + p.well_gap   # 穴とくぼみの間に 0.8 の壁を残す
    polys = [k.cap_poly() for k in alphas]
    v = min(k.v for k in alphas) + p.cap     # 明らかに当たる所から下げていく
    while min(poly_point_dist(poly, (u_b, v)) for poly in polys) < keep:
        v -= 0.05
    return u_b, round(v - 0.049, 1), r_hole


def _voice_key(p: Params, alphas: list[Key]) -> Key:
    pinky_front = alphas[0]
    v = pinky_front.v - p.cap / 2.0 - 1.5 - p.voice_cap / 2.0
    return Key("voice", "voice", p.voice_u, round(v, 2), cap_w=p.voice_cap, cap_d=p.voice_cap,
               sink=p.voice_sink)


def _search(us, vs, ok):
    """奥（文字キーに近い側）から順に探し、同じ奥行きなら内側を優先する。"""
    for v in vs:
        for u in us:
            if ok(u, v):
                return round(float(u), 2), round(float(v), 2)
    raise RuntimeError("置ける位置がありません。寸法を見直してください。")


def _frange(start: float, stop: float, step: float):
    n = int(math.floor(abs(stop - start) / abs(step))) + 1
    return [start + i * step for i in range(n)]


@lru_cache(maxsize=8)
def _common(p: Params):
    """左右に共通の値。手前の縁は、球と音声入力キーの両方が収まる位置にそろえる。"""
    alphas = _alpha_keys(p)
    u_out = -p.cap / 2.0 - p.margin_outer
    u_in = p.pitch * (len(p.stagger) - 1) + p.cap / 2.0 + p.margin_inner
    v_back = max(k.v for k in alphas) + p.cap / 2.0 + p.well_gap + p.rim
    u_b, v_b, r_hole = _ball_centre(p, alphas, u_in)
    voice = _voice_key(p, alphas)
    front_ball = min(v_b - ball_hole_radius(p, w) - EDGE_MIN - side_offset(p, w) for w in _levels(p))
    front_ball = min(front_ball, v_b - r_hole - p.ball_ring - 0.3 - p.chamfer_top)
    front_voice = voice.v - p.hole / 2.0 - VOICE_EDGE - _plug_inset(p)
    v_front = math.floor(min(front_ball, front_voice) * 10.0) / 10.0
    return alphas, u_out, u_in, v_front, v_back, (u_b, v_b), r_hole, voice


def make_layout(p: Params, side: str) -> Layout:
    if side not in ("left", "right"):
        raise ValueError(side)
    alphas, u_out, u_in, v_front, v_back, ball, r_hole, voice = _common(p)
    alpha_polys = [k.cap_poly() for k in alphas]
    rot = p.thumb_rot
    side_gap = p.pitch - p.cap                           # 横に並ぶキャップのすき間
    edge_keep = p.well_gap + EDGE_MIN + _plug_inset(p)   # キャップの端から外形まで
    keys: list[Key] = list(alphas)
    wheel = lip = None

    def clear_of_alphas(poly: np.ndarray) -> bool:
        """奥の文字キーとは thumb_drop、横に並ぶ文字キーとは side_gap あける。"""
        for a in alpha_polys:
            if abs(rot) < 1e-9:
                gap_u = max(a[:, 0].min() - poly[:, 0].max(), poly[:, 0].min() - a[:, 0].max())
                gap_v = max(a[:, 1].min() - poly[:, 1].max(), poly[:, 1].min() - a[:, 1].max())
                if gap_u < side_gap - 1e-9 and gap_v < p.thumb_drop - 1e-9:
                    return False
            elif poly_dist(poly, a) < p.thumb_drop - 1e-9:
                return False
        return True

    def inside_front(poly: np.ndarray) -> bool:
        return float(poly[:, 1].min()) >= v_front + edge_keep - 1e-9

    if side == "left":
        # ホイールの窓。内側の縁に寄せ、奥行きは右の球の中心に合わせる。
        win_len = p.wheel_len + 2 * (p.wheel_pad + p.wheel_play)
        win_dep = p.wheel_d + 1.0
        axle_low = wheel_axis_w(p) - p.axle_d / 2.0 - 0.2
        u1 = inner_limit(p, u_in, axle_low) - 0.8 - CRADLE
        u1 = min([u1] + [inner_limit(p, u_in, w) - EDGE_MIN + wheel_end_offset(p, w) for w in _levels(p)])
        u1 = math.floor(u1 * 10.0) / 10.0
        u0 = u1 - win_len
        wheel = (u0, u1, ball[1] - win_dep / 2.0, ball[1] + win_dep / 2.0)
        lu1 = u0 - 0.3
        lip = (lu1 - p.lip_w, lu1, wheel[2] - 0.6, wheel[3] + 0.6)
        cap_end = lip[0] - 0.3 - p.well_gap           # Space のキャップの内側の端
        w = p.space_w * p.pitch - (p.pitch - p.cap)

        def ok(u, v):
            poly = geom.rect_corners(u, v, w, p.cap, rot)
            return (clear_of_alphas(poly) and inside_front(poly)
                    and float(poly[:, 0].max()) <= cap_end + 1e-9)

        su, sv = _search(_frange(cap_end - w / 2.0 + 1.0, cap_end - w / 2.0 - 6.0, -0.05),
                         _frange(-12.0, v_front, -0.1), ok)
        keys += [Key("space", "thumb", su, sv, rot, cap_w=w, cap_d=p.cap), voice]
    else:
        ring_keep = r_hole + p.ball_ring + 0.6 + p.well_gap    # Enter のキャップから球の中心まで
        w1 = p.enter_w * p.pitch - (p.pitch - p.cap)

        def ok1(u, v):
            poly = geom.rect_corners(u, v, w1, p.cap, rot)
            return (clear_of_alphas(poly) and inside_front(poly)
                    and poly_point_dist(poly, ball) >= ring_keep)

        hi = ball[0] - r_hole - w1 / 2.0 + 2.0
        eu, ev = _search(_frange(hi, hi - 8.0, -0.05), _frange(-12.0, v_front, -0.1), ok1)
        enter = Key("enter", "thumb", eu, ev, rot, cap_w=w1, cap_d=p.cap)
        enter_poly = enter.cap_poly()

        def ok2(u, v):
            poly = geom.rect_corners(u, v, p.cap, p.cap, rot)
            return (clear_of_alphas(poly) and inside_front(poly)
                    and poly_dist(poly, enter_poly) >= side_gap - 1e-9
                    and float(poly[:, 0].min()) >= u_out + edge_keep)

        keys.append(enter)
        hi2 = eu - w1 / 2.0 - p.cap / 2.0 + 1.0
        try:
            tu, tv = _search(_frange(hi2, hi2 - 8.0, -0.05), _frange(-12.0, v_front, -0.1), ok2)
            keys.append(Key("thumb2", "thumb", tu, tv, rot, cap_w=p.cap, cap_d=p.cap))
        except RuntimeError:
            pass     # 球を大きくすると、2 個めの親指キーを置く場所がなくなる。そのときは付けない

    outline = (
        (u_out, v_front),
        (u_in, v_front),
        (u_in, v_back),
        (u_out + p.back_cut_u, v_back),
        (u_out, v_back - p.back_cut_v),
    )
    return Layout(side, tuple(keys), u_out, u_in, v_front, v_back, outline, ball, r_hole,
                  wheel, lip)


# ---- くぼみ --------------------------------------------------------------------

def wells(p: Params, lay: Layout) -> CrossSection:
    """キーキャップが入るくぼみ（上面から、スイッチが載る面まで）。"""
    parts = [k.cap_section(p.well_gap) for k in lay.keys if k.kind != "voice"]
    cs = geom.union_all(parts)
    return geom.close_gaps(cs, 0.6)


def voice_well(p: Params, lay: Layout) -> CrossSection | None:
    for k in lay.keys:
        if k.kind == "voice":
            return k.square(p.sw_flange + 2 * 0.3)
    return None
