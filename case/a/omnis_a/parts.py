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
    """キャップの下端と上面の高さ（そのキーのスイッチが載る面から。軸の先端が公称の高さのとき）。

    キャップは軸の先端に座るので、先端が低いスイッチでは、キャップもそのぶん低くなる。
    下端は、図面の公差の端（ストロークが長く、ハウジングが高く、軸の先端が低い）のスイッチを
    押し切っても cap_margin 以上が残る高さにする。文字キーのキャップはハウジングをまたぐので、
    つばの上面が相手。音声入力キーの小さいキャップはまたげないので、ハウジングの上端が相手になる。
    どのキャップも、下端と上面は、軸の先端から layer の整数倍の高さにある（params.py の説明）。
    """
    if key.kind == "voice":
        # 要る高さより上で、軸の先端から layer の整数倍になる高さ（印刷すると、下端と十字の穴の底の
        # 間は layer の整数倍にしかならない。足りない側へ丸められないよう、切り上げておく）
        need = p.sw_top + p.sw_height_tol + p.travel_max + p.sw_stem_tol + p.cap_margin
        skirt = p.sw_stem + math.ceil((need - p.sw_stem) / p.layer - 1e-9) * p.layer
        return round(skirt, 6), round(skirt + p.voice_cap_h, 6)
    return p.cap_skirt, p.cap_top


CAP_EDGE = 0.5        # キャップの上面の縁を落とす量
VOICE_TAPER = 0.6     # 音声入力キーの薄いキャップの絞り（斜面が寝すぎない量にする）


def cap_flat(p: Params, key: Key) -> tuple[float, float, float]:
    """印刷用のキャップの平らな上面。幅の半分、奥行きの半分、角の丸みの半径。"""
    taper = p.cap_taper if key.kind != "voice" else VOICE_TAPER
    half_w, half_d = key.cap_w / 2.0 - (taper + CAP_EDGE), key.cap_d / 2.0 - (taper + CAP_EDGE)
    return half_w, half_d, min(max(0.3, p.cap_round - CAP_EDGE), half_w - 0.05, half_d - 0.05)


def keycap(p: Params, key: Key) -> Manifold:
    """キーキャップ。原点はスイッチが載る面の、キーの中心。上が +z。"""
    w, d = key.cap_w, key.cap_d
    z0, z1 = cap_heights(p, key)
    taper = p.cap_taper if key.kind != "voice" else VOICE_TAPER
    edge = CAP_EDGE
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


# プレビューだけに使う、丸みのあるキーキャップの形（スイッチが載る面からの高さ、mm）
LOOK = {
    "alpha": dict(top=12.2, taper=2.2, fillet=0.9, corner=1.7, dish=0.35),
    "thumb": dict(top=12.2, taper=2.2, fillet=0.9, corner=1.7, dish=0.3),
    "voice": dict(top=9.2, taper=1.2, fillet=0.6, corner=1.4, dish=0.0),
}


def keycap_look(p: Params, key: Key) -> Manifold:
    """プレビュー用のキーキャップ。参考画像に近い、角と縁を丸めた形。印刷用の形（keycap）とは別。

    下端の高さと外形は印刷用と同じ。側面は少し絞り、上の縁を丸め、上面を浅い円筒面でくぼませる。
    中は詰まっていて、当たりの検査には使わない。
    """
    look = LOOK[key.kind]
    w, d = key.cap_w, key.cap_d
    z0, _z1 = cap_heights(p, key)
    z1, taper, rho = look["top"], look["taper"], look["fillet"]
    band = min(0.8, (z1 - z0) / 4.0)
    # 斜めの側面と、上の縁の丸み（半径 rho）が滑らかにつながる角度を求める
    tip = 0.3
    for _ in range(30):
        tx, tz = taper - rho * math.cos(tip), z1 - rho + rho * math.sin(tip)
        tip = math.atan2(tx, tz - (z0 + band))
    profile = [(0.0, z0), (0.0, z0 + band)]
    steps = 10
    for i in range(steps + 1):
        a = tip + (math.pi / 2.0 - tip) * i / steps
        profile.append((taper - rho * math.cos(a), z1 - rho + rho * math.sin(a)))
    rings = [(_rounded(w - 2 * inset, d - 2 * inset, look["corner"] + 0.35 * inset), z) for inset, z in profile]
    body = None
    for low, high in zip(rings[:-1], rings[1:]):
        piece = _stack([low, high])
        body = piece if body is None else body + piece
    if look["dish"] > 0:
        half = (w - 2 * taper) / 2.0
        radius = (half * half + look["dish"] ** 2) / (2.0 * look["dish"])
        scoop = Manifold.cylinder(d + 4.0, radius, radius, 256).translate((0.0, 0.0, -(d + 4.0) / 2.0))
        scoop = geom.move(scoop, geom.rot_x(90.0), np.array([0.0, 0.0, z1 - look["dish"] + radius]))
        body -= scoop
    return body


def place_on_key(p: Params, key: Key, m: Manifold, press: float = 0.0) -> Manifold:
    """キーの位置へ置く（キー面の座標）。press は押し込み量。"""
    return geom.move(m, geom.rot_z(key.rot), np.array([key.u, key.v, -key.sink - press]))


def switch(p: Params, tall: float = 0.0) -> tuple[Manifold, Manifold]:
    """Choc V2 のおおよその形。原点はスイッチが載る面。

    プレートより上のハウジングと、プレートより下のハウジング（中央の柱を含む）に分けて返す。
    上は、つば（15 mm 角）、その上のまっすぐな部分（下ハウジングと同じ幅）、上端へすぼまる部分。
    tall を渡すと、上のハウジングの高さをそのぶん増やす（公差の端を確かめるとき）。
    軸は入れていない（キーキャップと一緒に動くため）。底から出る脚は legs で作る。
    """
    f, b = p.sw_flange, p.sw_body
    upper = geom.prism(geom.rect(0, 0, f, f), 0.0, p.sw_flange_t + tall)
    upper += geom.prism(geom.rect(0, 0, b, b), p.sw_flange_t + tall - EPS, p.sw_shoulder + tall)
    upper += geom.loft(geom.rect(0, 0, b, b), p.sw_shoulder + tall - EPS,
                       geom.rect(0, 0, p.sw_crown, p.sw_crown), p.sw_top + tall)
    lower = geom.prism(geom.rect(0, 0, b, b), -p.sw_below, EPS)
    lower += Manifold.cylinder(p.sw_post + 0.2, 2.4, 2.4, 32).translate((0, 0, -p.sw_below - p.sw_post))
    return upper, lower


# ハウジングの底から出る 3 本の脚（図面の基板の穴の位置。スイッチを上から見た向き）。
# 太さは基板の穴の径で代用する（実物の脚より少し太い）。
TERMINALS = ((0.0, 5.9), (5.0, 3.8))    # 電気の端子 2 本。穴は直径 1.20
TERMINAL_D = 1.2
LEG = (-5.0, -5.15)                     # 端子でない脚 1 本。穴は直径 1.60
LEG_D = 1.6
QUARTERS = (0, 1, 2, 3)                 # スイッチを挿せる 4 つの向き（90° 刻み）


def reach(spots, diameter: float) -> float:
    """スイッチの中心から、脚のいちばん遠い所までの距離。"""
    return max(math.hypot(x, y) for x, y in spots) + diameter / 2.0


def legs(p: Params, quarter: int = 0, mirror: bool = False) -> tuple[Manifold, Manifold]:
    """スイッチの脚。電気の端子 2 本と、端子でない脚 1 本に分けて返す。原点はスイッチが載る面。

    quarter は挿す向き。爪の逃げは穴の 4 辺にあるので、スイッチは 90° 刻みの 4 通りに挿せる。
    quarter が 0 のとき、まん中寄りの端子が奥（+v）に来る。1 で外側、2 で手前、3 で内側。
    右手の形は左右を反転して作るので、反転する前の座標では脚の並びも鏡写しになる。
    右手ぶんを作るときは mirror を渡す。
    """
    flip = -1.0 if mirror else 1.0

    def rods(spots, diameter: float) -> Manifold:
        out = None
        for x, y in spots:
            rod = Manifold.cylinder(p.sw_pin + 0.2, diameter / 2.0, diameter / 2.0, 16)
            rod = rod.translate((flip * x, y, -p.sw_below - p.sw_pin))
            out = rod if out is None else out + rod
        return geom.move(out, geom.rot_z(90.0 * quarter), np.zeros(3))

    return rods(TERMINALS, TERMINAL_D), rods((LEG,), LEG_D)


def latch(p: Params, quarter: int = 0) -> Manifold:
    """プレートに掛かる爪（向かい合う 2 面に 1 本ずつ）。原点はスイッチが載る面。

    肩は sw_clip だけ下にあり、そこからハウジングの底までを、出 sw_clip_out、幅 sw_clip_w の
    箱で表す。quarter を 1 にすると、スイッチを 90° 回して挿したときの位置になる。
    """
    b = p.sw_body / 2.0
    one = geom.box(b - EPS, b + p.sw_clip_out, -p.sw_clip_w / 2.0, p.sw_clip_w / 2.0, -p.sw_below, -p.sw_clip)
    both = one + geom.move(one, geom.rot_z(180.0), np.zeros(3))
    return geom.move(both, geom.rot_z(90.0 * quarter), np.zeros(3))


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

def ledge(p: Params, key: Key) -> float:
    """そのキーの穴のふちで、爪を受ける板の厚み。

    ふつうは clip_t。スイッチを沈めた音声入力キーは、スイッチが載る面からプレートの下面までの
    残りが clip_t と同じか薄いとき、逃げを彫らずに、板の下面がそのまま爪を受ける。残りのほうが
    厚いとき（clip_t を薄くしたとき）は、その差だけ下面から逃げを彫る。スイッチが載る面は動かさない。
    """
    return min(p.plate_t - key.sink, p.clip_t)


def clip_relief(p: Params, key: Key, z_low: float, z_ledge: float, open_sides: tuple = (),
                size: float | None = None, reach: float = 5.0) -> Manifold | None:
    """爪の逃げ。穴の各辺の中央を、z_low から爪を受ける板の下（z_ledge）まで外へ広げる。

    辺の番号は 0 が +u、1 が +v、2 が -u、3 が -v（キーを回していれば一緒に回る）。
    スイッチはどの向きにも挿せるよう、4 辺すべてに彫る。open_sides に入れた辺は、
    逃げを reach だけ先まで延ばして、外へ抜く。
    """
    if z_ledge - z_low < 0.05:
        return None
    h = (p.hole if size is None else size) / 2.0
    out = None
    for i in range(4):
        depth = p.clip_d + (reach if i in open_sides else 0.0)
        notch = geom.box(h - EPS, h + depth, -p.clip_w / 2.0, p.clip_w / 2.0, z_low, z_ledge)
        notch = geom.move(notch, geom.rot_z(90.0 * i), np.zeros(3))
        out = notch if out is None else out + notch
    return geom.move(out, geom.rot_z(key.rot), np.array([key.u, key.v, 0.0]))


FIT_PITCH = 19.0    # 試し刷りの穴の間隔
FIT_EDGE = 1.0      # 試し刷りの外形の余白（穴の間隔の外側）


def fit_grid(p: Params) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """試し刷りの穴の大きさ（横に並ぶ 3 通り）と、爪を受ける板の厚み（手前と奥の 2 通り）。"""
    return (p.hole - 0.1, p.hole, p.hole + 0.1), (p.clip_t, p.clip_alt)


def fit_centre(i: int, j: int) -> tuple[float, float]:
    return FIT_EDGE + FIT_PITCH * (i + 0.5), FIT_EDGE + FIT_PITCH * (j + 0.5)


def fit_test(p: Params) -> Manifold:
    """スイッチ穴の試し刷り。

    横に穴の大きさを 3 通り並べ、手前の辺の刻みの数（1〜3）で見分ける。刻みのある手前の列は
    本番のプレートと同じ断面（板の厚み clip_t）、奥の列は板の厚みを clip_alt にした断面。
    プレートと同じく下面を造形面に置いて印刷するので、逃げの天井の垂れ方も本番と同じになる。
    """
    sizes, ledges = fit_grid(p)
    m = geom.box(0.0, FIT_PITCH * len(sizes) + 2 * FIT_EDGE, 0.0, FIT_PITCH * len(ledges) + 2 * FIT_EDGE,
                 0.0, p.plate_t)
    for j, thick in enumerate(ledges):
        for i, size in enumerate(sizes):
            cx, cy = fit_centre(i, j)
            key = Key(f"t{i}{j}", "alpha", cx, cy)
            m -= geom.prism(key.square(size), -1.0, p.plate_t + 1.0)
            m -= clip_relief(p, key, -1.0, p.plate_t - thick, size=size)
    for i in range(len(sizes)):
        cx, _cy = fit_centre(i, 0)
        for n in range(i + 1):
            m -= geom.box(cx - 3.0 + 2.2 * n, cx - 2.0 + 2.2 * n, -1.0, 0.9, -1.0, p.plate_t + 1.0)
    return m
