"""キー面の上の配置。キー面の座標 (u, v) で、左右とも内側を +u として決める。

文字キーはピッチと列のずれから置く。親指キー、球、ホイールの窓、手前の縁は、版A-1で
求めた位置を寸法（params.py）に書いてあり、ここではそのまま使う。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from manifold3d import CrossSection

from . import geom
from .params import Params

CRADLE = 1.6   # 軸受けの溝が窓の端から伸びる長さ


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
    ball: tuple[float, float]                  # 球の中心（左ではホイールのまわりのくぼみの中心）
    ball_hole_r: float
    wheel: tuple[float, float, float, float] | None   # 窓 (u0, u1, v0, v1)。左だけ
    lip: tuple[float, float, float, float] | None     # 仕切り (u0, u1, v0, v1)。付けるときだけ

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


def _alpha_keys(p: Params) -> list[Key]:
    keys = []
    for col, shift in enumerate(p.stagger):
        for row in range(p.rows):
            keys.append(Key(f"c{col}r{row}", "alpha", p.pitch * col, shift + p.pitch * row,
                            cap_w=p.cap, cap_d=p.cap))
    return keys


def _voice_key(p: Params, alphas: list[Key]) -> Key:
    pinky_front = alphas[0]
    v = pinky_front.v - p.cap / 2.0 - 1.5 - p.voice_cap / 2.0
    return Key("voice", "voice", p.voice_u, round(v, 2), cap_w=p.voice_cap, cap_d=p.voice_cap,
               sink=p.voice_sink)


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


def make_layout(p: Params, side: str) -> Layout:
    if side not in ("left", "right"):
        raise ValueError(side)
    alphas = _alpha_keys(p)
    u_out = -p.cap / 2.0 - p.margin_outer
    u_in = p.pitch * (len(p.stagger) - 1) + p.cap / 2.0 + p.margin_inner
    v_back = max(k.v for k in alphas) + p.cap / 2.0 + p.well_gap + p.rim
    r_hole = p.ball_d / 2.0 + p.ball_gap
    keys: list[Key] = list(alphas)
    wheel = lip = None

    if side == "left":
        # ホイールの窓。奥行きの中心は、右の球の中心と同じ。
        win_len = p.wheel_len + 2 * (p.wheel_pad + p.wheel_play)
        win_dep = p.wheel_d + 1.0
        u1 = p.wheel_u1
        u0 = u1 - win_len
        wheel = (u0, u1, p.ball_at[1] - win_dep / 2.0, p.ball_at[1] + win_dep / 2.0)
        if p.lip_h > 0:
            lu1 = u0 - 0.3
            lip = (lu1 - p.lip_w, lu1, wheel[2] - 0.6, wheel[3] + 0.6)
        w = p.space_w * p.pitch - (p.pitch - p.cap)
        keys += [Key("space", "thumb", p.space_at[0], p.space_at[1], p.thumb_rot, cap_w=w, cap_d=p.cap),
                 _voice_key(p, alphas)]
    else:
        w1 = p.enter_w * p.pitch - (p.pitch - p.cap)
        keys += [Key("enter", "thumb", p.enter_at[0], p.enter_at[1], p.thumb_rot, cap_w=w1, cap_d=p.cap),
                 Key("thumb2", "thumb", p.thumb2_at[0], p.thumb2_at[1], p.thumb_rot, cap_w=p.cap, cap_d=p.cap)]

    outline = (
        (u_out, p.v_front),
        (u_in, p.v_front),
        (u_in, v_back),
        (u_out + p.back_cut_u, v_back),
        (u_out, v_back - p.back_cut_v),
    )
    return Layout(side, tuple(keys), u_out, u_in, p.v_front, v_back, outline, p.ball_at, r_hole,
                  wheel, lip)


# ---- くぼみ --------------------------------------------------------------------

def well_rects(p: Params, lay: Layout) -> list[tuple[float, float, float, float, float]]:
    """キーキャップが入るくぼみを作る長方形 (u, v, 幅, 奥行き, 回転)。音声入力キーは別。"""
    return [(k.u, k.v, k.cap_w + 2 * p.well_gap, k.cap_d + 2 * p.well_gap, k.rot)
            for k in lay.keys if k.kind != "voice"]


def wells(p: Params, lay: Layout) -> CrossSection:
    """キーキャップが入るくぼみ（上面から、スイッチが載る面まで）。"""
    cs = geom.union_all([geom.rect(*r) for r in well_rects(p, lay)])
    return geom.close_gaps(cs, 0.6)


def voice_well(p: Params, lay: Layout) -> CrossSection | None:
    for k in lay.keys:
        if k.kind == "voice":
            return k.square(p.sw_flange + 2 * 0.3)
    return None
