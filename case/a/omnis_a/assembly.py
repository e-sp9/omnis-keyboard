"""組み立てた状態。左右とも机の座標で、内側が +x の向きに作る（右手は使う側で反転する）。"""

from __future__ import annotations

from functools import lru_cache

from manifold3d import Manifold

from . import layout as lay_mod, parts, plate as plate_mod, shell
from .layout import Layout
from .params import Params

SIDES = ("left", "right")


@lru_cache(maxsize=16)
def layout(p: Params, side: str) -> Layout:
    return lay_mod.make_layout(p, side)


@lru_cache(maxsize=16)
def plate_deck(p: Params, side: str) -> Manifold:
    return plate_mod.plate(p, layout(p, side))


@lru_cache(maxsize=16)
def base_world(p: Params, side: str) -> Manifold:
    return shell.base(p, layout(p, side))


def half(p: Params, side: str, press: float = 0.0, knurl: bool = True, tall: float = 0.0,
         look: bool = False) -> dict[str, Manifold]:
    """片手ぶんの部品（机の座標、内側が +x）。

    press はキーの押し込み量、tall はスイッチの上のハウジングの高さに足す量（公差の端を見るとき）。
    look を渡すと、キーキャップをプレビュー用の丸みのある形にする（検査には使わない）。
    """
    lay = layout(p, side)

    def world(m: Manifold) -> Manifold:
        return shell.to_world(p, lay, m)

    out: dict[str, Manifold] = {"plate": world(plate_deck(p, side))}
    base = base_world(p, side)
    upper, lower = base.split_by_plane((0.0, 0.0, 1.0), p.floor_top)     # 底の板から下が、黒くする台
    out["base"] = upper
    out["foot"] = lower

    caps = voice = sw_up = sw_low = sw_pins = stems = None
    sw_u, sw_l, sw_p = parts.switch(p, tall)
    stem = parts.stem(p, press)
    for key in lay.keys:
        cap = parts.place_on_key(p, key, parts.keycap_look(p, key) if look else parts.keycap(p, key), press)
        if key.kind == "voice":
            voice = cap
        else:
            caps = cap if caps is None else caps + cap
        a, b = parts.place_on_key(p, key, sw_u), parts.place_on_key(p, key, sw_l)
        c = parts.place_on_key(p, key, stem)
        d = parts.place_on_key(p, key, sw_p)
        sw_up = a if sw_up is None else sw_up + a
        sw_low = b if sw_low is None else sw_low + b
        sw_pins = d if sw_pins is None else sw_pins + d
        stems = c if stems is None else stems + c
    out["caps"] = world(caps)
    if voice is not None:
        out["voice_cap"] = world(voice)
    # ハウジングは軸の通り道を抜いておく（軸とキーキャップの筒はそこを動く）
    bore = None
    for key in lay.keys:
        hole = parts.place_on_key(p, key, parts.stem(p).translate((0, 0, -3.0)))
        bore = hole if bore is None else bore + hole
    out["switch_upper"] = world(sw_up - bore)
    out["switch_lower"] = world(sw_low)      # 下のハウジングと中央の柱
    out["switch_pins"] = world(sw_pins)      # 電気の端子
    out["stems"] = world(stems)
    if lay.wheel is not None:
        out["wheel"] = world(parts.wheel_in_place(p, lay, knurl))
        out["axle"] = world(parts.axle_in_place(p, lay))
    else:
        out["ball"] = world(parts.ball_in_place(p, lay))
    return out
