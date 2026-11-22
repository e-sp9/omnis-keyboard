"""STL を作ったときの寸法の記録（STL と同じ場所の build.json）。

build.py が書き、check.py、render.py、estimate.py が読む。寸法や傾きを変えて作った部品を、
作ったときと同じ寸法で確かめたり描いたりするためのもの。記録するのは既定値と違う寸法だけ。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .params import DEFAULT, Params

NAME = "build.json"
PARTS = ("left_plate", "right_plate", "left_base", "right_base", "left_wheel",
         "keycap_1u", "keycap_space", "keycap_enter", "keycap_voice", "fit_test")
BASES = ("left_base", "right_base")


def overrides(p: Params) -> dict:
    """既定値と違う寸法。"""
    out = {}
    for name in Params.__dataclass_fields__:
        value = getattr(p, name)
        if value != getattr(DEFAULT, name):
            out[name] = list(value) if isinstance(value, tuple) else value
    return out


def params_from(values: dict) -> Params:
    fields = Params.__dataclass_fields__
    kw = {}
    for name, value in values.items():
        if name not in fields:
            raise SystemExit(f"{NAME}: {name} という寸法はありません（古い記録なら build.py で出し直す）")
        current = getattr(DEFAULT, name)
        kw[name] = tuple(float(v) for v in value) if isinstance(current, tuple) else type(current)(value)
    return DEFAULT.with_(**kw)


def describe(p: Params) -> str:
    """既定値との違いを 1 行で。"""
    diff = overrides(p)
    if not diff:
        return "既定値のまま"
    return "、".join(f"{k}={','.join(f'{x:g}' for x in v) if isinstance(v, list) else v}" for k, v in diff.items())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def posture_tag(tilt: float | None = None, slope: float | None = None) -> str:
    """傾斜台だけを別の姿勢で出したときの、ファイル名の末尾（_tilt20_slope0 など）。"""
    tag = ""
    if tilt is not None:
        tag += f"_tilt{tilt:g}"
    if slope is not None:
        tag += f"_slope{slope:g}"
    return tag.replace(".", "p")


def read(folder: Path) -> dict:
    path = folder / NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("files", {})


def record(folder: Path, written: dict[str, Params]) -> None:
    """書き出した STL の寸法と中身のハッシュを記録する。同じ場所の、ほかの STL の記録は残す。"""
    files = read(folder)
    for name, p in written.items():
        files[name] = {"params": overrides(p), "sha256": digest(folder / name)}
    files = {name: entry for name, entry in sorted(files.items()) if (folder / name).exists()}
    (folder / NAME).write_text(json.dumps({"files": files}, ensure_ascii=False, indent=1) + "\n")


def params_of(folder: Path, name: str) -> Params:
    """その STL を作ったときの寸法。記録がなければ、出し直し方を伝えて止まる。"""
    if not (folder / name).exists():
        raise SystemExit(f"{folder / name} がありません。build.py で出してください。")
    entry = read(folder).get(name)
    if entry is None:
        raise SystemExit(f"{folder / NAME} に {name} の記録がありません。build.py で出し直してください。")
    return params_from(entry["params"])


def intact(folder: Path, name: str) -> bool:
    """STL が、build.py が書いたときのままか。"""
    entry = read(folder).get(name)
    return entry is not None and entry.get("sha256") == digest(folder / name)


def variants(folder: Path) -> list[str]:
    """その場所にある、姿勢だけ変えた傾斜台の末尾（_tilt20_slope0 など）。"""
    out = set()
    for name in read(folder):
        stem = name.removesuffix(".stl")
        if "_base_" in stem:
            out.add("_" + stem.split("_base_", 1)[1])
    return sorted(out)


def tag_options(tag: str) -> str:
    """末尾から、同じ傾斜台を選ぶためのオプション（--tilt 20 --slope 0 など）を作る。"""
    words = []
    for part in tag.strip("_").split("_"):
        for key in ("tilt", "slope"):
            if part.startswith(key):
                words.append(f"--{key} {part[len(key):].replace('p', '.')}")
    return " ".join(words)


def load_set(folder: Path, tilt: float | None = None, slope: float | None = None) -> tuple[Params, Params, dict]:
    """その場所の一式を作ったときの寸法。

    返すのは、プレートの一式の寸法、傾斜台の寸法（姿勢だけが違うことがある）、
    部品の名前 -> ファイルの名前。tilt や slope を渡すと、傾斜台はその姿勢で出したもの
    （left_base_tilt20_slope0.stl など）を使う。部品どうしの寸法が食い違っていたら、
    組めないので止まる。
    """
    tag = posture_tag(tilt, slope)
    files = {name: f"{name}{tag if name in BASES else ''}.stl" for name in PARTS}
    found = {name: params_of(folder, files[name]) for name in PARTS}
    p_plate = found["left_plate"]
    for name in PARTS:
        if name not in BASES and found[name] != p_plate:
            raise SystemExit(f"{files[name]} は、left_plate.stl と別の寸法（{describe(found[name])}）で"
                             "作られています。build.py で一式を出し直してください。")
    p = found["left_base"]
    if found["right_base"] != p:
        raise SystemExit("左右の傾斜台が別の寸法で作られています。build.py で出し直してください。")
    if (tilt is not None and abs(p.tilt - tilt) > 1e-9) or (slope is not None and abs(p.slope - slope) > 1e-9):
        raise SystemExit(f"{files['left_base']} の記録の姿勢が、左右 {p.tilt:g}°、前後 {p.slope:g}° です。"
                         "build.py で出し直してください。")
    if p_plate.with_(tilt=p.tilt, slope=p.slope) != p:
        raise SystemExit(f"プレート（{describe(p_plate)}）と傾斜台（{describe(p)}）が"
                         "別の寸法で作られています。同じ寸法で出し直してください。")
    return p_plate, p, files
