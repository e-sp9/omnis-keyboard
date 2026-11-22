"""STL を作ったときの記録（STL と同じ場所の build.json）。

build.py が書き、check.py、render.py、estimate.py が読む。STL ごとに、中身のハッシュと、
作ったときの寸法のすべて、形を作るコードの版（params.MODEL）を記録する。既定値をあとで
変えても、その STL を作ったときの寸法と姿勢のまま、確かめたり描いたりできる。

別の版のコードで作った STL は、今のコードでは同じ形を作り直せない。版のない古い記録
（既定値と違う寸法しか残っていない）は、作ったときの寸法が分からない。どちらも、読む前に
そう知らせて止まる。

    {"format": 2,
     "sets":  {"<寸法の組の名前>": {"model": "A-2.1", "params": {すべての寸法}}},
     "files": {"left_plate.stl": {"set": "<寸法の組の名前>", "sha256": "..."}}}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .params import DEFAULT, MODEL, Params

NAME = "build.json"
FORMAT = 2
PARTS = ("left_plate", "right_plate", "left_base", "right_base", "left_wheel",
         "keycap_1u", "keycap_space", "keycap_enter", "keycap_voice", "fit_test")
BASES = ("left_base", "right_base")


# ---- 寸法と記録の行き来 -----------------------------------------------------------

def values_of(p: Params) -> dict:
    """すべての寸法（JSON に書ける形）。"""
    out = {}
    for name in Params.__dataclass_fields__:
        value = getattr(p, name)
        out[name] = list(value) if isinstance(value, tuple) else value
    return out


def overrides(p: Params) -> dict:
    """今の既定値と違う寸法。"""
    base = values_of(DEFAULT)
    return {name: value for name, value in values_of(p).items() if value != base[name]}


def params_from(values: dict, where: str = NAME) -> Params:
    """記録された寸法から Params を作る。項目が今の params.py と合わなければ止まる。"""
    fields = Params.__dataclass_fields__
    missing = [name for name in fields if name not in values]
    extra = [name for name in values if name not in fields]
    if missing or extra:
        raise SystemExit(f"{where}: 記録された寸法の項目が、今の params.py と合いません"
                         f"（足りない: {'、'.join(missing) or 'なし'}／知らない: {'、'.join(extra) or 'なし'}）。"
                         "build.py で出し直してください。")
    kw = {}
    for name in fields:
        current = getattr(DEFAULT, name)
        value = values[name]
        kw[name] = tuple(float(v) for v in value) if isinstance(current, tuple) else type(current)(value)
    return Params(**kw)


def describe(p: Params) -> str:
    """今の既定値との違いを 1 行で。"""
    diff = overrides(p)
    if not diff:
        return "既定値のまま"
    return "、".join(f"{k}={','.join(f'{x:g}' for x in v) if isinstance(v, list) else v}" for k, v in diff.items())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_name(model: str, values: dict) -> str:
    text = json.dumps({"model": model, "params": values}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()[:10]


# ---- 読み書き -------------------------------------------------------------------

def _load(folder: Path) -> dict:
    path = folder / NAME
    if not path.exists():
        return {"format": FORMAT, "sets": {}, "files": {}}
    doc = json.loads(path.read_text())
    return {"format": doc.get("format", 1), "sets": doc.get("sets", {}), "files": doc.get("files", {})}


def registered(folder: Path) -> list[str]:
    """記録にある STL の名前。"""
    return sorted(_load(folder)["files"])


def record(folder: Path, written: dict[str, Params]) -> None:
    """書き出した STL の寸法、コードの版、中身のハッシュを記録する。

    同じ場所の、ほかの STL の記録は残す（古い形式の記録は、古い形式のまま残る）。
    なくなった STL の記録と、どの STL も使っていない寸法の組は落とす。
    """
    doc = _load(folder)
    for name, p in written.items():
        values = values_of(p)
        key = _set_name(MODEL, values)
        doc["sets"][key] = {"model": MODEL, "params": values}
        doc["files"][name] = {"set": key, "sha256": digest(folder / name)}
    files = {name: entry for name, entry in sorted(doc["files"].items()) if (folder / name).exists()}
    used = {entry.get("set") for entry in files.values()}
    sets = {key: value for key, value in sorted(doc["sets"].items()) if key in used}
    out = {"format": FORMAT, "sets": sets, "files": files}
    (folder / NAME).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


def _entry(folder: Path, name: str) -> tuple[dict, dict]:
    """その STL の記録と、寸法の組。扱えない記録なら、理由を伝えて止まる。"""
    if not (folder / name).exists():
        raise SystemExit(f"{folder / name} がありません。build.py で出してください。")
    doc = _load(folder)
    entry = doc["files"].get(name)
    if entry is None:
        raise SystemExit(f"{folder / NAME} に {name} の記録がありません。build.py で出し直してください。")
    if "set" not in entry or entry["set"] not in doc["sets"]:
        raise SystemExit(f"{folder / NAME} の {name} の記録は古い形式です（モデルの版がなく、当時の既定値と違う寸法しか"
                         "残っていません）。作ったときの寸法と姿勢が分からないので、今のスクリプトでは扱えません。"
                         "その STL を作ったコミットのスクリプトを使うか、build.py で出し直してください。")
    return entry, doc["sets"][entry["set"]]


def model_of(folder: Path, name: str) -> str:
    """その STL を作ったコードの版。"""
    return _entry(folder, name)[1].get("model", "?")


def params_of(folder: Path, name: str) -> Params:
    """その STL を作ったときの寸法。別の版のコードで作ったものなら、そう伝えて止まる。"""
    _file, rec = _entry(folder, name)
    if rec.get("model") != MODEL:
        raise SystemExit(f"{folder / name} は、モデルの版 {rec.get('model', '?')} で作られています。今のスクリプトの版は "
                         f"{MODEL} で、同じ寸法からでも同じ形を作り直せません。その版のコミットのスクリプトを使うか、"
                         "build.py で出し直してください。")
    return params_from(rec["params"], f"{folder / NAME} の {name}")


def intact(folder: Path, name: str) -> bool:
    """STL が、build.py が書いたときのままか。"""
    entry = _load(folder)["files"].get(name)
    return entry is not None and (folder / name).exists() and entry.get("sha256") == digest(folder / name)


def fingerprint(folder: Path, files: dict[str, str]) -> str:
    """一式の指紋。部品の STL の、記録されたハッシュをまとめた短い文字列。

    check.py と estimate.py の出力でこれが同じなら、同じ一式を見ている。
    """
    entries = _load(folder)["files"]
    text = "\n".join(f"{name}:{entries.get(name, {}).get('sha256', '-')}" for name in sorted(files.values()))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ---- 姿勢だけ変えた傾斜台 ----------------------------------------------------------

def posture_tag(tilt: float | None = None, slope: float | None = None) -> str:
    """傾斜台だけを別の姿勢で出したときの、ファイル名の末尾（_tilt20_slope0 など）。"""
    tag = ""
    if tilt is not None:
        tag += f"_tilt{tilt:g}"
    if slope is not None:
        tag += f"_slope{slope:g}"
    return tag.replace(".", "p")


def variants(folder: Path) -> list[str]:
    """その場所にある、姿勢だけ変えた傾斜台の末尾（_tilt20_slope0 など）。"""
    out = set()
    for name in registered(folder):
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


# ---- 一式 -----------------------------------------------------------------------

def load_set(folder: Path, tilt: float | None = None, slope: float | None = None) -> tuple[Params, Params, dict]:
    """その場所の一式を作ったときの寸法。

    返すのは、プレートの一式の寸法、傾斜台の寸法（姿勢だけが違うことがある）、
    部品の名前 -> ファイルの名前。tilt や slope を渡すと、傾斜台はその姿勢で出したもの
    （left_base_tilt20_slope0.stl など）を使う。古い記録や別の版の部品が混じっていたら、
    そう伝えて止まる。部品どうしの寸法が食い違っていても、組めないので止まる。
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


def changed(folder: Path, files: dict[str, str]) -> list[str]:
    """一式のうち、build.py が書いたあとで中身が変わった STL。"""
    return [name for name in files.values() if not intact(folder, name)]
