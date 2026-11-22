"""STL を stl/ に出す。

    .venv/bin/python build.py                 # すべての部品
    .venv/bin/python build.py --tilt 48       # 傾斜台だけを別の角度で（left_base_tilt48.stl など）
    .venv/bin/python build.py --set ball_d=27 --out out/ball27   # 寸法を変えて別の場所へ

書き出した STL は読み直し、閉じた形でなければ止まる。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from omnis_a import assembly, parts, plate  # noqa: E402
from omnis_a.layout import Key  # noqa: E402
from omnis_a.params import DEFAULT, Params  # noqa: E402


def to_trimesh(manifold) -> trimesh.Trimesh:
    mesh = manifold.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                           faces=np.asarray(mesh.tri_verts), process=False)


def write_stl(manifold, path: Path) -> trimesh.Trimesh:
    """STL に書き、読み直して閉じた形か確かめる。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    to_trimesh(manifold).export(path)
    back = trimesh.load(path)
    if not (back.is_watertight and back.is_winding_consistent and back.volume > 0):
        raise SystemExit(f"{path.name}: 閉じた形になっていません")
    return back


def base_parts(p: Params) -> dict:
    """傾斜台。机に置く向きのまま。右手は左右を反転する。"""
    left = assembly.base_world(p, "left")
    right = assembly.base_world(p, "right").mirror((1.0, 0.0, 0.0))
    return {"left_base": left, "right_base": right}


def plate_parts(p: Params) -> dict:
    return {f"{side}_plate": plate.printable(p, assembly.layout(p, side)) for side in assembly.SIDES}


def keycap_parts(p: Params) -> dict:
    """印刷するキーキャップ（任意）。上面を下にした向き。"""
    left, right = assembly.layout(p, "left"), assembly.layout(p, "right")
    one = Key("1u", "alpha", 0.0, 0.0, cap_w=p.cap, cap_d=p.cap)
    return {
        "keycap_1u": parts.keycap_printable(p, one),
        "keycap_space": parts.keycap_printable(p, left.key("space")),
        "keycap_enter": parts.keycap_printable(p, right.key("enter")),
        "keycap_voice": parts.keycap_printable(p, left.key("voice")),
    }


def all_parts(p: Params) -> dict:
    out = {}
    out.update(plate_parts(p))
    out.update(base_parts(p))
    out["left_wheel"] = parts.wheel(p).translate((0.0, 0.0, p.wheel_len / 2.0))
    out.update(keycap_parts(p))
    out["fit_test"] = parts.fit_test(p)
    return out


def fit_problems(p: Params) -> list[str]:
    """組んだときに傾斜台がプレートやスイッチに食い込まないかを見る。"""
    out = []
    for side in assembly.SIDES:
        half = assembly.half(p, side, knurl=False)
        base = half["base"] + half["foot"]
        inside = half["plate"] + half["switch_lower"]
        for extra in ("ball", "wheel"):
            if extra in half:
                inside += half[extra]
        overlap = (base ^ inside).volume()
        if overlap > 0.05:
            out.append(f"{side} の傾斜台が、プレートかスイッチ、球、ホイールと {overlap:.1f} mm³ 重なる"
                       "（内側の壁を立てたときは margin_inner を増やす）")
    return out


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    fields = Params.__dataclass_fields__
    for pair in pairs:
        name, _, raw = pair.partition("=")
        if name not in fields:
            raise SystemExit(f"--set: {name} という寸法はありません")
        current = getattr(DEFAULT, name)
        if isinstance(current, tuple):
            out[name] = tuple(float(v) for v in raw.split(","))
        else:
            out[name] = type(current)(raw)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tilt", type=float, help="傾斜台だけを、この傾きで出す")
    ap.add_argument("--set", action="append", default=[], metavar="名前=値", help="寸法を上書きする")
    ap.add_argument("--out", type=Path, default=HERE / "stl")
    args = ap.parse_args()

    p = DEFAULT.with_(**parse_overrides(args.set))
    if args.tilt is not None:
        p = p.with_(tilt=args.tilt)
    try:
        assembly.layout(p, "left"), assembly.layout(p, "right")
    except RuntimeError as err:
        raise SystemExit(f"この寸法では配置できません: {err}")
    for problem in fit_problems(p):
        raise SystemExit(f"この寸法では部品が当たります: {problem}")
    if args.tilt is not None:
        tag = f"_tilt{args.tilt:g}".replace(".", "p")
        todo = {f"{name}{tag}": m for name, m in base_parts(p).items()}
    else:
        todo = all_parts(p)

    print(f"傾き {p.tilt:g}°、内側の壁の寝かせ {p.lean:g}°")
    for name, m in todo.items():
        back = write_stl(m, args.out / f"{name}.stl")
        size = back.bounds[1] - back.bounds[0]
        print(f"  {name}.stl  {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} mm  "
              f"{back.volume / 1000:.1f} cm³  閉じた形")


if __name__ == "__main__":
    main()
