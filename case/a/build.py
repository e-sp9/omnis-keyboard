"""STL を stl/ に出す。

    .venv/bin/python build.py                      # すべての部品
    .venv/bin/python build.py --tilt 20 --slope 0  # 傾斜台だけを別の姿勢で（left_base_tilt20_slope0.stl など）
    .venv/bin/python build.py --set hole=14.2 --out ~/omnis-hole142   # 寸法を変えて別の場所へ

書き出した STL は読み直し、閉じた形でなければ止まる。作ったときの寸法は、STL と同じ場所の
build.json に記録する。check.py、render.py、estimate.py はこの記録を読んで、同じ寸法で動く。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from omnis_a import assembly, manifest, parts, plate, shell  # noqa: E402
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
        inside = half["plate"] + half["switch_lower"] + half["switch_pins"]
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
    ap.add_argument("--tilt", type=float, help="傾斜台だけを、この左右の傾きで出す")
    ap.add_argument("--slope", type=float, help="傾斜台だけを、この前後の傾きで出す")
    ap.add_argument("--set", action="append", default=[], metavar="名前=値", help="寸法を上書きする")
    ap.add_argument("--out", type=Path, default=HERE / "stl")
    args = ap.parse_args()
    args.out = args.out.expanduser()

    p = DEFAULT.with_(**parse_overrides(args.set))
    only_base = args.tilt is not None or args.slope is not None
    if args.tilt is not None:
        p = p.with_(tilt=args.tilt)
    if args.slope is not None:
        p = p.with_(slope=args.slope)
    try:
        assembly.layout(p, "left"), assembly.layout(p, "right")
    except RuntimeError as err:
        raise SystemExit(f"この寸法では配置できません: {err}")
    for problem in fit_problems(p):
        raise SystemExit(f"この寸法では部品が当たります: {problem}")
    if only_base:
        tag = manifest.posture_tag(args.tilt, args.slope)
        todo = {f"{name}{tag}": m for name, m in base_parts(p).items()}
        # 載せるプレートは同じ場所のものを使う。別の寸法で作ってあったら、合わないので止まる。
        for side in assembly.SIDES:
            entry = manifest.read(args.out).get(f"{side}_plate.stl")
            if entry is not None and manifest.params_from(entry["params"]).with_(tilt=p.tilt, slope=p.slope) != p:
                raise SystemExit(f"{args.out} のプレートは別の寸法（{manifest.describe(manifest.params_from(entry['params']))}）"
                                 "で作られています。同じ --set を付けるか、--out で別の場所へ出してください。")
    else:
        todo = all_parts(p)

    leans = shell.wall_leans(p, assembly.layout(p, "left"))
    print(f"姿勢は左右 {p.tilt:g}°、前後 {p.slope:g}°（合わせて {shell.steepest(p):.1f}°）。"
          f"壁の寝かせは内側 {leans['cheek']:.1f}°、奥 {leans['back']:.1f}°。寸法は{manifest.describe(p)}")
    for name, m in todo.items():
        back = write_stl(m, args.out / f"{name}.stl")
        size = back.bounds[1] - back.bounds[0]
        print(f"  {name}.stl  {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} mm  "
              f"{back.volume / 1000:.1f} cm³  閉じた形")
    manifest.record(args.out, {f"{name}.stl": p for name in todo})
    print(f"寸法の記録: {args.out / manifest.NAME}")
    if only_base:
        print(f"確かめるには: check.py --stl {args.out} {manifest.tag_options(tag)}")


if __name__ == "__main__":
    main()
