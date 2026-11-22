"""プレビュー画像を preview/ に出す。

    .venv/bin/python render.py              # すべての図
    .venv/bin/python render.py hero front   # 選んだ図だけ

    hero     参考画像と同じ向き（手前のやや上から、左右を並べる）。compare.jpg も出す
    tilt20   同じプレートを 20° の傾斜台に載せた形。参考画像と同じカメラ。compare_tilt20.jpg も出す
    seat     座った目の高さから
    front    真正面
    layout   プレートを平らに置いて真上から
    parts    印刷する部品を、印刷する向きのまま並べた図

    .venv/bin/python render.py --tilt 48            # 傾斜台を left_base_tilt48.stl などに替えた形
    .venv/bin/python render.py --stl ~/omnis-ball27  # 別の場所に出した一式

形は STL を読むのではなく、build.py と同じ関数から直接作る。寸法は既定値ではなく、STL と同じ
場所の build.json（build.py が書く、作ったときの寸法の記録）から読む。stl/ の一式は preview/ に、
それ以外は使い捨ての置き場（環境変数 OMNIS_SCRATCH、なければ ~/.cache/omnis-case-a）に出す。
描画用の一時ファイルも使い捨ての置き場に書く。
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build  # noqa: E402
from omnis_a import assembly, geom, manifest, parts, workdir  # noqa: E402
from omnis_a.params import Params  # noqa: E402

PREVIEW = HERE / "preview"
CACHE = workdir.scratch("render")
REFERENCE = HERE / "reference" / "chamfer-inner-wheel.jpg"

# REFERENCE_VIEW は、同じプレートを 20° の傾斜台に載せた形を参考画像に重ねて合わせたカメラ。
# HERO は同じ向きのまま、43° の形が画面に収まるところまで引いたもの。
REFERENCE_VIEW = dict(gap=34.0, elevation=31.0, distance=820.0, fov=16.7, target=(0.0, -2.0, 22.0),
                      size=(1500, 750), spp=512)
HERO = dict(REFERENCE_VIEW, fov=19.3, target=(0.0, -2.0, 31.0), size=(1500, 840))
SEAT = dict(REFERENCE_VIEW, elevation=52.0, fov=19.3, target=(0.0, 2.0, 30.0), size=(1500, 900))
ILLUSTRATION_TILT = 20.0

MATERIALS = {
    "case": {"type": "principled", "base_color": {"type": "rgb", "value": [0.74, 0.73, 0.72]},
             "metallic": 1.0, "roughness": 0.46},
    "foot": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.010, 0.010, 0.010]}},
    "caps": {"type": "principled", "base_color": {"type": "rgb", "value": [0.022, 0.022, 0.023]},
             "roughness": 0.55, "specular": 0.5},
    "voice": {"type": "principled", "base_color": {"type": "rgb", "value": [0.78, 0.40, 0.25]},
              "metallic": 0.85, "roughness": 0.45},
    "ball": {"type": "principled", "base_color": {"type": "rgb", "value": [0.018, 0.018, 0.019]},
             "roughness": 0.40, "specular": 0.5},
    "wheel": {"type": "principled", "base_color": {"type": "rgb", "value": [0.05, 0.05, 0.055]},
              "metallic": 0.9, "roughness": 0.38},
    "switch": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.012, 0.012, 0.012]}},
    "desk": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.345, 0.285, 0.255]}},
    "bed": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.20, 0.20, 0.21]}},
}

GROUP_MATERIAL = {
    "plate": "case", "base": "case", "foot": "foot", "caps": "caps", "voice_cap": "voice",
    "ball": "ball", "wheel": "wheel", "axle": "wheel", "switch_upper": "switch",
    "switch_lower": "switch", "switch_pins": "switch", "stems": "switch",
}


# ---- Mitsuba -------------------------------------------------------------------

def _mi():
    import mitsuba as mi

    if mi.variant() is None:
        for variant in ("metal_ad_rgb", "llvm_ad_rgb", "scalar_rgb"):
            try:
                mi.set_variant(variant)
                break
            except Exception:      # その環境で使えない描画方式は飛ばす
                continue
        else:
            raise RuntimeError("Mitsuba の描画方式を選べませんでした。")
    return mi


def _shape(name: str, material: str, manifold) -> dict:
    """形を Mitsuba に渡す。球は面で分けず、なめらかな球として渡す。"""
    if material == "ball":
        bb = manifold.bounding_box()
        centre = [(bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, (bb[2] + bb[5]) / 2]
        shape = {"type": "sphere", "center": centre, "radius": (bb[3] - bb[0]) / 2}
    else:
        mesh = manifold.to_mesh()
        tm = trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                             faces=np.asarray(mesh.tri_verts), process=False)
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / f"{name}.ply"
        tm.export(path)
        shape = {"type": "ply", "filename": str(path), "face_normals": True}
    shape["bsdf"] = {"type": "ref", "id": f"mat_{material}"}
    return shape


def _lights(mi, scale: float) -> dict:
    def panel(origin, target, w, h, radiance):
        to_world = mi.ScalarTransform4f().look_at(origin=origin, target=target, up=[0, 0, 1]) @ \
            mi.ScalarTransform4f().scale([w / 2.0, h / 2.0, 1.0])
        return {"type": "rectangle", "to_world": to_world,
                "emitter": {"type": "area", "radiance": {"type": "rgb", "value": radiance}}}

    k = scale
    return {
        "key": panel([-420, -520, 980], [0, 0, 20], 1000, 800, [7.0 * k, 6.9 * k, 6.7 * k]),
        "fill": panel([800, -350, 520], [0, 0, 20], 700, 500, [1.5 * k, 1.5 * k, 1.55 * k]),
        "rim": panel([150, 950, 650], [0, 0, 20], 900, 400, [2.2 * k, 2.2 * k, 2.2 * k]),
        "sky": {"type": "constant", "radiance": {"type": "rgb", "value": [0.16 * k, 0.15 * k, 0.145 * k]}},
    }


def _tonemap(linear: np.ndarray, exposure: float) -> Image.Image:
    x = np.clip(linear[..., :3] * exposure, 0.0, None)
    x = x / (1.0 + 0.18 * x)                        # 明るい所をゆるく抑える
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return Image.fromarray((np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8))


def shoot(items, origin, target, size, spp: int, fov: float = 20.0, ortho_width: float | None = None,
          desk: str = "desk", light: float = 1.0, exposure: float = 1.0, up=(0, 0, 1)) -> Image.Image:
    """(名前, 材質, 形) の一覧を、決めたカメラで描く。ortho_width を渡すと平行投影。"""
    mi = _mi()
    to_world = mi.ScalarTransform4f().look_at(origin=list(origin), target=list(target), up=list(up))
    film = {"type": "hdrfilm", "width": size[0], "height": size[1], "rfilter": {"type": "gaussian"}}
    if ortho_width is None:
        sensor = {"type": "perspective", "fov": fov, "fov_axis": "x", "to_world": to_world,
                  "film": film, "sampler": {"type": "independent"}}
    else:
        half = ortho_width / 2.0
        sensor = {"type": "orthographic",
                  "to_world": to_world @ mi.ScalarTransform4f().scale([half, half, 1.0]),
                  "film": film, "sampler": {"type": "independent"}}
    scene = {"type": "scene", "integrator": {"type": "path", "max_depth": 7}, "sensor": sensor}
    for name, bsdf in MATERIALS.items():
        scene[f"mat_{name}"] = dict(bsdf)
    scene["floor"] = {"type": "rectangle", "to_world": mi.ScalarTransform4f().scale([4000.0, 4000.0, 1.0]),
                      "bsdf": {"type": "ref", "id": f"mat_{desk}"}}
    scene.update(_lights(mi, light))
    for name, material, manifold in items:
        scene[name] = _shape(name, material, manifold)
    loaded = mi.load_dict(scene)
    acc, done, seed = None, 0, 0
    while done < spp:          # 一度に描く回数を抑え、メモリを使い切らないようにする
        n = min(128, spp - done)
        img = np.array(mi.render(loaded, spp=n, seed=seed))
        acc = img * n if acc is None else acc + img * n
        done += n
        seed += 1
    return _tonemap(acc / spp, exposure)


# ---- 並べ方 ---------------------------------------------------------------------

def pair(p: Params, gap: float) -> list[tuple[str, str, object]]:
    """左右を机の上に並べた状態。足元の間隔が gap。"""
    out = []
    for side in assembly.SIDES:
        half = assembly.half(p, side)
        shift = gap / 2.0 + assembly.footprint_x(p, side)[1]
        for group, m in half.items():
            if side == "left":
                m = m.translate((-shift, 0.0, 0.0))
            else:
                m = m.mirror((1.0, 0.0, 0.0)).translate((shift, 0.0, 0.0))
            out.append((f"{side}_{group}", GROUP_MATERIAL[group], m))
    return out


def _orbit(cfg: dict) -> tuple:
    e = math.radians(cfg["elevation"])
    tx, ty, tz = cfg["target"]
    return (tx, ty - cfg["distance"] * math.cos(e), tz + cfg["distance"] * math.sin(e))


def view(p: Params, cfg: dict) -> Image.Image:
    return shoot(pair(p, cfg["gap"]), _orbit(cfg), cfg["target"], cfg["size"], cfg["spp"], fov=cfg["fov"])


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def stack(top: Image.Image, bottom: Image.Image) -> Image.Image:
    """2枚を上下に並べる。幅は上の画像に合わせる。"""
    w = top.width
    low = bottom.resize((w, round(bottom.height * w / bottom.width)), Image.LANCZOS)
    out = Image.new("RGB", (w, top.height + low.height + 6), (255, 255, 255))
    out.paste(top, (0, 0))
    out.paste(low, (0, top.height + 6))
    return out


# ---- それぞれの図 -----------------------------------------------------------------

def hero(p: Params, spp: int | None = None) -> Image.Image:
    return view(p, dict(HERO, spp=spp or HERO["spp"]))


def tilt20(p: Params, spp: int | None = None) -> Image.Image:
    cfg = dict(REFERENCE_VIEW, spp=spp or REFERENCE_VIEW["spp"])
    return view(p.with_(tilt=ILLUSTRATION_TILT), cfg)


def seat(p: Params, spp: int | None = None) -> Image.Image:
    return view(p, dict(SEAT, spp=spp or SEAT["spp"]))


def front(p: Params, spp: int | None = None) -> Image.Image:
    """真正面（平行投影）。傾きと外寸を書き込む。"""
    gap = 34.0
    size, width = (1500, 560), 300.0
    centre_z = 38.0
    wall = geom.box(-2000.0, 2000.0, 400.0, 410.0, 0.0, 2000.0)     # 背景の壁
    img = shoot(pair(p, gap) + [("backdrop", "desk", wall)], (0.0, -900.0, centre_z),
                (0.0, 0.0, centre_z), size, spp or 256, ortho_width=width)
    scale = size[0] / width

    def px(x: float, z: float) -> tuple[float, float]:
        return (size[0] / 2 + x * scale, size[1] / 2 - (z - centre_z) * scale)

    draw = ImageDraw.Draw(img)
    ink, font = (40, 40, 40), _font(26)
    bb = assembly.base_world(p, "left").bounding_box()
    w_base = bb[3] - bb[0]
    half = assembly.half(p, "left", knurl=False)
    top_z = (half["plate"] + half["caps"]).bounding_box()[5]
    x0 = -(gap / 2 + w_base)
    # 幅
    y = px(0, -7)[1]
    draw.line([px(x0, 0)[0], y, px(-gap / 2, 0)[0], y], fill=ink, width=2)
    for x in (x0, -gap / 2):
        draw.line([px(x, 0)[0], y - 8, px(x, 0)[0], y + 8], fill=ink, width=2)
    draw.text((px(x0 + w_base / 2, 0)[0], y + 6), f"{w_base:.0f} mm", fill=ink, font=font, anchor="ma")
    # 高さ
    x = px(x0 - 9, 0)[0]
    draw.line([x, px(0, 0)[1], x, px(0, top_z)[1]], fill=ink, width=2)
    for z in (0.0, top_z):
        draw.line([x - 8, px(0, z)[1], x + 8, px(0, z)[1]], fill=ink, width=2)
    draw.text((x - 10, px(0, top_z / 2)[1]), f"{top_z:.0f} mm", fill=ink, font=font, anchor="rm")
    # 傾き。外側の端から水平線と、キー面に沿った線を引く
    ox, oz = px(x0, 0)[0] + 2, px(0, 3.0)[1]
    r = 150
    draw.line([ox, oz, ox + r, oz], fill=ink, width=2)
    t = math.radians(p.tilt)
    draw.line([ox, oz, ox + r * math.cos(t), oz - r * math.sin(t)], fill=ink, width=2)
    draw.arc([ox - 90, oz - 90, ox + 90, oz + 90], -p.tilt, 0, fill=ink, width=2)
    draw.text((ox + 100, oz - 44), f"{p.tilt:.0f}°", fill=ink, font=font, anchor="lm")
    return img


def _flat_plate(p: Params, side: str) -> list[tuple[str, str, object]]:
    """プレートを平らに置き、スイッチとキーキャップを付けた状態（下面が z = 0）。"""
    lay = assembly.layout(p, side)
    lift = (0.0, 0.0, p.plate_t)
    items = [("plate", "case", assembly.plate_deck(p, side).translate(lift))]
    caps = housings = None
    upper = parts.switch(p)[0]
    for key in lay.keys:
        cap = parts.place_on_key(p, key, parts.keycap(p, key)).translate(lift)
        if key.kind == "voice":
            items.append(("voice", "voice", cap))
        else:
            caps = cap if caps is None else caps + cap
        h = parts.place_on_key(p, key, upper + parts.stem(p)).translate(lift)
        housings = h if housings is None else housings + h
    items += [("caps", "caps", caps), ("housings", "switch", housings)]
    if lay.wheel is not None:
        items.append(("wheel", "wheel", parts.wheel_in_place(p, lay).translate(lift)))
        items.append(("axle", "wheel", parts.axle_in_place(p, lay).translate(lift)))
    else:
        items.append(("ball", "ball", parts.ball_in_place(p, lay).translate(lift)))
    return items


def layout(p: Params, spp: int | None = None) -> Image.Image:
    """プレートを平らに置いて真上から見た配置。下が手前。"""
    gap = 26.0
    items = []
    for side in assembly.SIDES:
        lay = assembly.layout(p, side)
        shift = gap / 2.0 + lay.u_in
        for name, material, m in _flat_plate(p, side):
            if side == "left":
                m = m.translate((-shift, 0.0, 0.0))
            else:
                m = m.mirror((1.0, 0.0, 0.0)).translate((shift, 0.0, 0.0))
            items.append((f"{side}_{name}", material, m))
    return shoot(items, (0.0, 2.3, 900.0), (0.0, 2.3, 0.0), (1500, 720), spp or 256,
                 ortho_width=236.0, up=(0, 1, 0))


def parts_sheet(p: Params, spp: int | None = None) -> Image.Image:
    """印刷する部品を、印刷する向きのまま並べる。番号は README の表と同じ。"""
    todo = build.all_parts(p)
    # (番号, 名前, 置く位置 x, y)
    spots = [(1, "left_plate", -176, -78), (2, "right_plate", -64, -78),
             (3, "left_base", -181, 34), (4, "right_base", -64, 34),
             (5, "left_wheel", 62, -66), (6, "fit_test", 62, 30),
             (7, "keycap_1u", 90, -70), (8, "keycap_space", 124, -70),
             (9, "keycap_enter", 62, -22), (10, "keycap_voice", 102, -22)]
    items, labels = [], []
    for number, name, x, y in spots:
        m = todo[name]
        bb = m.bounding_box()
        m = m.translate((x - bb[0], y - bb[1], -bb[2]))
        material = "voice" if name == "keycap_voice" else "wheel" if name == "left_wheel" else \
            "caps" if name.startswith("keycap") else "case"
        items.append((name, material, m))
        labels.append((number, x + (bb[3] - bb[0]) / 2, y - 9.0))
    target = (-10.0, 12.0, 18.0)
    cfg = dict(elevation=56.0, distance=1000.0, target=target)
    origin = _orbit(cfg)
    size, fov = (1500, 1000), 25.5
    img = shoot(items, origin, target, size, spp or 256, fov=fov, desk="bed", light=1.25)
    # 番号を、机の上の点をカメラに写した位置に書く
    eye, tgt = np.array(origin), np.array(target)
    fwd = (tgt - eye) / np.linalg.norm(tgt - eye)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)
    focal = size[0] / 2 / math.tan(math.radians(fov) / 2)
    draw, font = ImageDraw.Draw(img), _font(30)
    for number, x, y in labels:
        d = np.array([x, y, 0.0]) - eye
        u, v, w = d @ right, d @ upv, d @ fwd
        cx, cy = size[0] / 2 + focal * u / w, size[1] / 2 - focal * v / w
        draw.ellipse([cx - 21, cy - 21, cx + 21, cy + 21], fill=(245, 245, 240))
        draw.text((cx, cy), str(number), fill=(20, 20, 20), font=font, anchor="mm")
    return img


VIEWS = ("hero", "tilt20", "seat", "front", "layout", "parts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("views", nargs="*", metavar="図", help=" ".join(VIEWS))
    ap.add_argument("--spp", type=int, help="1画素あたりの試行回数（粗く速く見たいときに小さくする）")
    ap.add_argument("--stl", type=Path, default=HERE / "stl", help="STL と build.json のある場所")
    ap.add_argument("--tilt", type=float, help="この傾きで出した傾斜台（left_base_tilt48.stl など）の形で描く")
    ap.add_argument("--out", type=Path, help="画像の出し先")
    args = ap.parse_args()
    wanted = args.views or list(VIEWS)
    unknown = [v for v in wanted if v not in VIEWS]
    if unknown:
        ap.error(f"知らない図です: {' '.join(unknown)}")
    stl = args.stl.expanduser().resolve()
    _p_plate, p, _files = manifest.load_set(stl, args.tilt)
    if args.out is None:
        tag = manifest.tilt_tag(args.tilt) if args.tilt is not None else ""
        args.out = PREVIEW if (stl == HERE / "stl" and not tag) else workdir.scratch("preview", stl.name + tag)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"寸法は {stl / manifest.NAME} の記録から：{manifest.describe(p)}。出し先は {args.out}")
    reference = Image.open(REFERENCE).convert("RGB")

    def save(img: Image.Image, name: str) -> None:
        img.save(args.out / name, quality=90)
        print(f"  {name}  {img.width} × {img.height}")

    for name in wanted:
        if name == "hero":
            img = hero(p, args.spp)
            save(img, "hero.jpg")
            save(stack(reference, img), "compare.jpg")
        elif name == "tilt20":
            img = tilt20(p, args.spp)
            save(img, "tilt20.jpg")
            save(stack(reference, img), "compare_tilt20.jpg")
        elif name == "seat":
            save(seat(p, args.spp), "seat.jpg")
        elif name == "front":
            save(front(p, args.spp), "front.jpg")
        elif name == "layout":
            save(layout(p, args.spp), "layout.jpg")
        elif name == "parts":
            save(parts_sheet(p, args.spp), "parts.jpg")
    shutil.rmtree(CACHE, ignore_errors=True)


if __name__ == "__main__":
    main()
