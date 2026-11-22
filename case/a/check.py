"""STL と、組んだ形を確かめて表にする。

    .venv/bin/python check.py                       # stl/ の一式
    .venv/bin/python check.py --tilt 20 --slope 0   # 傾斜台を left_base_tilt20_slope0.stl などに替えて確かめる
    .venv/bin/python check.py --stl ~/omnis-hole142  # 別の場所に出した一式

寸法は既定値ではなく、STL と同じ場所の build.json（build.py が書く、作ったときの寸法の記録）から読む。
確かめること：STL が記録どおりの寸法の形か、閉じた形か、外寸、スイッチ穴と爪の掛かり、姿勢と壁の向き、
部品どうしの当たり、プレートを差し込む向き、支えなしで印刷できるか、倒れにくさ。
どれかが設計値から外れていたら終了コード 1 で終わる。表は画面に出し、使い捨ての置き場にも書く。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import Manifold, Mesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build  # noqa: E402
from omnis_a import analysis, assembly, geom, manifest, parts, plate as plate_mod, shell, workdir  # noqa: E402
from omnis_a.layout import Key  # noqa: E402
from omnis_a.params import Params  # noqa: E402

PRINTED = manifest.PARTS
JP = {"left": "左", "right": "右"}
BRIDGE_MAX = 10.0       # 支えなしで渡してよい橋の長さ（mm）
CANTILEVER_MAX = 0.8    # 支えなしで張り出してよい量（mm）。0.42 mm の線 2 本ぶんまで
RIM_MIN = 0.7           # プレートの差し込み部の側面に残す壁の厚み（mm）


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures: list[str] = []

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def table(self, head: list[str], rows: list[list]) -> None:
        self.add("| " + " | ".join(head) + " |")
        self.add("| " + " | ".join("---" for _ in head) + " |")
        for row in rows:
            self.add("| " + " | ".join(str(c) for c in row) + " |")
        self.add()

    def expect(self, ok: bool, what: str) -> None:
        if not ok:
            self.failures.append(what)


def load_manifold(path: Path) -> tuple[trimesh.Trimesh, Manifold]:
    mesh = trimesh.load(path)
    m = Manifold(Mesh(vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
                      tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
    return mesh, m


def span(values) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:.2f}" if hi - lo < 0.005 else f"{lo:.2f}〜{hi:.2f}"


# ---- 1. 形と外寸 ----------------------------------------------------------------

def shapes(rep: Report, stl: Path, files: dict, p_plate: Params, p: Params) -> dict:
    rep.add("## 各 STL の形と外寸")
    rep.add()
    rep.add("「記録どおりの形」は、build.json に記録された寸法から形を作り直し、STL の体積と外寸が")
    rep.add("それと合うか、STL が build.py の書いたままか（ハッシュ）を見たもの。")
    rep.add()
    expected = build.all_parts(p_plate)
    expected.update(build.base_parts(p))
    rows, meshes = [], {}
    for name in PRINTED:
        mesh, solid = load_manifold(stl / files[name])
        meshes[name] = mesh
        closed = bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0)
        pieces = len(solid.decompose())
        size = mesh.bounds[1] - mesh.bounds[0]
        want = expected[name]
        box = np.array(want.bounding_box())
        same = (abs(want.volume() - mesh.volume) <= 1e-4 * want.volume() + 0.05
                and np.abs(box - np.concatenate(mesh.bounds)).max() < 5e-3
                and manifest.intact(stl, files[name]))
        rows.append([f"`{files[name]}`", "はい" if closed else "いいえ", pieces,
                     f"{size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f}",
                     f"{mesh.volume / 1000:.1f}", len(mesh.faces), "はい" if same else "いいえ"])
        rep.expect(closed, f"{files[name]} が閉じた形でない")
        rep.expect(pieces == 1, f"{files[name]} が {pieces} 個に分かれている")
        rep.expect(same, f"{files[name]} が、記録された寸法から作り直した形と合わない（build.py で出し直す）")
    rep.table(["ファイル", "閉じた形", "かたまりの数", "外寸 X × Y × Z（mm）", "体積（cm³）", "三角形の数",
               "記録どおりの形"], rows)
    return meshes


def assembled(rep: Report, p: Params) -> None:
    rep.add("## 組んだときの外寸（片手）")
    rep.add()
    rows = []
    for side in assembly.SIDES:
        half = assembly.half(p, side, knurl=False)
        case = half["plate"] + half["base"] + half["foot"]
        full = case + half["caps"]
        for extra in ("voice_cap", "ball", "wheel"):
            if extra in half:
                full += half[extra]
        for label, m in (("ケースだけ", case), ("キーキャップと球まで", full)):
            bb = m.bounding_box()
            rows.append([JP[side], label, f"{bb[3] - bb[0]:.1f}", f"{bb[4] - bb[1]:.1f}", f"{bb[5]:.1f}"])
    rep.table(["側", "範囲", "幅 X（mm）", "奥行き Y（mm）", "高さ Z（mm）"], rows)
    rep.add(f"底は厚さ {p.floor_top:.1f} mm の板で閉じている。開口は奥の壁のケーブルの窓だけ：")
    rep.add()
    rows = []
    for side in assembly.SIDES:
        lay = assembly.layout(p, side)
        base = assembly.base_world(p, side)
        floor = base.slice(p.foot_h / 2.0)
        holes = sum(1 for poly in geom.polygons(floor) if _signed_area(poly) < 0)
        marks = base.slice(p.pad_mark_depth / 2.0)
        closed_area = floor.area()
        window = shell.cable_window(p, lay) ^ shell.body(p, lay)
        bb = window.bounding_box()
        rows.append([JP[side], f"{closed_area:.0f}", holes, f"{floor.area() - marks.area():.0f}",
                     f"{bb[3] - bb[0]:.1f} × {bb[5] - bb[2]:.1f}", f"{bb[2] - p.floor_top:.1f}"])
        rep.expect(holes == 0, f"{side} の傾斜台の底に穴がある")
        rep.expect(abs((bb[3] - bb[0]) - p.cable_w) < 0.01 and abs((bb[5] - bb[2]) - p.cable_h) < 0.01,
                   f"{side} のケーブルの窓が設計値と違う")
    rep.table(["側", "底の台の面積（mm²）", "底を貫く穴の数", "ゴム足の目印の溝の面積（mm²）",
               "ケーブルの窓 幅 × 高さ（mm）", "底の板の上面から窓の下端まで（mm）"], rows)
    rep.add("プレートの差し込み部の側面と、スイッチ穴、爪の逃げ、キーのくぼみ、球の穴、ホイールの窓の間に残る壁の、"
            f"いちばん薄い所の厚みも測った。{RIM_MIN:.1f} mm 以上あればよいとした：")
    rep.add()
    rows = []
    for side in assembly.SIDES:
        lay = assembly.layout(p, side)
        plug, deck = shell.plug_solid(p, lay), assembly.plate_deck(p, side)
        lo, hi = 0.0, 3.0                      # 側面から内へ、どこまで肉が途切れずに続くか
        for _ in range(14):
            mid = (lo + hi) / 2.0
            if ((plug - shell.plug_solid(p, lay, shrink=mid)) - deck).volume() > 0.02:
                hi = mid
            else:
                lo = mid
        opened = sum(len(plate_mod.open_sides(p, key, plug)) for key in lay.keys)
        rows.append([JP[side], f"{lo:.2f}", opened])
        rep.expect(lo >= RIM_MIN - 0.005 or opened > 0, f"{side} のプレートで、差し込み部の側面の壁が {lo:.2f} mm しかない")
    rep.table(["側", "いちばん薄い壁の厚み（mm）", "側面まで抜いた爪の逃げの数"], rows)


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ---- 2. スイッチ穴 --------------------------------------------------------------

def _first_hit(solid: Manifold, origin: np.ndarray, angle: float, length: float, z: float) -> float | None:
    """origin から angle（度）の向きへ、高さ z で細い棒を伸ばし、最初に肉に当たるまでの距離。"""
    bar = geom.box(0.0, length, -0.05, 0.05, z - 0.05, z + 0.05)
    bar = geom.move(bar, geom.rot_z(angle), np.array([origin[0], origin[1], 0.0]))
    hit = bar ^ solid
    if hit.volume() < 1e-9:
        return None
    verts = np.asarray(hit.to_mesh().vert_properties)[:, :2]
    a = math.radians(angle)
    return float(((verts - origin) @ np.array([math.cos(a), math.sin(a)])).min())


def _column(solid: Manifold, at: np.ndarray, z_top: float) -> tuple[float, float] | None:
    """at を通る細い柱と形の重なりのうち、上端が z_top にあるものの上下の範囲。"""
    col = geom.box(at[0] - 0.05, at[0] + 0.05, at[1] - 0.05, at[1] + 0.05, -5.0, 60.0) ^ solid
    for piece in col.decompose():
        bb = piece.bounding_box()
        if abs(bb[5] - z_top) < 0.01:
            return bb[2], bb[5]
    return None


def measure_hole(solid: Manifold, centre: np.ndarray, rot: float, z_top: float, thick: float, p: Params,
                 size: float) -> dict:
    """STL の穴ひとつを測る。solid は印刷する向き、z_top はスイッチが載る面の高さ、thick は板の厚みの設計値。"""
    local = solid ^ geom.box(centre[0] - 14.0, centre[0] + 14.0, centre[1] - 14.0, centre[1] + 14.0, -1.0, 60.0)
    h = size / 2.0
    out = {"size": [], "ledge": [], "width": [], "depth": [], "open": 0, "reliefs": 0}
    z_under = z_top - thick
    for i in range(4):
        ang = rot + 90.0 * i
        a = math.radians(ang)
        normal = np.array([math.cos(a), math.sin(a)])
        r = _first_hit(local, centre, ang, h + 3.0, z_top - thick / 2.0)        # 板の厚みの中ほどで、穴のふちまで
        out["size"].append(None if r is None else r)
        col = _column(local, centre + normal * (h + p.clip_d / 2.0), z_top)       # 逃げの真上の板
        out["ledge"].append(None if col is None else col[1] - col[0])
        if z_under < 0.05:
            continue                                                             # 板の下面がそのまま爪を受ける
        z_mid = z_under / 2.0
        back = _first_hit(local, centre, ang, h + p.clip_d + 1.5, z_mid)          # 逃げの奥の壁まで
        if back is None:
            out["open"] += 1
        else:
            out["depth"].append(back - h)
        mid = centre + normal * (h + p.clip_d / 2.0)
        left = _first_hit(local, mid, ang + 90.0, p.clip_w, z_mid)
        right = _first_hit(local, mid, ang - 90.0, p.clip_w, z_mid)
        if left is not None and right is not None:
            out["width"].append(left + right)
            out["reliefs"] += 1
    return out


def latch_grip(solid: Manifold, placed: list, play: float, p: Params, size: float) -> tuple[float, bool]:
    """爪の形を穴に置き、爪の肩から板までのすき間と、引き上げたときに板に掛かるかを見る。

    placed は、挿す向き 2 通りぶんの爪（印刷する向きの座標）。play は設計上の遊び、size は穴の大きさ。
    """
    over = 0.05                                                    # 遊びより多く引き上げる量
    bite = over * p.sw_clip_w * (p.sw_body / 2.0 + p.sw_clip_out - size / 2.0)   # 爪 1 本が板に食い込む体積
    gap, caught = math.inf, True
    for latch in placed:
        gap = min(gap, solid.min_gap(latch, 5.0))
        free = (solid ^ latch.translate((0.0, 0.0, play - 0.03))).volume()
        held = (solid ^ latch.translate((0.0, 0.0, play + over))).volume()
        caught = caught and free < 1e-6 and held > 1.8 * bite       # 向かい合う 2 本とも掛かること
    return gap, caught


def holes(rep: Report, p: Params, stl: Path) -> None:
    rep.add("## スイッチ穴と、爪を受ける板")
    rep.add()
    rep.add(f"穴の設計値は {p.hole:.2f} mm 角（Choc V2 の下ハウジング {p.sw_body:.2f} mm に "
            f"{p.hole - p.sw_body:.2f} mm のすき間）。")
    rep.add(f"Choc V2 の爪の肩は、図面（CPG135301D03）でつばの下面から {p.sw_clip:.2f} mm 下にある。穴のふちは厚み "
            f"{p.clip_t:.2f} mm の平らな板で、その下は各辺の中央を幅 {p.clip_w:.1f} × 奥行き {p.clip_d:.1f} mm だけ"
            "外へ広げてある（爪の逃げ）。")
    rep.add("STL を測った。穴は板の厚みの中ほど、板の厚みは逃げの真上、逃げは板の下面と逃げの天井の中ほどで測っている。")
    rep.add("「爪の肩から板まで」は、図面から起こした爪の形（幅 "
            f"{p.sw_clip_w:.1f}、出 {p.sw_clip_out:.1f} mm）を穴に置いて測ったすき間で、スイッチが上下に動ける量になる。")
    rep.add("「引き上げると掛かる」は、その爪を遊びより 0.05 mm 多く引き上げると板に食い込むこと。"
            "スイッチをどの向きに挿しても掛かるよう、90° 回した向きでも確かめた。")
    rep.add()
    rows, worst = [], 0.0
    pitches_u, pitches_v = [], []
    labels = {"alpha": "文字キー", "thumb": "親指キー", "voice": "音声入力キー"}
    for side in assembly.SIDES:
        lay = assembly.layout(p, side)
        _mesh, m = load_manifold(stl / f"{side}_plate.stl")
        sign = 1.0 if side == "left" else -1.0
        groups: dict[str, dict] = {}
        centres = {}
        for key in lay.keys:
            centre = np.array([sign * key.u, key.v])
            z_top = p.plate_t - key.sink
            thick = parts.ledge(p, key)
            got = measure_hole(m, centre, sign * key.rot, z_top, thick, p, p.hole)
            if any(v is None for v in got["size"]) or any(v is None for v in got["ledge"]):
                rep.expect(False, f"{side} {key.name} の穴か、爪を受ける板が見つからない")
                continue
            g = groups.setdefault(key.kind, {"n": 0, "u": [], "v": [], "ledge": [], "width": [], "depth": [],
                                             "open": 0, "reliefs": 0, "need": 0, "gap": [], "caught": True,
                                             "want": thick})
            if z_top - thick > 0.05:
                g["need"] += 4                                     # 板より下に肉がある穴は、4 辺に逃げが要る
            r = got["size"]
            g["n"] += 1
            g["u"].append(r[0] + r[2])
            g["v"].append(r[1] + r[3])
            g["ledge"] += got["ledge"]
            g["width"] += got["width"]
            g["depth"] += got["depth"]
            g["open"] += got["open"]
            g["reliefs"] += got["reliefs"]
            # 穴の中心。向かい合うふちまでの距離の差の半分だけ、置いた位置からずれている
            a = math.radians(sign * key.rot)
            shift = np.array([(r[0] - r[2]) / 2.0, (r[1] - r[3]) / 2.0])
            turn = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            centres[key.name] = (centre + turn @ shift) * np.array([sign, 1.0])
            worst = max(worst, abs(r[0] + r[2] - p.hole), abs(r[1] + r[3] - p.hole))
            placed = []
            for quarter in (0, 1):
                latch = parts.place_on_key(p, key, parts.latch(p, quarter)).translate((0.0, 0.0, p.plate_t))
                placed.append(latch if side == "left" else latch.mirror((1.0, 0.0, 0.0)))
            gap, caught = latch_grip(m, placed, p.sw_clip - thick, p, p.hole)
            g["gap"].append(gap)
            g["caught"] = g["caught"] and caught
        for kind, g in groups.items():
            relief = "なし（板の下面で受ける）"
            if g["reliefs"]:
                relief = f"{g['reliefs']} か所、幅 {span(g['width'])}"
                if g["depth"]:
                    relief += f"、奥行き {span(g['depth'])}"
                if g["open"]:
                    relief += f"（うち {g['open']} か所は外へ抜ける）"
            rows.append([JP[side], labels[kind], g["n"], span(g["u"]), span(g["v"]), span(g["ledge"]), relief,
                         span(g["gap"]), "はい" if g["caught"] else "いいえ"])
            rep.expect(max(abs(v - g["want"]) for v in g["ledge"]) < 0.01,
                       f"{side} {labels[kind]}：爪を受ける板の厚みが {span(g['ledge'])} mm（設計値 {g['want']:.2f}）")
            rep.expect(max(abs(v - (p.sw_clip - g["want"])) for v in g["gap"]) < 0.01,
                       f"{side} {labels[kind]}：爪の肩から板まで {span(g['gap'])} mm")
            rep.expect(g["caught"], f"{side} {labels[kind]}：引き上げても爪が板に掛からない穴がある")
            rep.expect(p.sw_clip - p.sw_height_tol - g["want"] > 1e-9,
                       f"{side} {labels[kind]}：板の厚み {g['want']:.2f} mm では、爪の肩が公差の下限"
                       f"（{p.sw_clip - p.sw_height_tol:.2f} mm）のスイッチで爪が開かない")
            rep.expect(g["reliefs"] == g["need"],
                       f"{side} {labels[kind]}：爪の逃げが {g['reliefs']} か所（要るのは {g['need']} か所）")
            rep.expect(all(abs(v - p.clip_w) < 0.01 for v in g["width"])
                       and all(abs(v - p.clip_d) < 0.01 for v in g["depth"]),
                       f"{side} {labels[kind]}：爪の逃げの寸法が設計値と違う")
        cols = len(p.stagger)
        for c in range(cols):
            for r in range(p.rows - 1):
                a, b = centres.get(f"c{c}r{r}"), centres.get(f"c{c}r{r + 1}")
                if a is not None and b is not None:
                    pitches_v.append(float(b[1] - a[1]))
        for c in range(cols - 1):
            a, b = centres.get(f"c{c}r0"), centres.get(f"c{c + 1}r0")
            if a is not None and b is not None:
                pitches_u.append(float(b[0] - a[0]))
    rep.table(["側", "キー", "穴の数", "穴の幅 u（mm）", "穴の奥行き v（mm）", "爪を受ける板の厚み（mm）",
               "爪の逃げ（mm）", "爪の肩から板まで（mm）", "引き上げると掛かる"], rows)
    rep.add(f"穴の設計値からのずれは最大 {worst:.3f} mm。列のピッチは {span(pitches_u)} mm、"
            f"段のピッチは {span(pitches_v)} mm。")
    rep.add(f"爪の肩の寸法の公差は ±{p.sw_height_tol:.2f} mm で、下限の {p.sw_clip - p.sw_height_tol:.2f} mm でも"
            "板の下面より下に来ることも確かめている。")
    rep.add("逃げの天井は下に支えのない所へ渡す線なので、刷り上がりでは少し垂れる。垂れたぶん、この遊びは減る。")
    rep.add()
    rep.expect(worst < 0.01, f"スイッチ穴が設計値から {worst:.3f} mm ずれている")
    rep.expect(all(abs(v - p.pitch) < 0.01 for v in pitches_u + pitches_v), "ピッチが設計値と違う")

    _mesh, m = load_manifold(stl / "fit_test.stl")
    sizes, ledges = parts.fit_grid(p)
    rep.add("試し刷りの小片 `fit_test.stl`。穴の大きさは手前の辺の刻みの数で見分ける。刻みのある手前の列が本番の"
            f"プレートと同じ断面で、奥の列は板の厚みを {p.clip_alt:.1f} mm にしてある：")
    rep.add()
    rows = []
    for j, thick in enumerate(ledges):
        for i, size in enumerate(sizes):
            centre = np.array(parts.fit_centre(i, j))
            got = measure_hole(m, centre, 0.0, p.plate_t, thick, p, size)
            if any(v is None for v in got["size"] + got["ledge"]) or got["reliefs"] != 4 or got["open"]:
                rep.expect(False, f"fit_test.stl の穴（列 {j + 1}、刻み {i + 1}）が設計どおりでない")
                continue
            r = got["size"]
            key = Key("t", "alpha", centre[0], centre[1])
            placed = [parts.place_on_key(p, key, parts.latch(p, q)).translate((0.0, 0.0, p.plate_t)) for q in (0, 1)]
            gap, caught = latch_grip(m, placed, p.sw_clip - thick, p, size)
            rows.append(["手前" if j == 0 else "奥", i + 1, f"{size:.2f}", f"{r[0] + r[2]:.2f} × {r[1] + r[3]:.2f}",
                         f"{thick:.2f}", span(got["ledge"]), f"{span(got['width'])} × {span(got['depth'])}",
                         f"{gap:.2f}", "はい" if caught else "いいえ"])
            rep.expect(abs(r[0] + r[2] - size) < 0.01 and abs(r[1] + r[3] - size) < 0.01
                       and max(abs(v - thick) for v in got["ledge"]) < 0.01 and caught,
                       f"fit_test.stl の穴（列 {j + 1}、刻み {i + 1}）が設計値と違う")
    rep.table(["列", "刻みの数", "穴の設計値（mm）", "測った穴（mm）", "板の厚みの設計値（mm）", "測った板の厚み（mm）",
               "爪の逃げ 幅 × 奥行き（mm）", "爪の肩から板まで（mm）", "引き上げると掛かる"], rows)
    rep.expect(len(rows) == len(sizes) * len(ledges), "fit_test.stl の穴の数が設計と違う")


# ---- 3. 姿勢 --------------------------------------------------------------------

def _wall_faces(mesh: trimesh.Trimesh, normal: np.ndarray, offset: float) -> np.ndarray:
    """メッシュの面のうち、決めた平面（外向きの単位ベクトル normal、normal·点 = offset）の上にあるもの。"""
    on_plane = np.abs(mesh.triangles_center @ normal - offset) < 2e-3
    return np.flatnonzero((mesh.face_normals @ normal > 0.99999) & on_plane)


def _top_edge(mesh: trimesh.Trimesh, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """壁の面の上の縁の両端（低い方、高い方）。いちばん高い頂点と、そこから離れた側でいちばん高い頂点。"""
    pts = mesh.vertices[np.unique(mesh.faces[faces])]
    high = pts[pts[:, 2].argmax()]
    far = pts[np.linalg.norm(pts[:, :2] - high[:2], axis=1) > 0.6 * np.ptp(pts[:, :2], axis=0).max()]
    return far[far[:, 2].argmax()], high


def _rise(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.atan2(b[2] - a[2], math.hypot(b[0] - a[0], b[1] - a[1])))


def posture(rep: Report, p: Params, meshes: dict) -> None:
    rep.add("## 姿勢と壁の向き")
    rep.add()
    rep.add("傾斜台の STL を測った。プレートが載る面は、手前の壁の上の縁と外側の壁の上の縁を通る。")
    rep.add("手前の壁の上の縁が水平から上がる角度が左右の傾き、外側の壁の上の縁が上がる角度が前後の傾き。")
    rep.add("合わせた傾きは、その 2 本の縁が張る面と水平のなす角。壁の寝かせは、壁の外面と鉛直のなす角。")
    rep.add(f"設計値は左右 {p.tilt:.1f}°、前後 {p.slope:.1f}°、合わせて {shell.steepest(p):.1f}°。")
    rep.add()
    rows = []
    for side in assembly.SIDES:
        mesh = meshes[f"{side}_base"]
        lay = assembly.layout(p, side)
        rot, trans = shell.frame(p, lay)
        want = shell.wall_leans(p, lay)
        flip = np.array([1.0 if side == "left" else -1.0, 1.0, 1.0])
        found, lean = {}, {}
        for name, pl in shell.walls(p, lay):
            if name not in shell.MAINS:
                continue
            world = pl.moved(rot, trans)
            faces = _wall_faces(mesh, world.n * flip, world.d)
            if len(faces) == 0:
                rep.expect(False, f"{side} の傾斜台で、{name} の壁の面が見つからない")
                continue
            found[name] = faces
            normal = (mesh.face_normals[faces] * mesh.area_faces[faces][:, None]).sum(axis=0)
            lean[name] = math.degrees(math.asin(normal[2] / np.linalg.norm(normal)))
        if not all(k in found for k in ("front", "outer", "cheek", "back")):
            continue
        a0, b0 = _top_edge(mesh, found["front"])
        a1, b1 = _top_edge(mesh, found["outer"])
        tilt, slope = _rise(a0, b0), _rise(a1, b1)
        normal = np.cross(b0 - a0, b1 - a1)
        both = math.degrees(math.acos(abs(normal[2]) / np.linalg.norm(normal)))
        rows.append([JP[side], f"{tilt:.2f}°", f"{slope:.2f}°", f"{both:.2f}°", f"{lean['front']:.2f}°",
                     f"{lean['outer']:.2f}°", f"{lean['cheek']:.2f}°", f"{lean['back']:.2f}°"])
        rep.expect(abs(tilt - p.tilt) < 0.02 and abs(slope - p.slope) < 0.02,
                   f"{side} の傾斜台の姿勢が、左右 {tilt:.2f}°、前後 {slope:.2f}°")
        rep.expect(abs(both - shell.steepest(p)) < 0.02, f"{side} の傾斜台の合わせた傾きが {both:.2f}°")
        rep.expect(all(abs(lean[k] - want[k]) < 0.02 for k in lean),
                   f"{side} の傾斜台の壁の寝かせが設計値と違う")
    rep.table(["側", "左右の傾き", "前後の傾き", "合わせた傾き", "手前の壁の寝かせ", "外側の壁の寝かせ",
               "内側の壁の寝かせ", "奥の壁の寝かせ"], rows)
    lay = assembly.layout(p, "left")
    corners = {"外側の手前": (lay.u_out, lay.v_front), "内側の手前": (lay.u_in, lay.v_front),
               "内側の奥": (lay.u_in, lay.v_back), "外側の奥": (lay.u_out, lay.v_back)}
    rep.add("上面の四隅が机から上がる高さ（面取りをする前の角の位置）："
            + "、".join(f"{k} {shell.world_point(p, lay, (u, v, p.top))[2]:.1f} mm" for k, (u, v) in corners.items()) + "。")
    rep.add()


# ---- 4. 当たり ------------------------------------------------------------------

def clearances(rep: Report, p: Params) -> None:
    rep.add("## 部品どうしの当たり")
    rep.add()
    rep.add("組んだ位置での重なりの体積と、いちばん近い所のすき間。重なりが 0 なら食い込んでいない。")
    rep.add(f"キーキャップは印刷する場合の形（外形 {p.cap:.1f} mm 角）で確かめた。すき間 0.00 は、そこで触れて載っていることを表す。")
    rep.add(f"「公差の端」は、図面の公差のうちキャップにいちばん厳しい組み合わせ：ストローク {p.travel_max:.2f} mm"
            f"（{p.sw_travel:.1f}＋{p.sw_travel_tol:.2f}）、ハウジングの高さ＋{p.sw_height_tol:.2f} mm。")
    rep.add()
    rows, spare = [], []
    for side in assembly.SIDES:
        jp = JP[side]
        rest = assembly.half(p, side, knurl=False)
        down = assembly.half(p, side, press=p.sw_travel, knurl=False)
        edge = assembly.half(p, side, press=p.travel_max, knurl=False, tall=p.sw_height_tol)
        lay = assembly.layout(p, side)
        rot, _trans = shell.frame(p, lay)
        push = rot @ np.array([0.0, 0.0, -1.0])
        body_only = None      # スイッチのうち、穴に入る下ハウジングだけ
        for key in lay.keys:
            b = parts.place_on_key(p, key, geom.prism(geom.rect(0, 0, p.sw_body, p.sw_body), -p.sw_below, 0.0))
            body_only = b if body_only is None else body_only + b
        body_only = shell.to_world(p, lay, body_only)

        def caps_of(half: dict) -> Manifold:
            return half["caps"] + half["voice_cap"] if "voice_cap" in half else half["caps"]

        base = rest["base"] + rest["foot"]
        pairs = [
            ("押していないとき", "傾斜台 と プレート", base, rest["plate"], 0.0),
            ("押していないとき", "傾斜台 と スイッチの下ハウジング・中央の柱", base, rest["switch_lower"], 0.5),
            ("押していないとき", "傾斜台 と スイッチの電気の端子", base, rest["switch_pins"], 0.5),
            ("押していないとき", "プレート と スイッチの下ハウジング", rest["plate"], body_only, 0.05),
            ("押していないとき", "プレート と キーキャップ", rest["plate"], caps_of(rest), 0.3),
            ("押し切ったとき", "プレート と キーキャップ", rest["plate"], caps_of(down), 0.3),
            ("押し切ったとき", "スイッチのハウジング と キーキャップ", rest["switch_upper"], caps_of(down), 0.2),
            ("公差の端で押し切ったとき", "プレート と キーキャップ", rest["plate"], caps_of(edge), 0.3),
            ("公差の端で押し切ったとき", "スイッチのハウジング と キーキャップ", edge["switch_upper"], caps_of(edge), 0.2),
        ]
        # 押し切った位置から、キャップがハウジングに当たるまでに、あとどれだけ下がれるか
        for label, group in (("文字キーと親指キーのキャップ", "caps"), ("音声入力キーのキャップ", "voice_cap")):
            if group not in rest:
                continue
            cells = []
            for half in (down, edge):
                lo, hi = 0.0, 3.0
                for _ in range(22):
                    mid = (lo + hi) / 2.0
                    if (half["switch_upper"] ^ half[group].translate(tuple(push * mid))).volume() > 1e-7:
                        hi = mid
                    else:
                        lo = mid
                cells.append(lo)
            spare.append([jp, label, f"{cells[0]:.2f}", f"{cells[1]:.2f}"])
            rep.expect(cells[1] >= p.cap_margin - 0.011,
                       f"{side}: {label}は、公差の端のスイッチを押し切ると、ハウジングまで {cells[1]:.2f} mm しか残らない")
        if "wheel" in rest:
            pairs += [
                ("押していないとき", "プレート と ホイール（窓のまん中）", rest["plate"], rest["wheel"], p.wheel_play),
                ("押していないとき", "傾斜台 と ホイール", base, rest["wheel"], 0.5),
                ("公差の端で押し切ったとき", "キーキャップ と ホイール", caps_of(edge), rest["wheel"], 0.5),
            ]
        if "ball" in rest:
            pairs += [
                ("押していないとき", "プレート と 球", rest["plate"], rest["ball"], 0.0),
                ("押していないとき", "傾斜台 と 球", base, rest["ball"], 0.5),
                ("公差の端で押し切ったとき", "キーキャップ と 球", caps_of(edge), rest["ball"], 0.5),
            ]
        for state, label, a, b, need in pairs:
            overlap = (a ^ b).volume()
            gap = a.min_gap(b, 10.0)
            rows.append([jp, state, label, f"{overlap:.3f}", f"{gap:.2f}"])
            rep.expect(overlap < 0.05, f"{side}: {label}（{state}）が {overlap:.3f} mm³ 重なっている")
            rep.expect(gap >= need - 0.011, f"{side}: {label}（{state}）のすき間が {gap:.2f} mm")
    rep.table(["側", "状態", "組み合わせ", "重なり（mm³）", "すき間（mm）"], rows)
    rep.add("上の表の「スイッチのハウジング と キーキャップ」は、キャップの内側の角とハウジングの角の、横のすき間。")
    rep.add("押し切ったキャップの下端がハウジングに当たるまでの残りは、キャップをさらに押し下げて測った。"
            f"公差の端でも {p.cap_margin:.2f} mm が残るように、キャップの下端の高さを決めている。")
    rep.add()
    rep.table(["側", "キャップ", f"公称のストローク（{p.sw_travel:.1f} mm）で押し切ったあとの残り（mm）",
               f"公差の端（{p.travel_max:.2f} mm、ハウジング＋{p.sw_height_tol:.2f} mm）で押し切ったあとの残り（mm）"], spare)

    lay = assembly.layout(p, "left")
    if lay.wheel is not None:
        rep.add("ホイールの軸もキー面と一緒に傾くので、ホイールは低い側（外側）へ寄る。窓の両端の壁には、軸の下だけを "
                f"{p.wheel_pad:.1f} mm 出した当たりがあり、寄ったホイールの端面はそこで止まる。")
        rep.add()
        deck = assembly.plate_deck(p, "left")
        pads = plate_mod.wheel_pads(p, lay)
        walls = deck - pads              # 当たりを除いた、窓の壁
        wheel = parts.wheel_in_place(p, lay, knurl=False)
        rows = []
        for label, sign in (("低い側（外側）", -1.0), ("高い側（内側）", 1.0)):
            # 寄せていき、当たりに触れる位置を探す
            lo, hi = 0.0, p.wheel_play + p.wheel_pad + 0.5
            for _ in range(24):
                mid = (lo + hi) / 2.0
                if (deck ^ wheel.translate((sign * mid, 0.0, 0.0))).volume() > 1e-7:
                    hi = mid
                else:
                    lo = mid
            touch = lo
            moved = wheel.translate((sign * touch, 0.0, 0.0))
            pad_gap = pads.min_gap(moved, 5.0)
            wall_gap = walls.min_gap(moved, 5.0)
            rows.append([label, f"{touch:.2f}", f"{wall_gap:.2f}"])
            rep.expect(abs(touch - p.wheel_play) < 0.02, f"ホイールが{label}へ {touch:.2f} mm 動く（設計値 {p.wheel_play:.2f}）")
            rep.expect(pad_gap < 0.02, f"ホイールを{label}へ寄せても当たりに触れない")
            rep.expect(wall_gap >= 0.3, f"ホイールを{label}へ寄せると、窓の壁とのすき間が {wall_gap:.2f} mm")
        rep.table(["寄せる向き", "当たりに触れるまでに動く量（mm）", "そのときの、ホイールと窓の壁のすき間（mm）"], rows)


# ---- 5. 差し込む向き ------------------------------------------------------------

def insertion(rep: Report, p: Params) -> None:
    rep.add("## プレートを差し込む向き")
    rep.add()
    rep.add("スイッチ、ホイール、球を付けたプレートを、載せる位置から離して、傾斜台と重なる体積を見た。")
    rep.add("傾斜面（キー面）に垂直な向きなら、どの距離でも重ならない。真上へ持ち上げると、プレートの差し込み部が、"
            "寝ている壁（キー面が高い側の壁）に食い込む。")
    rep.add()
    lifts = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
    rows = []
    for side in assembly.SIDES:
        lay = assembly.layout(p, side)
        half = assembly.half(p, side, knurl=False)
        base = half["base"] + half["foot"]
        moving = half["plate"] + half["switch_lower"] + half["switch_pins"]
        for extra in ("wheel", "ball"):
            if extra in half:
                moving += half[extra]
        rot, _trans = shell.frame(p, lay)
        for label, direction, must_clear in (("傾斜面に垂直", rot @ np.array([0.0, 0.0, 1.0]), True),
                                             ("真上", np.array([0.0, 0.0, 1.0]), False)):
            volumes = [(base ^ moving.translate(tuple(direction * d))).volume() for d in lifts]
            rows.append([JP[side], label] + [f"{v:.1f}" for v in volumes])
            if must_clear:
                rep.expect(max(volumes) < 0.05, f"{side}: 傾斜面に垂直に差し込んでも {max(volumes):.1f} mm³ 重なる")
    rep.table(["側", "離す向き"] + [f"{d:g} mm（mm³）" for d in lifts], rows)


# ---- 6. 印刷 --------------------------------------------------------------------

def printing(rep: Report, meshes: dict, files: dict) -> None:
    rep.add("## 支えなしで印刷できるか")
    rep.add()
    rep.add("STL の向き（印刷する向き）のまま、下を向いた面を調べた。角度は垂直の壁を 0°、天井を 90° として測る。")
    rep.add("天井は、ひとつながりの面ごとに縁を調べた。縁の下に壁が立ち上がっていれば、そこが支えになる。")
    rep.add("「橋」は両側の支えの間に渡す天井で、支えから支えまでの間隔を測る。「張り出し」は片側にしか支えのない天井で、")
    rep.add(f"支えからの距離を測る。橋は {BRIDGE_MAX:g} mm、張り出しは {CANTILEVER_MAX:g} mm までを支えなしで印刷できるとした。")
    rep.add()
    rows = []
    for name in PRINTED:
        oh = analysis.overhang(meshes[name])
        rows.append([f"`{files[name]}`", f"{oh.steepest:.0f}°", f"{oh.over_limit:.0f}", f"{oh.ceiling:.0f}",
                     f"{oh.bridge:.1f}" if oh.bridge > 0 else "なし",
                     f"{oh.cantilever:.2f}" if oh.cantilever > 0 else "なし"])
        rep.expect(oh.bridge <= BRIDGE_MAX + 1e-6, f"{files[name]} に {oh.bridge:.1f} mm の橋がある")
        rep.expect(oh.cantilever <= CANTILEVER_MAX + 1e-6, f"{files[name]} に {oh.cantilever:.2f} mm の張り出しがある")
        rep.expect(oh.over_limit <= 20.0, f"{files[name]} に 47.5° を超える斜面が {oh.over_limit:.0f} mm² ある")
    rep.table(["ファイル", "下向きの斜面の最大（天井を除く）", "47.5° を超える斜面（mm²）", "天井の面積（mm²）",
               "橋の長さの最大（mm）", "張り出しの最大（mm）"], rows)


# ---- 7. 倒れにくさ --------------------------------------------------------------

def _force(value: float) -> str:
    return "倒れない" if math.isinf(value) else f"{value:.1f} N（約 {value / 9.81e-3:.0f} gf）"


def tipping(rep: Report, p: Params) -> None:
    rep.add("## 倒れにくさの目安")
    rep.add()
    rep.add("キーを、キー面に垂直に押したとする。押す力の向きを机まで延ばした点が支えの内側にあれば、")
    rep.add("どれだけ強く押しても倒れない。外側にあるときは、支えの縁を支点にして倒れ始める力を出した。")
    rep.add(f"支えは、傾斜台の底の目印（輪の溝）に貼った平らなゴム足 4 枚（直径 {p.pad_d:g} mm、厚み {p.pad_h:g} mm）とした。"
            f"ゴム足の縁は台の縁から {p.pad_edge:g} mm 内側に来る。ゴム足の厚みのぶんケースが浮き、"
            "力の向きが机に届く点はそのぶん、キー面が高い側へずれる。")
    rep.add(f"重さは、印刷物を PLA（1.24 g/cm³）の体積に、スライスした重さとの比（プレート "
            f"{analysis.FILL['plate']}、傾斜台 {analysis.FILL['base']}）を掛け、スイッチを 1 個 "
            f"{analysis.SWITCH_G} g として見積もった。")
    rep.add()
    rows, notes = [], []
    for side in assembly.SIDES:
        st = analysis.stability(p, side)
        jp = JP[side]
        for col in range(len(p.stagger)):
            names = [f"c{col}r{r}" for r in range(p.rows)]
            force = min(st.tip_force[n] for n in names)
            over = max(st.landing[n] for n in names)
            rows.append([jp, f"{col + 1} 列め", f"{over:.1f}" if over > 0 else "内側", _force(force)])
        for key in assembly.layout(p, side).keys:
            if key.kind == "alpha":
                continue
            label = {"space": "親指キー（Space）", "enter": "親指キー（Enter）", "thumb2": "親指キー（外側）",
                     "voice": "音声入力キー"}[key.name]
            rows.append([jp, label, f"{st.landing[key.name]:.1f}" if st.landing[key.name] > 0 else "内側",
                         _force(st.tip_force[key.name])])
        worst = min(st.tip_force.values())
        rep.expect(worst > 4.0, f"{side}: {worst:.1f} N で倒れ始める")
        notes.append(f"{jp}：重さの見積もり {st.mass:.0f} g、重心は外側の手前の角から {st.cg[0]:.1f} mm 内側、"
                     f"{st.cg[1] - assembly.layout(p, side).v_front:.1f} mm 奥、底から {st.cg[2]:.1f} mm の高さ。")
    rep.table(["側", "キー（列は小指側から数える）", "力の向きが支えの外に出る量（mm）", "倒れ始める力"], rows)
    for note in notes:
        rep.add(note)
    rep.add()

    rep.add("ゴム足の貼り方を変えたとき（いちばん倒れやすいキー）：")
    rep.add()
    rows = []
    for label, support in ((f"平らなゴム足（直径 {p.pad_d:g}、厚み {p.pad_h:g} mm）を目印の位置に貼る", "pads"),
                           ("同じ位置に、中心だけで机に触れるゴム足（半球形など）を貼る", "centres"),
                           ("ゴム足を貼らず、台をじかに机に置く", "foot")):
        cells = []
        for side in assembly.SIDES:
            st = analysis.stability(p, side, support)
            worst = min(st.tip_force, key=st.tip_force.get)
            cells += [f"{st.landing[worst]:.1f}" if st.landing[worst] > 0 else "内側", _force(st.tip_force[worst])]
        rows.append([label] + cells)
    rep.table(["支え", "左：支えの外に出る量（mm）", "左：倒れ始める力", "右：支えの外に出る量（mm）", "右：倒れ始める力"], rows)

    rep.add("傾斜台だけを別の姿勢で作ったとき（プレートは同じ。左、ゴム足は上と同じ）：")
    rep.add()
    rows = []
    seen = []
    for t, sl in ((p.tilt, p.slope), (20.0, 0.0), (43.0, 0.0)):
        if (t, sl) in seen:
            continue
        seen.append((t, sl))
        q = p.with_(tilt=t, slope=sl)
        st = analysis.stability(q, "left")
        leans = shell.wall_leans(q, assembly.layout(q, "left"))
        bb = assembly.base_world(q, "left").bounding_box()
        top = shell.world_point(q, assembly.layout(q, "left"), (assembly.layout(q, "left").u_in,
                                                              assembly.layout(q, "left").v_back, q.top))[2]
        rows.append([f"左右 {t:g}°、前後 {sl:g}°" + ("（この一式）" if (t, sl) == (p.tilt, p.slope) else ""),
                     f"{shell.steepest(q):.1f}°", f"{leans['cheek']:.0f}°", f"{leans['back']:.0f}°",
                     f"{bb[3] - bb[0]:.0f} × {bb[4] - bb[1]:.0f}", f"{max(top, shell.world_point(q, assembly.layout(q, 'left'), (assembly.layout(q, 'left').u_in, assembly.layout(q, 'left').v_front, q.top))[2]):.0f}",
                     _force(min(st.tip_force.values()))])
    rep.table(["姿勢", "合わせた傾き", "内側の壁の寝かせ", "奥の壁の寝かせ", "足もとの幅 × 奥行き（mm）",
               "上面のいちばん高い角（mm）", "倒れ始める力"], rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stl", type=Path, default=HERE / "stl", help="STL と build.json のある場所")
    ap.add_argument("--tilt", type=float, help="この左右の傾きで出した傾斜台（left_base_tilt20_slope0.stl など）を確かめる")
    ap.add_argument("--slope", type=float, help="この前後の傾きで出した傾斜台を確かめる")
    ap.add_argument("--report", type=Path, help="表を書くファイル（既定は使い捨ての置き場）")
    args = ap.parse_args()
    stl = args.stl.expanduser()
    tag = manifest.posture_tag(args.tilt, args.slope)
    report = args.report or workdir.scratch(f"check-report{tag}.md")

    p_plate, p, files = manifest.load_set(stl, args.tilt, args.slope)
    rep = Report()
    rep.add(f"確かめた場所は `{stl}`。寸法は `{manifest.NAME}` の記録から読んだ：プレートの一式は{manifest.describe(p_plate)}、"
            f"傾斜台の姿勢は左右 {p.tilt:g}°、前後 {p.slope:g}°。")
    rep.add()
    meshes = shapes(rep, stl, files, p_plate, p)
    assembled(rep, p)
    holes(rep, p, stl)
    posture(rep, p, meshes)
    clearances(rep, p)
    insertion(rep, p)
    printing(rep, meshes, files)
    tipping(rep, p)

    rep.add("## 結果")
    rep.add()
    if rep.failures:
        rep.add("設計値から外れた所がある：")
        rep.add()
        for f in rep.failures:
            rep.add(f"- {f}")
    else:
        rep.add("すべての項目が設計値どおりだった。")
    others = [t for t in manifest.variants(stl) if t != tag]
    if others:
        rep.add()
        rep.add("この場所には、姿勢だけ変えた傾斜台もある。確かめるには "
                + "、".join(f"`check.py {manifest.tag_options(t)}`" for t in others) + "。")
    text = "\n".join(rep.lines) + "\n"
    print(text)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text)
    print(f"表を書いた場所: {report}")
    sys.exit(1 if rep.failures else 0)


if __name__ == "__main__":
    main()
