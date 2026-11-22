"""stl/ の STL と、組んだ形を確かめて表にする。

    .venv/bin/python check.py            # 画面に出し、out/check-report.md にも書く

確かめること：閉じた形か、外寸、スイッチ穴の寸法、傾き、部品どうしの当たり、
支えなしで印刷できるか、倒れにくさ。どれかが設計値から外れていたら終了コード 1 で終わる。
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

from omnis_a import analysis, assembly, geom, parts, shell  # noqa: E402
from omnis_a.params import DEFAULT, Params  # noqa: E402

PRINTED = ["left_plate", "right_plate", "left_base", "right_base", "left_wheel",
           "keycap_1u", "keycap_space", "keycap_enter", "keycap_voice", "fit_test"]


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


# ---- 1. 形と外寸 ----------------------------------------------------------------

def shapes(rep: Report, stl: Path) -> dict:
    rep.add("## 各 STL の形と外寸")
    rep.add()
    rows, meshes = [], {}
    for name in PRINTED:
        mesh, solid = load_manifold(stl / f"{name}.stl")
        meshes[name] = mesh
        closed = bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0)
        pieces = len(solid.decompose())
        size = mesh.bounds[1] - mesh.bounds[0]
        rows.append([f"`{name}.stl`", "はい" if closed else "いいえ", pieces,
                     f"{size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f}",
                     f"{mesh.volume / 1000:.1f}", len(mesh.faces)])
        rep.expect(closed, f"{name}.stl が閉じた形でない")
        rep.expect(pieces == 1, f"{name}.stl が {pieces} 個に分かれている")
    rep.table(["ファイル", "閉じた形", "かたまりの数", "外寸 X × Y × Z（mm）", "体積（cm³）", "三角形の数"], rows)
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
            rows.append([{"left": "左", "right": "右"}[side], label,
                         f"{bb[3] - bb[0]:.1f}", f"{bb[4] - bb[1]:.1f}", f"{bb[5]:.1f}"])
    rep.table(["側", "範囲", "幅 X（mm）", "奥行き Y（mm）", "高さ Z（mm）"], rows)


# ---- 2. スイッチ穴 --------------------------------------------------------------

def _hole_at(polys: list[np.ndarray], centre: np.ndarray) -> np.ndarray | None:
    for poly in polys:
        if np.linalg.norm(poly.mean(axis=0) - centre) < 1.0 and len(poly) <= 8:
            return poly
    return None


def holes(rep: Report, p: Params, stl: Path) -> None:
    rep.add("## スイッチ穴")
    rep.add()
    rep.add(f"設計値は {p.hole:.2f} mm 角（Choc V2 の下ハウジング {p.sw_body:.2f} mm に "
            f"{p.hole - p.sw_body:.2f} mm のすき間）。穴のふちで爪が掛かる板の厚みは {p.land:.1f} mm。")
    rep.add("STL を、ふちの板の厚みの中ほどで切って測った。")
    rep.add()
    rows, worst = [], 0.0
    pitches_u, pitches_v = [], []
    for side in assembly.SIDES:
        lay = assembly.layout(p, side)
        _mesh, m = load_manifold(stl / f"{side}_plate.stl")
        sign = 1.0 if side == "left" else -1.0
        groups: dict[str, list] = {}
        centres = {}
        for key in lay.keys:
            z = p.plate_t - key.sink - p.land / 2.0
            polys = geom.polygons(m.slice(z))
            poly = _hole_at(polys, np.array([sign * key.u, key.v]))
            label = {"alpha": "文字キー", "thumb": "親指キー", "voice": "音声入力キー"}[key.kind]
            if poly is None:
                rep.expect(False, f"{side} {key.name} の穴が見つからない")
                continue
            size = poly.max(axis=0) - poly.min(axis=0)
            groups.setdefault(label, []).append(size)
            centres[key.name] = poly.mean(axis=0) * np.array([sign, 1.0])
            worst = max(worst, float(np.abs(size - p.hole).max()))
        for label, sizes in groups.items():
            arr = np.array(sizes)
            rows.append([{"left": "左", "right": "右"}[side], label, len(sizes),
                         f"{arr[:, 0].min():.2f}〜{arr[:, 0].max():.2f}",
                         f"{arr[:, 1].min():.2f}〜{arr[:, 1].max():.2f}"])
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
    rep.table(["側", "キー", "穴の数", "幅 u（mm）", "奥行き v（mm）"], rows)
    rep.add(f"設計値からのずれは最大 {worst:.3f} mm。列のピッチは {min(pitches_u):.2f}〜{max(pitches_u):.2f} mm、"
            f"段のピッチは {min(pitches_v):.2f}〜{max(pitches_v):.2f} mm。")
    rep.add()
    rep.expect(worst < 0.01, f"スイッチ穴が設計値から {worst:.3f} mm ずれている")
    rep.expect(all(abs(v - p.pitch) < 0.01 for v in pitches_u + pitches_v), "ピッチが設計値と違う")

    _mesh, m = load_manifold(stl / "fit_test.stl")
    polys = geom.polygons(m.slice(p.plate_t - p.land / 2.0))
    def squarish(poly: np.ndarray) -> bool:
        size = poly.max(axis=0) - poly.min(axis=0)
        return 13.0 < size.min() and size.max() < 15.5

    squares = sorted((poly for poly in polys if squarish(poly)), key=lambda q: q[:, 0].mean())
    rep.add("試し刷りの小片 `fit_test.stl` の穴（横の刻みの数で見分ける）：")
    rep.add()
    rep.table(["刻みの数", "設計値（mm）", "測った値（mm）"],
              [[i + 1, f"{p.hole - 0.1 + 0.1 * i:.2f}",
                " × ".join(f"{v:.2f}" for v in (sq.max(0) - sq.min(0)))] for i, sq in enumerate(squares)])
    rep.expect(len(squares) == 3, "fit_test.stl の穴が 3 個でない")


# ---- 3. 傾き --------------------------------------------------------------------

def _wall_top_edge(mesh: trimesh.Trimesh, normal_y: float, inward: float) -> tuple[np.ndarray, np.ndarray]:
    """手前（normal_y = -1）か奥（+1）の壁の外面で、内側へ向かって上がるいちばん長い縁。

    プレートが載る面は、この縁でそれぞれの壁の外面と交わる。
    """
    y_wall = mesh.bounds[0][1] if normal_y < 0 else mesh.bounds[1][1]
    faces = np.flatnonzero((np.abs(mesh.face_normals[:, 1] - normal_y) < 1e-6)
                           & (np.abs(mesh.triangles_center[:, 1] - y_wall) < 1e-3))
    edges = np.sort(mesh.faces[faces][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    best = None
    for a, b in mesh.vertices[uniq[counts == 1]]:
        if (b[0] - a[0]) * inward < 0:
            a, b = b, a
        run, rise = abs(b[0] - a[0]), b[2] - a[2]
        if run < 1e-6 or rise / run < 0.05:
            continue
        length = math.hypot(run, rise)
        if best is None or length > best[0]:
            best = (length, a, b)
    return best[1], best[2]


def tilt(rep: Report, p: Params, meshes: dict) -> None:
    rep.add("## 傾き")
    rep.add()
    rep.add("`*_base.stl` で、プレートが載る面が手前と奥の壁の外面と交わる縁を測った。")
    rep.add(f"内側の壁は、外面の向きを測った。設計値は傾き {p.tilt:.1f}°、内側の壁の寝かせ {p.lean:.1f}°（垂直から）。")
    rep.add()
    rows = []
    for side in assembly.SIDES:
        mesh = meshes[f"{side}_base"]
        inward = 1.0 if side == "left" else -1.0
        a0, b0 = _wall_top_edge(mesh, -1.0, inward)
        a1, b1 = _wall_top_edge(mesh, 1.0, inward)
        front = math.degrees(math.atan2(b0[2] - a0[2], abs(b0[0] - a0[0])))
        back = math.degrees(math.atan2(b1[2] - a1[2], abs(b1[0] - a1[0])))
        # 同じ x での高さを手前と奥で比べる。差がなければ前後には傾いていない。
        z_back = a1[2] + (b1[2] - a1[2]) * (a0[0] - a1[0]) / (b1[0] - a1[0])
        fore_aft = math.degrees(math.atan2(z_back - a0[2], abs(a1[1] - a0[1])))

        n, area = mesh.face_normals, mesh.area_faces
        cand = np.flatnonzero((n[:, 0] * inward > 0.3) & (np.abs(n[:, 1]) < 1e-6) & (n[:, 2] > 0.05))
        groups: dict[tuple, float] = {}
        for i in cand:
            key = tuple(np.round(n[i], 4))
            groups[key] = groups.get(key, 0.0) + float(area[i])
        wall = max(groups, key=groups.get)
        lean = math.degrees(math.asin(wall[2]))
        rows.append([{"left": "左", "right": "右"}[side], f"{front:.2f}°", f"{back:.2f}°",
                     f"{abs(fore_aft):.2f}°", f"{lean:.2f}°"])
        rep.expect(abs(front - p.tilt) < 0.02 and abs(back - p.tilt) < 0.02,
                   f"{side}_base の傾きが {front:.2f}°／{back:.2f}°")
        rep.expect(abs(fore_aft) < 0.02, f"{side}_base が前後に {fore_aft:.2f}° 傾いている")
        rep.expect(abs(lean - p.lean) < 0.02, f"{side}_base の内側の壁の寝かせが {lean:.2f}°")
    rep.table(["側", "手前の縁の傾き", "奥の縁の傾き", "前後の傾き", "内側の壁の寝かせ（垂直から）"], rows)


# ---- 4. 当たり ------------------------------------------------------------------

def clearances(rep: Report, p: Params) -> None:
    rep.add("## 部品どうしの当たり")
    rep.add()
    rep.add("組んだ位置での重なりの体積と、いちばん近い所のすき間。重なりが 0 なら食い込んでいない。")
    rep.add("キーキャップは印刷する場合の形（外形 "
            f"{p.cap:.1f} mm 角）で確かめた。すき間 0.00 は、そこで触れて載っていることを表す。")
    rep.add()
    rows = []
    for side in assembly.SIDES:
        jp = {"left": "左", "right": "右"}[side]
        rest = assembly.half(p, side, knurl=False)
        down = assembly.half(p, side, press=p.sw_travel, knurl=False)
        lay = assembly.layout(p, side)
        body_only = None      # スイッチのうち、穴に入る下ハウジングだけ
        for key in lay.keys:
            b = parts.place_on_key(p, key, geom.prism(geom.rect(0, 0, p.sw_body, p.sw_body), -p.sw_below, 0.0))
            body_only = b if body_only is None else body_only + b
        body_only = shell.to_world(p, lay, body_only)
        caps_rest = rest["caps"] + rest["voice_cap"] if "voice_cap" in rest else rest["caps"]
        caps_down = down["caps"] + down["voice_cap"] if "voice_cap" in down else down["caps"]
        pairs = [
            ("押していないとき", "傾斜台 と プレート", rest["base"] + rest["foot"], rest["plate"], 0.0),
            ("押していないとき", "傾斜台 と スイッチの下側", rest["base"] + rest["foot"], rest["switch_lower"], 0.5),
            ("押していないとき", "プレート と スイッチの下ハウジング", rest["plate"], body_only, 0.05),
            ("押していないとき", "プレート と キーキャップ", rest["plate"], caps_rest, 0.3),
            ("押し切ったとき", "プレート と キーキャップ", rest["plate"], caps_down, 0.3),
            ("押し切ったとき", "スイッチのハウジング と キーキャップ", rest["switch_upper"], caps_down, 0.1),
        ]
        if "wheel" in rest:
            pairs += [
                ("押していないとき", "プレート と ホイール", rest["plate"], rest["wheel"], 0.3),
                ("押していないとき", "傾斜台 と ホイール", rest["base"] + rest["foot"], rest["wheel"], 0.5),
                ("押し切ったとき", "キーキャップ と ホイール", caps_down, rest["wheel"], 0.5),
            ]
        if "ball" in rest:
            pairs += [
                ("押していないとき", "プレート と 球", rest["plate"], rest["ball"], 0.0),
                ("押していないとき", "傾斜台 と 球", rest["base"] + rest["foot"], rest["ball"], 0.5),
                ("押し切ったとき", "キーキャップ と 球", caps_down, rest["ball"], 0.5),
            ]
        for state, label, a, b, need in pairs:
            overlap = (a ^ b).volume()
            gap = a.min_gap(b, 10.0)
            rows.append([jp, state, label, f"{overlap:.3f}", f"{gap:.2f}"])
            rep.expect(overlap < 0.05, f"{side}: {label}（{state}）が {overlap:.3f} mm³ 重なっている")
            rep.expect(gap >= need - 0.011, f"{side}: {label}（{state}）のすき間が {gap:.2f} mm")
    rep.table(["側", "状態", "組み合わせ", "重なり（mm³）", "すき間（mm）"], rows)


# ---- 5. 印刷 --------------------------------------------------------------------

def printing(rep: Report, meshes: dict) -> None:
    rep.add("## 支えなしで印刷できるか")
    rep.add()
    rep.add("STL の向き（印刷する向き）のまま、下を向いた面を調べた。角度は垂直の壁を 0°、天井を 90° として測る。")
    rep.add("天井は、ひとつながりの面ごとに「面積 ÷ 縁の長さの半分」で幅を出した。細い張り出しなら張り出す量、")
    rep.add("橋なら渡す長さにほぼ等しい。")
    rep.add()
    rows = []
    for name in PRINTED:
        oh = analysis.overhang(meshes[name])
        rows.append([f"`{name}.stl`", f"{oh.steepest:.0f}°", f"{oh.over_limit:.0f}", f"{oh.ceiling:.0f}",
                     f"{oh.width:.1f}" if oh.ceiling > 0.5 else "なし"])
        rep.expect(oh.width <= 10.0, f"{name}.stl に幅 {oh.width:.1f} mm の天井がある")
        rep.expect(oh.over_limit <= 20.0, f"{name}.stl に 47.5° を超える斜面が {oh.over_limit:.0f} mm² ある")
    rep.table(["ファイル", "下向きの斜面の最大（天井を除く）", "47.5° を超える斜面（mm²）", "天井の面積（mm²）",
               "天井の幅の最大（mm）"], rows)


# ---- 6. 倒れにくさ --------------------------------------------------------------

def tipping(rep: Report, p: Params) -> None:
    rep.add("## 倒れにくさの目安")
    rep.add()
    rep.add("キーを、キー面に垂直に押したとする。押す力の向きを机まで延ばした点が足の内側にあれば、")
    rep.add("どれだけ強く押しても倒れない。外側にあるときは、足の縁を支点にして倒れ始める力を出した。")
    rep.add(f"重さは、印刷物を PLA（1.24 g/cm³）の体積に、スライスした重さとの比（プレート "
            f"{analysis.FILL['plate']}、傾斜台 {analysis.FILL['base']}）を掛け、スイッチを 1 個 "
            f"{analysis.SWITCH_G} g として見積もった。")
    rep.add()
    rows = []
    for side in assembly.SIDES:
        st = analysis.stability(p, side)
        jp = {"left": "左", "right": "右"}[side]
        for col in range(len(p.stagger)):
            names = [f"c{col}r{r}" for r in range(p.rows)]
            force = min(st.tip_force[n] for n in names)
            over = max(st.landing[n] for n in names)
            rows.append([jp, f"{col + 1} 列め", f"{over:.1f}" if over > 0 else "内側",
                         "倒れない" if math.isinf(force) else f"{force:.1f} N（約 {force / 9.81e-3:.0f} gf）"])
        for key in assembly.layout(p, side).keys:
            if key.kind == "alpha":
                continue
            force = st.tip_force[key.name]
            label = {"space": "親指キー（Space）", "enter": "親指キー（Enter）", "thumb2": "親指キー（外側）",
                     "voice": "音声入力キー"}[key.name]
            rows.append([jp, label, f"{st.landing[key.name]:.1f}" if st.landing[key.name] > 0 else "内側",
                         "倒れない" if math.isinf(force) else f"{force:.1f} N（約 {force / 9.81e-3:.0f} gf）"])
        worst = min(st.tip_force.values())
        rep.expect(worst > 4.0, f"{side}: {worst:.1f} N で倒れ始める")
        rep.add(f"{jp}：重さの見積もり {st.mass:.0f} g、重心は外側の端から {st.cg[0]:.1f} mm 内側、高さ {st.cg[2]:.1f} mm。")
    rep.add()
    rep.table(["側", "キー（列は小指側から数える）", "力の向きが足の外に出る量（mm）", "倒れ始める力"], rows)

    rep.add("内側の壁の寝かせ方を変えたとき（左、いちばん倒れやすい内側の列）：")
    rep.add()
    rows = []
    for lean in (9.0, 15.0, 20.0, 25.0, p.lean_nominal, 32.0):
        st = analysis.stability(p.with_(lean_nominal=lean), "left")
        worst = min(st.tip_force.values())
        rows.append([f"{lean:.0f}°" + ("（A-1）" if lean == p.lean_nominal else ""),
                     f"{st.foot_x[1] - st.foot_x[0]:.0f}", f"{worst:.1f} N"])
    rep.table(["内側の壁の寝かせ", "足の幅（mm）", "倒れ始める力"], rows)

    rep.add("傾きを変えたとき（内側の壁の寝かせは傾きと一緒に増える）：")
    rep.add()
    rows = []
    for t in (38.0, 43.0, 48.0, 50.0, 55.0):
        q = p.with_(tilt=t)
        st = analysis.stability(q, "left")
        bb = assembly.base_world(q, "left").bounding_box()
        rows.append([f"{t:.0f}°", f"{q.lean:.0f}°", f"{bb[3] - bb[0]:.0f} × {bb[4] - bb[1]:.0f} × {bb[5]:.0f}",
                     f"{min(st.tip_force.values()):.1f} N"])
    rep.table(["傾き", "内側の壁の寝かせ", "傾斜台の外寸（mm）", "倒れ始める力"], rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stl", type=Path, default=HERE / "stl")
    ap.add_argument("--report", type=Path, default=HERE / "out" / "check-report.md")
    args = ap.parse_args()

    p = DEFAULT
    rep = Report()
    meshes = shapes(rep, args.stl)
    assembled(rep, p)
    holes(rep, p, args.stl)
    tilt(rep, p, meshes)
    clearances(rep, p)
    printing(rep, meshes)
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
    text = "\n".join(rep.lines) + "\n"
    print(text)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text)
    sys.exit(1 if rep.failures else 0)


if __name__ == "__main__":
    main()
