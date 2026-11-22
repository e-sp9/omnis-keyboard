"""形から計算で確かめること。重心と倒れにくさ、印刷の向きでの張り出し。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import assembly, geom, parts, shell
from .params import Params

PLA = 1.24e-3        # g/mm³
SWITCH_G = 1.8       # Choc V2 1個のおおよその重さ
BALL_DENSITY = 1.35e-3
# 印刷物は中が粗く埋まるので、体積から出した重さより軽い。BambuStudio の見積もり
# （0.20mm Standard、壁 2 本、充填 15%）と体積の比から決めた。
FILL = {"plate": 0.67, "base": 0.87, "foot": 0.87}


def centroid(manifold) -> tuple[np.ndarray, float]:
    """重心と体積。"""
    mesh = manifold.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    f = np.asarray(mesh.tri_verts)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    return ((a + b + c) / 4.0 * vol[:, None]).sum(0) / vol.sum(), float(vol.sum())


# ---- 倒れにくさ -----------------------------------------------------------------

SUPPORTS = ("pads", "centres", "foot")


def _hull(points: np.ndarray) -> np.ndarray:
    """点の集まりを囲む凸多角形（反時計回り）。"""
    pts = sorted(map(tuple, np.round(points, 9)))

    def half(seq):
        out: list = []
        for q in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (q[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(q)
        return out[:-1]

    return np.array(half(pts) + half(reversed(pts)), dtype=float)


def support_polygon(p: Params, lay, support: str = "pads") -> tuple[np.ndarray, float]:
    """机に触れている範囲を囲む多角形と、ケースが机から浮く高さ。

    pads     平らなゴム足を目印の位置に貼ったとき。支点はゴム足の縁
    centres  半球形のゴム足など、中心だけで机に触れるものを同じ位置に貼ったとき
    foot     ゴム足を貼らず、足をじかに机に置いたとき。支点は足の縁
    """
    if support == "foot":
        return geom.polygons(shell.foot_outline(p, lay))[0], 0.0
    centres = np.array(shell.pad_centres(p, lay))
    if support == "centres":
        return _hull(centres), p.pad_h
    if support != "pads":
        raise ValueError(support)
    ring = np.array([[math.cos(a), math.sin(a)] for a in np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)])
    return _hull(np.concatenate([c + p.pad_d / 2.0 * ring for c in centres])), p.pad_h


@dataclass(frozen=True)
class Stability:
    mass: float                 # g
    cg: np.ndarray              # 机の座標（ケースの底から測った高さ）
    foot_x: tuple[float, float]  # 机に触れている範囲の、外側の端と内側の端
    tip_force: dict             # キーの名前 -> 倒れ始める力（N）。inf は倒れない
    landing: dict               # キーの名前 -> 押す力の向きが机に届く点が、支えの外に出る量（mm）


def stability(p: Params, side: str, support: str = "pads") -> Stability:
    """キーを、キー面に垂直に押したときの倒れにくさ。

    押す力の向きを机まで延ばした点が支えの内側なら、どれだけ強く押しても倒れない。
    外側なら、支えの縁を支点にして、力のモーメントが自重のモーメントを超えると倒れ始める。
    ゴム足を貼るとケースがその厚みだけ浮き、力の向きが机に届く点はさらに内側へずれる。
    """
    lay = assembly.layout(p, side)
    half = assembly.half(p, side, knurl=False)
    items = []
    for name, density in (("plate", PLA * FILL["plate"]), ("base", PLA * FILL["base"]),
                          ("foot", PLA * FILL["foot"]), ("caps", PLA),
                          ("voice_cap", PLA), ("wheel", PLA), ("ball", BALL_DENSITY)):
        if name in half:
            c, vol = centroid(half[name])
            items.append((vol * density, c))
    c, _ = centroid(half["switch_upper"])
    items.append((SWITCH_G * len(lay.keys), c))
    mass = sum(m for m, _ in items)
    cg = sum(m * c for m, c in items) / mass

    ground_poly, lift = support_polygon(p, lay, support)
    rot, trans = shell.frame(p, lay)
    push = rot @ np.array([0.0, 0.0, -1.0])
    tip, landing = {}, {}
    for key in lay.keys:
        _z0, z1 = parts.cap_heights(p, key)
        point = rot @ np.array([key.u, key.v, z1 - key.sink]) + trans
        ground = point + push * (-(point[2] + lift) / push[2])
        best, beyond = math.inf, 0.0
        for i in range(len(ground_poly)):
            a, b = ground_poly[i], ground_poly[(i + 1) % len(ground_poly)]
            edge = b - a
            out = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
            over = float((ground[:2] - a) @ out)
            if over > 1e-9:
                arm = float(-((cg[:2] - a) @ out))
                force = mass * 9.81e-3 * arm / (abs(push[2]) * over)
                if force < best:
                    best, beyond = force, over
        tip[key.name] = best
        landing[key.name] = beyond
    return Stability(mass, cg, (float(ground_poly[:, 0].min()), float(ground_poly[:, 0].max())), tip, landing)


# ---- 張り出し -------------------------------------------------------------------

@dataclass(frozen=True)
class Overhang:
    steepest: float        # 下を向いた斜面の、垂直からの角度の最大（度）。天井は含めない
    over_limit: float      # limit を超える斜面の面積（mm²）。天井は含めない
    ceiling: float         # 天井（85° 以上）の面積（mm²）
    bridge: float          # 両端に支えのある天井で、支えから支えまでの間隔の最大（mm）
    cantilever: float      # 片側にしか支えのない天井で、支えからの張り出しの最大（mm）


def _span(tris: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray, held: np.ndarray) -> tuple[float, float]:
    """天井ひとつぶんの、橋の長さと張り出し。

    tris は天井の三角形（上から見た頂点）、seg_a と seg_b は天井の縁の線分の両端、
    held は、その線分の下に壁があって支えになるかどうか。
    三角形の中心を通る直線を、向きを変えながら引く。両側とも支えのある縁に届く向きが
    あれば、そこは橋として渡せる。その点での橋の長さは、届く向きのうちいちばん短い長さ。
    どの向きでも片側が支えのない縁に出るなら、そこは張り出しで、支えのある縁までの距離を測る。
    """
    if not held.any():
        return 0.0, math.inf           # どこにも支えがない
    edge = seg_b - seg_a
    # 5° 刻みの向きと、縁に平行な向き
    along = np.degrees(np.arctan2(edge[:, 1], edge[:, 0])) % 180.0
    angles = np.radians(np.unique(np.round(np.concatenate([np.arange(0.0, 180.0, 5.0), along]) * 2.0) / 2.0))
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    def reach(pts: np.ndarray) -> float:
        """点の集まりから、支えのある縁までの距離の最大。"""
        a, e = seg_a[held], edge[held]
        t = np.einsum("pki,ki->pk", pts[:, None, :] - a[None], e) / np.einsum("ki,ki->k", e, e)
        near = a[None] + np.clip(t, 0.0, 1.0)[..., None] * e[None]
        return float(np.linalg.norm(pts[:, None, :] - near, axis=2).min(axis=1).max())

    bridge = cantilever = 0.0
    for corners in tris:
        pt = corners.mean(axis=0)
        rel = seg_a - pt
        best = math.inf
        for d in dirs:
            denom = d[0] * edge[:, 1] - d[1] * edge[:, 0]
            ok = np.abs(denom) > 1e-12
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (rel[:, 0] * edge[:, 1] - rel[:, 1] * edge[:, 0]) / denom     # 直線に沿った位置
                s = (rel[:, 0] * d[1] - rel[:, 1] * d[0]) / denom                 # 線分の中での位置
            hit = ok & (s >= -1e-9) & (s <= 1.0 + 1e-9)
            fwd, back = hit & (t > 1e-9), hit & (t < -1e-9)
            if not fwd.any() or not back.any():
                continue
            t_f = np.where(fwd, t, np.inf)
            t_b = np.where(back, -t, np.inf)
            i, j = int(t_f.argmin()), int(t_b.argmin())
            # 同じ距離に支えのない縁もあれば（角を通る直線）、支えがないものとして扱う
            free_f = fwd & ~held & (t_f <= t_f[i] + 1e-6)
            free_b = back & ~held & (t_b <= t_b[j] + 1e-6)
            if held[i] and held[j] and not free_f.any() and not free_b.any():
                best = min(best, float(t_f[i] + t_b[j]))
        if math.isfinite(best):
            bridge = max(bridge, best)
        else:
            # 張り出しがいちばん大きくなるのは縁の上なので、三角形の頂点でも測る
            cantilever = max(cantilever, reach(np.vstack([corners, pt[None]])))
    return bridge, cantilever


def overhang(mesh, limit: float = 47.5, min_area: float = 0.02) -> Overhang:
    """印刷する向きのメッシュ（trimesh）で、下に支えがない面を調べる。

    天井（ほぼ水平で下を向いた面）は、ひとつながりの面ごとに縁を調べる。縁の先の面が下へ
    続いていれば、そこには下から立ち上がる壁があり、天井の支えになる。上へ続いていれば、
    天井はそこで終わっていて支えがない。
    """
    normals = mesh.face_normals
    areas = mesh.area_faces
    tri = mesh.triangles
    z_min = float(mesh.bounds[0][2])
    on_bed = tri[:, :, 2].max(axis=1) < z_min + 0.01
    down = (-normals[:, 2]).clip(0.0, 1.0)
    angle = np.degrees(np.arcsin(down))
    faces = (~on_bed) & (areas > min_area) & (angle > 0.5)
    slope_all = faces & (angle < 85.0)
    steepest = float(angle[slope_all].max()) if slope_all.any() else 0.0
    slope = slope_all & (angle > limit)
    roof = (~on_bed) & (angle >= 85.0)

    bridge = cantilever = 0.0
    if roof.any():
        idx = np.flatnonzero(roof)
        parent = {int(i): int(i) for i in idx}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b in mesh.face_adjacency:
            a, b = int(a), int(b)
            if a in parent and b in parent:
                parent[find(a)] = find(b)
        groups: dict[int, list[int]] = {}
        for i in parent:
            groups.setdefault(find(i), []).append(i)

        # 縁の先の面を引けるよう、辺 -> その辺を持つ面 の表を作る
        owners: dict[tuple[int, int], list[int]] = {}
        for fi, face in enumerate(mesh.faces):
            for k in range(3):
                e = (int(face[k]), int(face[(k + 1) % 3]))
                owners.setdefault((min(e), max(e)), []).append(fi)

        verts = mesh.vertices
        for members in groups.values():
            if float(areas[members].sum()) < 0.5:
                continue
            inside = set(members)
            seg_a, seg_b, held = [], [], []
            for fi in members:
                face = mesh.faces[fi]
                for k in range(3):
                    e = (int(face[k]), int(face[(k + 1) % 3]))
                    others = [o for o in owners[(min(e), max(e))] if o not in inside]
                    if not others:
                        continue
                    third = [int(v) for v in mesh.faces[others[0]] if int(v) not in e][0]
                    z_edge = min(verts[e[0]][2], verts[e[1]][2])
                    seg_a.append(verts[e[0]][:2])
                    seg_b.append(verts[e[1]][:2])
                    held.append(bool(verts[third][2] < z_edge - 1e-6))
            if not seg_a:
                continue
            b, c = _span(tri[members][:, :, :2], np.array(seg_a), np.array(seg_b), np.array(held))
            bridge, cantilever = max(bridge, b), max(cantilever, c)
    return Overhang(steepest, float(areas[slope].sum()), float(areas[roof].sum()), bridge, cantilever)
