"""Omnis A案ケース（版A-1）の寸法。

長さは mm、角度は度。値を変えたら build.py → check.py → render.py の順に出し直す。
座標は「キー面の座標」(u, v, w) で書く。

    u : 行の向き。外側（小指側）から内側（親指側）へ。小指の列の中心が 0。
    v : 列の向き。手前から奥へ。中指の列の手前のキーの中心が 0。
    w : キー面に垂直。スイッチが載る面（スイッチ板の上面）が 0、上が正。

左右とも「内側が +u」の向きで作り、右手は最後に左右を反転する。
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Params:
    # ---- 傾き ---------------------------------------------------------------
    tilt: float = 43.0          # キー面の傾き。内側（親指側）が高い。前後には傾けない
    tilt_nominal: float = 43.0  # プレートの端面を決めた角度。これは変えない
    lean_nominal: float = 28.0  # tilt_nominal のとき、内側の壁が垂直から寝る角度
    h_outer: float = 9.5        # 外側の端で、キー面の縁が机から上がる高さ（最小）

    # ---- キー配置（Kailh Choc V2、17 mm ピッチ）--------------------------------
    pitch: float = 17.0
    cap: float = 16.5           # 1u のキーキャップの外形（17 mm ピッチ用）
    rows: int = 3
    # 列ごとの奥行きのずれ。小指、薬指、中指、人差し指、内側の人差し指
    stagger: tuple[float, ...] = (-14.0, -8.0, 0.0, -3.0, -2.0)

    # ---- Choc V2（図面 CPG135301D03）-----------------------------------------
    sw_body: float = 13.95      # プレートより下のハウジング
    sw_flange: float = 15.0     # プレートの上に載るつば
    sw_flange_t: float = 0.75   # つばの厚み（図面から読んだおおよその値）
    sw_shoulder: float = 1.6    # プレート上面から、ハウジングがすぼまり始める高さ（同上）
    sw_crown: float = 12.4      # ハウジング上端の幅（同上）
    sw_top: float = 3.1         # プレート上面からハウジング上端
    sw_stem: float = 6.4        # プレート上面から軸の先端（押していないとき）
    sw_travel: float = 3.2
    sw_below: float = 2.2       # プレート上面からハウジングの底
    sw_post: float = 3.3        # 底から中央の柱の先まで
    sw_pin: float = 3.0         # 底から端子の先まで

    # ---- プレート -------------------------------------------------------------
    hole: float = 14.1          # スイッチ穴（13.95 に 0.15 のすき間）
    land: float = 0.8           # 穴のふちで爪が掛かる板の厚み
    flare: float = 0.95         # ふちの下を 45° で広げる量（支えなしで印刷するため）
    bezel: float = 4.5          # スイッチ板の上面から、見えている上面までの高さ
    plate_t: float = 3.2        # スイッチ板の上面から、プレートの下面まで
    well_gap: float = 0.6       # キーキャップとくぼみの壁のすき間
    rim: float = 3.55           # くぼみの壁から外形まで（内側と奥の最小）

    # ---- 外形 -----------------------------------------------------------------
    margin_outer: float = 3.75  # 小指の列のキャップの外端から外形まで
    margin_inner: float = 4.75  # 内側の列のキャップの内端から外形まで
    back_cut_u: float = 34.0    # 奥の外側の角を斜めに落とす幅
    back_cut_v: float = 12.4    # 同じく奥行き
    chamfer_top: float = 1.2    # 上面のまわりの面取り
    chamfer_corner: float = 2.0  # 縦の角の面取り
    wall: float = 2.4           # 傾斜台の壁の厚み（0.4 mm ノズルで 6 本）

    # ---- プレートと傾斜台の合わせ目 ---------------------------------------------
    seam: float = 2.0           # 上面から合わせ目までの深さ（キー面に垂直）
    seat_land: float = 0.4      # 合わせ目の外周の平らな幅。そこから 45° で内へ下がる
    plug_gap: float = 0.25      # プレートの差し込み部と傾斜台の壁のすき間

    # ---- 足と底 ---------------------------------------------------------------
    foot_h: float = 2.0         # 足の高さ（黒にするならここまでを黒で印刷）
    foot_inset: float = 1.6     # 足が外形から引っ込む量（45° の面取りでつなぐ）
    foot_step: float = 0.4      # 面取りの下端からさらに引っ込む量
    floor_w: float = 9.0        # 底の枠の幅（ゴム足を貼る面）

    # ---- 親指キー、音声入力キー --------------------------------------------------
    thumb_rot: float = 0.0      # 親指キーの回転。内側の端が手前に来る向きが負
    space_w: float = 1.75       # 左の親指キー（Space）の幅。u 単位
    enter_w: float = 1.25       # 右の親指キー（Enter）の幅
    thumb_drop: float = 1.6     # 親指キーと、その奥の文字キーのキャップのすき間
    voice_cap: float = 14.0     # 音声入力キーのキャップ
    voice_sink: float = 2.0     # 音声入力キーのスイッチを沈める量
    voice_u: float = -1.0       # 音声入力キーの中心

    # ---- スクロールホイール（左）------------------------------------------------
    wheel_d: float = 10.0       # ホイールの直径
    wheel_len: float = 15.0     # ホイールの長さ（軸の向き＝u）
    wheel_end: float = 2.5      # ホイールの両端の面取り
    wheel_rise: float = 2.0     # 上面から出る量
    wheel_gap: float = 0.4      # ホイールと窓のすき間
    axle_d: float = 1.75        # 軸。フィラメントの切れ端を使う
    lip_h: float = 2.2          # Space とホイールの間の仕切りの高さ
    lip_w: float = 2.0          # 仕切りの幅

    # ---- トラックボール（右）----------------------------------------------------
    ball_d: float = 25.0
    ball_rise: float = 11.4     # 上面から出る量
    ball_gap: float = 0.6       # 球と穴のすき間
    ball_ring: float = 1.5      # 穴のまわりの面取りの幅
    ball_seat_deg: float = 60.0  # 受けの突起の位置。球の真下から測った角度

    # ---- キーキャップ（印刷する場合の形。プレビューもこの形）--------------------------
    cap_skirt: float = 4.3      # スイッチ板の上面から、キャップの下端（押していないとき）
    cap_top: float = 10.5       # 同じく、キャップの上面
    cap_taper: float = 1.7      # 上面が下端より片側ずつ小さくなる量
    cap_wall: float = 0.9
    cap_roof: float = 1.4
    cap_round: float = 1.6      # 角の丸み
    voice_cap_h: float = 2.45   # 音声入力キーのキャップの高さ
    stem_post: float = 5.3      # 軸にかぶせる筒の外径（軸のまわりの壁の内径 5.5 より細く）
    stem_slot: tuple[float, float] = (4.15, 1.35)   # 十字の溝（軸は 4.00 × 1.30）
    stem_depth: float = 3.8

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    # ---- 導いた値 -------------------------------------------------------------
    @property
    def lean(self) -> float:
        """いまの tilt のとき、内側の壁が垂直から寝る角度。

        プレートの内側の端面と壁をひと続きの面にするので、キー面と壁のなす角は
        tilt によらず一定。tilt を増やすと壁もそのぶん寝る。
        """
        return self.lean_nominal + (self.tilt - self.tilt_nominal)

    @property
    def top(self) -> float:
        """見えている上面の高さ（w）。"""
        return self.bezel

    @property
    def bottom(self) -> float:
        """プレートの下面の高さ（w）。"""
        return -self.plate_t


DEFAULT = Params()
