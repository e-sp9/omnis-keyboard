"""Omnis A案ケース（版A-2）の寸法。

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
    # ---- 姿勢 ---------------------------------------------------------------
    # 参考画像のキー面は、奥と内側（親指側）が高い。左右だけでなく前後にも傾いている。
    tilt: float = 7.0           # 左右の傾き。キーの行の向きが水平から上がる角度。内側が高い
    slope: float = 18.0         # 前後の傾き。キーの列の向きが水平から上がる角度。奥が高い
    tilt_nominal: float = 7.0   # プレートの端面を決めた姿勢。この 2 つは変えない
    slope_nominal: float = 18.0
    cheek_lean: float = 7.0     # 内側の壁を垂直から寝かせる角度（上が外側へ）。傾きを増やすと、これより寝る
    plug_kappa: float = 15.0    # プレートの差し込み部の内側の面が、キー面に垂直な向きから寝る角度
    h_low: float = 12.0         # 外側の手前の角で、上面の縁が机から上がる高さ（最小）
    clear_floor: float = 1.4    # スイッチの下端と底の上面の間に残すすき間

    # ---- キー配置（Kailh Choc V2、17 mm ピッチ）--------------------------------
    pitch: float = 17.0
    cap: float = 16.5           # 1u のキーキャップの外形（17 mm ピッチ用）
    rows: int = 3
    # 列ごとの奥行きのずれ。小指、薬指、中指、人差し指、内側の人差し指
    stagger: tuple[float, ...] = (-14.0, -8.0, 0.0, -3.0, -2.0)

    # ---- Choc V2（図面 CPG135301D03）-----------------------------------------
    sw_body: float = 13.95      # プレートより下のハウジング
    sw_flange: float = 15.0     # プレートの上に載るつば
    sw_flange_t: float = 0.8    # つばの厚み（図面から読んだおおよその値。0.77 を切り上げた）
    sw_shoulder: float = 1.6    # プレート上面から、ハウジングがすぼまり始める高さ（同上）
    sw_crown: float = 12.4      # ハウジング上端の幅（同上）
    sw_top: float = 3.1         # プレート上面からハウジング上端
    sw_stem: float = 6.4        # プレート上面から軸の先端（押していないとき）
    sw_travel: float = 3.2
    sw_travel_tol: float = 0.25  # ストロークの公差（3.2±0.25）
    sw_height_tol: float = 0.10  # 高さの寸法の公差（図面の一般公差。5 mm 以下は ±0.10）
    sw_below: float = 2.2       # プレート上面からハウジングの底
    sw_post: float = 3.3        # 底から中央の柱の先まで
    sw_pin: float = 3.0         # 底から端子の先まで
    sw_clip: float = 1.65       # プレート上面（つばの下面）から、プレートに掛かる爪の肩まで
    sw_clip_w: float = 4.0      # 爪の幅（図から読んだおおよその値。向かい合う2面の中央に1本ずつ）
    sw_clip_out: float = 0.3    # 爪が下ハウジングの面から出る量（同上）

    # ---- プレート -------------------------------------------------------------
    hole: float = 14.1          # スイッチ穴（13.95 に 0.15 のすき間）
    # 穴のふちは、爪を受ける厚み clip_t の平らな板にする。その下は、穴の各辺の中央を
    # clip_w × clip_d だけ外へ広げ、爪が開く場所（逃げ）にする。爪の肩 sw_clip との差が、
    # スイッチの上下の遊びになる。逃げの天井は下に支えのない所へ渡す線なので、少し垂れる。
    # 垂れたぶん遊びは減る。0.2 mm 積層の層の境目に合わせて、0.2 刻みの値にする。
    clip_t: float = 1.4         # 爪を受ける板の厚み
    clip_alt: float = 1.2       # 試し刷りの奥の列の板の厚み（1.4 で爪が開ききらないときの代わり）
    clip_w: float = 6.0         # 爪の逃げの幅（爪の幅 4.0 に両側 1.0 の余裕）
    clip_d: float = 0.8         # 爪の逃げの奥行き（穴のふちから外へ。爪の出 0.3 に余裕を足した）
    bezel: float = 4.5          # スイッチ板の上面から、見えている上面までの高さ
    plate_t: float = 3.2        # スイッチ板の上面から、プレートの下面まで
    well_gap: float = 0.6       # キーキャップとくぼみの壁のすき間
    rim: float = 3.55           # くぼみの壁から外形まで（内側と奥の最小）

    # ---- 外形 -----------------------------------------------------------------
    margin_outer: float = 3.9   # 小指の列のキャップの外端から外形まで
    margin_inner: float = 4.75  # 内側の列のキャップの内端から外形まで
    back_cut_u: float = 34.0    # 奥の外側の角を斜めに落とす幅
    back_cut_v: float = 12.4    # 同じく奥行き
    chamfer_top: float = 1.5    # 上面と側面の間の面取りの帯
    chamfer_soft: float = 0.35  # 帯の上下の縁をさらに落とす量（縁を丸く見せる）
    chamfer_corner: float = 2.2  # 縦の角の面取り
    wall: float = 2.4           # 傾斜台の壁の厚み（0.4 mm ノズルで 6 本）
    well_chamfer: float = 0.4   # キーのくぼみの縁の面取り

    # ---- プレートと傾斜台の合わせ目 ---------------------------------------------
    seam: float = 2.2           # 上面から合わせ目までの深さ（キー面に垂直）。面取りの帯のすぐ下
    seat_land: float = 0.4      # 合わせ目の外周の平らな幅。そこから 45° で内へ下がる
    plug_gap: float = 0.25      # プレートの差し込み部と傾斜台の壁のすき間

    # ---- 足と底 ---------------------------------------------------------------
    # 底の台。壁の下端から 45° で foot_inset だけ内へ絞り、そこから foot_h だけまっすぐ下ろす。
    # 黒にするなら、壁の下端（高さ foot_h + foot_inset、底の板の上面と同じ）までを黒で印刷する。
    foot_h: float = 1.0         # 台のまっすぐな部分の高さ
    foot_inset: float = 1.2     # 台が壁から引っ込む量
    pad_d: float = 8.0          # 貼るゴム足の直径（平らなもの）
    pad_h: float = 2.0          # ゴム足の厚み。厚いほど倒れやすくなる
    pad_edge: float = 0.6       # ゴム足の縁から台の縁まで
    pad_mark_w: float = 0.8     # 貼る位置の目印（底に彫る輪の溝）の幅
    pad_mark_depth: float = 0.4
    cable_w: float = 13.0       # 奥の壁のケーブルの窓の幅
    cable_h: float = 7.0        # 同じく高さ
    cable_sill: float = 1.6     # 底の板の上面から窓の下端まで（壁の内側の、底の隅の斜面より上に開ける）

    # ---- 親指キー、音声入力キー --------------------------------------------------
    thumb_rot: float = 0.0      # 親指キーの回転。内側の端が手前に来る向きが負
    space_w: float = 1.75       # 左の親指キー（Space）の幅。u 単位
    enter_w: float = 1.25       # 右の親指キー（Enter）の幅
    voice_cap: float = 14.0     # 音声入力キーのキャップ
    voice_sink: float = 1.8     # 音声入力キーのスイッチを沈める量。残る板がちょうど clip_t になる深さ
    voice_u: float = -1.0       # 音声入力キーの中心

    # ---- 配置（版A-1で決めた位置。A-2 では動かさない）---------------------------
    v_front: float = -41.9                           # 手前の縁
    space_at: tuple[float, float] = (41.03, -21.1)   # 左の親指キー（Space）の中心
    enter_at: tuple[float, float] = (37.17, -21.1)   # 右の親指キー（Enter）の中心
    thumb2_at: tuple[float, float] = (18.05, -26.1)  # 右の外側の親指キーの中心
    ball_at: tuple[float, float] = (63.35, -25.2)    # 球の中心。左ではホイールのまわりのくぼみの中心
    wheel_u1: float = 74.7                           # ホイールの窓の内側の端

    # ---- スクロールホイール（左）------------------------------------------------
    wheel_d: float = 10.0       # ホイールの直径
    wheel_len: float = 14.4     # ホイールの長さ（軸の向き＝u）
    wheel_end: float = 2.5      # ホイールの両端の面取り
    wheel_rise: float = 2.0     # 上面から出る量
    wheel_pad: float = 0.5      # 窓の両端の壁で、軸の下だけを窓の中へ出した当たりの高さ
    wheel_pad_r: float = 1.5    # 当たりの幅の半分（ホイールの端面は、軸に近いここに触れて止まる）
    wheel_play: float = 0.2     # ホイールの端面と当たりのすき間（片側）
    axle_d: float = 1.75        # 軸。フィラメントの切れ端を使う
    lip_h: float = 0.0          # Space とホイールの間の仕切りの高さ（参考画像にないので付けない。付けるなら 2.2）
    lip_w: float = 2.0          # 仕切りの幅
    pocket_r: float = 2.5       # ホイールの窓の角の丸み（上から見て）
    pocket_ring: float = 1.0    # ホイールの窓のふちを斜めに落とす幅
    pocket_ring_depth: float = 0.7   # 同じく深さ

    # ---- トラックボール（右）----------------------------------------------------
    ball_d: float = 25.0
    ball_rise: float = 11.4     # 上面から出る量
    ball_gap: float = 0.6       # 球と穴のすき間
    ball_ring: float = 2.2      # 穴のふちをなだらかに落とす輪の幅
    ball_ring_depth: float = 1.3   # 同じく深さ（穴の壁で、上面からこれだけ下がる）
    ball_seat_deg: float = 60.0  # 受けの突起の位置。球の真下から測った角度

    # ---- キーキャップ（印刷する場合の形。プレビューもこの形）--------------------------
    cap_skirt: float = 4.6      # スイッチ板の上面から、キャップの下端（押していないとき）
    cap_margin: float = 0.25    # 公差の端のスイッチを押し切っても、キャップの下に残すすき間
    cap_top: float = 10.5       # 同じく、キャップの上面
    cap_taper: float = 1.7      # 上面が下端より片側ずつ小さくなる量
    cap_wall: float = 0.9
    cap_roof: float = 1.4
    cap_round: float = 1.6      # 角の丸み
    voice_cap_h: float = 2.1    # 音声入力キーのキャップの厚み（下端から上面まで）
    stem_post: float = 5.3      # 軸にかぶせる筒の外径（軸のまわりの壁の内径 5.5 より細く）
    stem_slot: tuple[float, float] = (4.15, 1.35)   # 十字の溝（軸は 4.00 × 1.30）
    stem_depth: float = 3.8

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    # ---- 導いた値 -------------------------------------------------------------
    @property
    def floor_top(self) -> float:
        """底の板の上面の高さ（机から）。壁の下端と同じ高さ。"""
        return self.foot_h + self.foot_inset

    @property
    def travel_max(self) -> float:
        """ストロークの上限。"""
        return self.sw_travel + self.sw_travel_tol

    @property
    def top(self) -> float:
        """見えている上面の高さ（w）。"""
        return self.bezel

    @property
    def bottom(self) -> float:
        """プレートの下面の高さ（w）。"""
        return -self.plate_t


DEFAULT = Params()
