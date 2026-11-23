"""BambuStudio の CLI でスライスし、印刷時間とフィラメントの重さを見積もる（任意）。

    .venv/bin/python estimate.py             # stl/ の一式をすべて
    .venv/bin/python estimate.py base        # 名前に base を含む部品だけ
    .venv/bin/python estimate.py --tilt 20 --slope 0   # 傾斜台を left_base_tilt20_slope0.stl などに替えて
    .venv/bin/python estimate.py --stl ~/omnis-hole142  # 別の場所に出した一式

スライスするのは、STL と同じ場所の build.json（build.py が書く記録）に登録された一式の部品
（10 個）だけ。check.py と同じ決まりで一式を選ぶ。一式の部品のどれかに記録がない、別の寸法や
別の版で作ってある、build.py が書いたあとで中身が変わっている、のどれかなら、そう知らせて
止まる。同じフォルダーにそれ以外の STL があっても、読まない（止まりもしない）。
出力の「一式の指紋」が check.py のものと同じなら、検査した一式と同じものを見積もっている。

スライスするだけで、プリンターには何も送らない。BambuStudio の設定フォルダは読むだけにし、
CLI が書くもの（3MF、設定の写し、ログ）は使い捨ての置き場の中に閉じ込める（--datadir）。
置き場は環境変数 OMNIS_SCRATCH、なければ ~/.cache/omnis-case-a。リポジトリの中には書かない。

終了コード：すべて見積もれたら 0。部品の読み込みやスライスにひとつでも失敗したら 1。
BambuStudio が入っていない環境では「見積もりなし」と出して 0 で終わる（何も試していないため）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from omnis_a import manifest, workdir  # noqa: E402
from omnis_a.params import MODEL  # noqa: E402

APP = Path("/Applications/BambuStudio.app/Contents/MacOS/BambuStudio")
PRESETS = Path.home() / "Library/Application Support/BambuStudio/system/BBL"

# この Mac の BambuStudio で選ばれていたプリセット
MACHINE = "Bambu Lab A1 mini 0.4 nozzle"
PROCESS = "0.20mm Standard @BBL A1M"
FILAMENT = "Bambu PLA Basic @BBL A1M"

# 表に出す順（README の部品の番号と同じ）
ORDER = ("left_plate", "right_plate", "left_base", "right_base", "left_wheel",
         "fit_test", "keycap_1u", "keycap_space", "keycap_enter", "keycap_voice")


def resolve(kind: str, name: str) -> dict:
    """プリセットを、継承元（inherits）と取り込み（include）までたどって 1 つにまとめる。"""
    data = json.loads((PRESETS / kind / f"{name}.json").read_text())
    merged: dict = {}
    if data.get("inherits"):
        merged.update(resolve(kind, data["inherits"]))
    for extra in data.get("include", []):
        merged.update(resolve(kind, extra))
    merged.update(data)
    merged.pop("inherits", None)
    merged.pop("include", None)
    return merged


def slice_stl(stl: Path, work: Path, settings: dict) -> dict:
    """STL を 1 個スライスして、時間と重さを読む。できなければ {"error": 理由} を返す。"""
    out = work / f"{stl.stem}.3mf"
    out.unlink(missing_ok=True)
    cmd = [str(APP), "--datadir", str(work / "datadir"),
           "--load-settings", f"{settings['machine']};{settings['process']}",
           "--load-filaments", str(settings["filament"]),
           "--orient", "0", "--arrange", "1", "--slice", "0",
           "--outputdir", str(work), "--export-3mf", out.name, str(stl)]
    # CLI は実行した場所に result.json を書くので、作業用のフォルダで動かす。STL と作業用の
    # フォルダは絶対パスで渡してあるので、どこから呼んでも同じファイルを読む。
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=work)
    except subprocess.TimeoutExpired:
        return {"error": "20 分たっても終わらなかった"}
    if run.returncode != 0 or not out.exists():
        tail = " / ".join((run.stdout + run.stderr).strip().splitlines()[-3:])
        return {"error": f"終了コード {run.returncode}: {tail}"}
    try:
        with zipfile.ZipFile(out) as z:
            info = z.read("Metadata/slice_info.config").decode()
    except (zipfile.BadZipFile, KeyError) as err:
        return {"error": f"3MF を読めなかった（{err}）"}

    def grab(pattern: str):
        m = re.search(pattern, info)
        return m.group(1) if m else None

    seconds, grams = grab(r'key="prediction" value="(\d+)"'), grab(r'key="weight" value="([\d.]+)"')
    if seconds is None or grams is None:
        return {"error": "3MF に時間か重さの記録がなかった"}
    return {"seconds": int(seconds), "grams": float(grams),
            "meters": float(grab(r'used_m="([\d.]+)"') or 0.0),
            "support": grab(r'key="support_used" value="(\w+)"') or "?"}


def clock(seconds: int) -> str:
    h, rest = divmod(int(round(seconds / 60.0)), 60)
    return f"{h}時間{rest:02d}分" if h else f"{rest}分"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("only", nargs="?", default="", help="名前にこの文字を含む部品だけ")
    ap.add_argument("--stl", type=Path, default=HERE / "stl", help="STL と build.json のある場所")
    ap.add_argument("--tilt", type=float, help="この左右の傾きで出した傾斜台（left_base_tilt20_slope0.stl など）を見積もる")
    ap.add_argument("--slope", type=float, help="この前後の傾きで出した傾斜台を見積もる")
    ap.add_argument("--work", type=Path, help="CLI が書くものの置き場（既定は使い捨ての置き場の estimate/）")
    ap.add_argument("--json", type=Path, help="結果を書くファイル（既定は使い捨ての置き場の estimate.json）")
    args = ap.parse_args()
    # CLI は作業用のフォルダで動かすので、相対パスはここで絶対パスにしておく
    stl = args.stl.expanduser().resolve()
    tag = manifest.posture_tag(args.tilt, args.slope)
    work = (args.work or workdir.scratch("estimate")).expanduser().resolve()
    out_json = (args.json or workdir.scratch(f"estimate{tag}.json")).expanduser().resolve()

    # 登録された一式だけを選ぶ。古い記録、別の版、寸法の食い違い、中身が変わった STL は、ここで知らせて止まる
    _p_plate, p, files = manifest.load_intact_set(stl, args.tilt, args.slope)
    todo = [name for name in ORDER if not args.only or args.only in files[name]]
    if not todo:
        raise SystemExit(f"名前に「{args.only}」を含む部品はありません。あるのは: {'、'.join(files[n] for n in ORDER)}")

    if not APP.exists() or not PRESETS.exists():
        print("見積もりなし（BambuStudio か、そのプリセットが見つからない）")
        sys.exit(0)
    work.mkdir(parents=True, exist_ok=True)
    settings = {}
    for kind, name in (("machine", MACHINE), ("process", PROCESS), ("filament", FILAMENT)):
        settings[kind] = work / f"{kind}.json"
        settings[kind].write_text(json.dumps(resolve(kind, name), ensure_ascii=False, indent=1))

    version = subprocess.run([str(APP), "--help"], capture_output=True, text=True, timeout=60, cwd=work).stdout
    version = (re.search(r"BambuStudio-([\d.]+)", version) or [None, "?"])[1]
    mark = manifest.fingerprint(stl, files)
    print(f"BambuStudio {version} / {MACHINE} / {PROCESS} / {FILAMENT}")
    print(f"見積もる一式: `{stl}`（モデルの版 {MODEL}、姿勢は左右 {p.tilt:g}°、前後 {p.slope:g}°、一式の指紋 `{mark}`）")
    print()
    print("| ファイル | 印刷時間 | 重さ（g） | フィラメント（m） | サポート |")
    print("| --- | --- | --- | --- | --- |")
    results, failed, seconds, grams = {}, [], 0, 0.0
    for name in todo:
        file = files[name]
        r = slice_stl(stl / file, work, settings)
        results[file] = r
        if "error" in r:
            failed.append(file)
            print(f"| `{file}` | 見積もりなし（{r['error']}） | | | |")
            continue
        seconds += 60 * round(r["seconds"] / 60.0)     # 表に出す値どうしで合計が合うように丸めてから足す
        grams += round(r["grams"], 1)
        support = {"false": "なし", "true": "あり"}.get(r["support"], r["support"])
        print(f"| `{file}` | {clock(r['seconds'])} | {r['grams']:.1f} | {r['meters']:.2f} | {support} |")
    total = "合計" if not failed else f"合計（見積もれた {len(todo) - len(failed)} 個ぶん）"
    print(f"| {total} | {clock(seconds)} | {grams:.1f} | | |")
    if not args.only:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"bambustudio": version, "machine": MACHINE, "process": PROCESS,
                                        "filament": FILAMENT, "model": MODEL, "fingerprint": mark,
                                        "parts": results}, ensure_ascii=False, indent=1) + "\n")
        print(f"\n結果を書いた場所: {out_json}")
    if failed:
        print(f"\n見積もれなかった部品がある: {'、'.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
