"""BambuStudio の CLI でスライスし、印刷時間とフィラメントの重さを見積もる（任意）。

    .venv/bin/python estimate.py             # stl/ の部品をすべて
    .venv/bin/python estimate.py base        # 名前に base を含む部品だけ
    .venv/bin/python estimate.py --tilt 20 --slope 0   # 傾斜台を left_base_tilt20_slope0.stl などに替えて

スライスするだけで、プリンターには何も送らない。BambuStudio の設定フォルダは読むだけにし、
CLI が書くもの（3MF、設定の写し、ログ）は使い捨ての置き場の中に閉じ込める（--datadir）。
置き場は環境変数 OMNIS_SCRATCH、なければ ~/.cache/omnis-case-a。リポジトリの中には書かない。
BambuStudio が入っていない環境では「見積もりなし」と出して終わる。
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

APP = Path("/Applications/BambuStudio.app/Contents/MacOS/BambuStudio")
PRESETS = Path.home() / "Library/Application Support/BambuStudio/system/BBL"

# この Mac の BambuStudio で選ばれていたプリセット
MACHINE = "Bambu Lab A1 mini 0.4 nozzle"
PROCESS = "0.20mm Standard @BBL A1M"
FILAMENT = "Bambu PLA Basic @BBL A1M"

PARTS = ["left_plate", "right_plate", "left_base", "right_base", "left_wheel",
         "fit_test", "keycap_1u", "keycap_space", "keycap_enter", "keycap_voice"]


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
    out = work / f"{stl.stem}.3mf"
    out.unlink(missing_ok=True)
    cmd = [str(APP), "--datadir", str(work / "datadir"),
           "--load-settings", f"{settings['machine']};{settings['process']}",
           "--load-filaments", str(settings["filament"]),
           "--orient", "0", "--arrange", "1", "--slice", "0",
           "--outputdir", str(work), "--export-3mf", out.name, str(stl)]
    # CLI は実行した場所に result.json を書くので、作業用のフォルダで動かす
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=work)
    if run.returncode != 0 or not out.exists():
        tail = " / ".join((run.stdout + run.stderr).strip().splitlines()[-3:])
        return {"error": f"終了コード {run.returncode}: {tail}"}
    with zipfile.ZipFile(out) as z:
        info = z.read("Metadata/slice_info.config").decode()

    def grab(pattern: str):
        m = re.search(pattern, info)
        return m.group(1) if m else None

    return {"seconds": int(grab(r'key="prediction" value="(\d+)"')),
            "grams": float(grab(r'key="weight" value="([\d.]+)"')),
            "meters": float(grab(r'used_m="([\d.]+)"') or 0.0),
            "support": grab(r'key="support_used" value="(\w+)"') or "?"}


def clock(seconds: int) -> str:
    h, rest = divmod(int(round(seconds / 60.0)), 60)
    return f"{h}時間{rest:02d}分" if h else f"{rest}分"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("only", nargs="?", default="", help="名前にこの文字を含む部品だけ")
    ap.add_argument("--stl", type=Path, default=HERE / "stl")
    ap.add_argument("--tilt", type=float, help="この左右の傾きで出した傾斜台（left_base_tilt20_slope0.stl など）を見積もる")
    ap.add_argument("--slope", type=float, help="この前後の傾きで出した傾斜台を見積もる")
    ap.add_argument("--work", type=Path, help="CLI が書くものの置き場（既定は使い捨ての置き場の estimate/）")
    ap.add_argument("--json", type=Path, help="結果を書くファイル（既定は使い捨ての置き場の estimate.json）")
    args = ap.parse_args()
    args.stl = args.stl.expanduser()
    tag = manifest.posture_tag(args.tilt, args.slope)
    args.work = args.work or workdir.scratch("estimate")
    args.json = args.json or workdir.scratch(f"estimate{tag}.json")

    if not APP.exists() or not PRESETS.exists():
        print("見積もりなし（BambuStudio か、そのプリセットが見つからない）")
        sys.exit(0)
    args.work.mkdir(parents=True, exist_ok=True)
    settings = {}
    for kind, name in (("machine", MACHINE), ("process", PROCESS), ("filament", FILAMENT)):
        settings[kind] = args.work / f"{kind}.json"
        settings[kind].write_text(json.dumps(resolve(kind, name), ensure_ascii=False, indent=1))

    version = subprocess.run([str(APP), "--help"], capture_output=True, text=True, timeout=60,
                             cwd=args.work).stdout
    version = (re.search(r"BambuStudio-([\d.]+)", version) or [None, "?"])[1]
    print(f"BambuStudio {version} / {MACHINE} / {PROCESS} / {FILAMENT}")
    print()
    print("| ファイル | 印刷時間 | 重さ（g） | フィラメント（m） | サポート |")
    print("| --- | --- | --- | --- | --- |")
    results, seconds, grams = {}, 0, 0.0
    for name in PARTS:
        if name.endswith("_base"):
            name += tag
        if args.only and args.only not in name:
            continue
        r = slice_stl(args.stl / f"{name}.stl", args.work, settings)
        results[name] = r
        if "error" in r:
            print(f"| `{name}.stl` | 見積もりなし（{r['error']}） | | | |")
            continue
        seconds += 60 * round(r["seconds"] / 60.0)     # 表に出す値どうしで合計が合うように丸めてから足す
        grams += round(r["grams"], 1)
        support = {"false": "なし", "true": "あり"}.get(r["support"], r["support"])
        print(f"| `{name}.stl` | {clock(r['seconds'])} | {r['grams']:.1f} | {r['meters']:.2f} | {support} |")
    print(f"| 合計 | {clock(seconds)} | {grams:.1f} | | |")
    if not args.only:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"bambustudio": version, "machine": MACHINE, "process": PROCESS,
                                         "filament": FILAMENT, "parts": results},
                                        ensure_ascii=False, indent=1) + "\n")
        print(f"\n結果を書いた場所: {args.json}")


if __name__ == "__main__":
    main()
