"""使い捨ての出力（スライスした 3MF、描画用の一時ファイル、検査の表）の置き場。

リポジトリの中には書かない。環境変数 OMNIS_SCRATCH があればその場所、なければ
~/.cache/omnis-case-a を使う。
"""

from __future__ import annotations

import os
from pathlib import Path


def scratch(*parts: str) -> Path:
    root = os.environ.get("OMNIS_SCRATCH")
    base = Path(root).expanduser() if root else Path.home() / ".cache" / "omnis-case-a"
    return base.joinpath(*parts)
