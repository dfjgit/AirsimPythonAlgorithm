from __future__ import annotations

from pathlib import Path

import generate_two_stage_stage_docs_v3 as source


source.DOCX_PATH = source.OUTPUT_DIR / "TwoStage_Stage1_Stage2_Algorithms_v4.docx"
source.TEXT_PATH = source.OUTPUT_DIR / "TwoStage_Stage1_Stage2_Algorithms_v4.txt"


def main() -> None:
    source.main()


if __name__ == "__main__":
    main()
