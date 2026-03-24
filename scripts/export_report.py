#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from carsguard.core.config import load_project_configs
from carsguard.reports.serializers import report_to_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a JSON report into a text report.")
    parser.add_argument("report_json", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output-txt", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_project_configs(args.config_dir)
    default_output_dir = Path(cfg["paths"]["reports_output_dir"])

    with args.report_json.open("r", encoding="utf-8") as f:
        report = json.load(f)

    text = report_to_text(report)

    if args.output_txt is None:
        print(text)
    else:
        output_txt = args.output_txt
        if not output_txt.is_absolute():
            output_txt = default_output_dir / output_txt
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        output_txt.write_text(text, encoding="utf-8")
        print(f"Saved text report to {output_txt}")


if __name__ == "__main__":
    main()