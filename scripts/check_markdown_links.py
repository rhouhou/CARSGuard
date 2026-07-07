from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse


LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def is_external_link(target: str) -> bool:
    parsed = urlparse(target)
    return parsed.scheme in {"http", "https", "mailto"}


def clean_target(target: str) -> str:
    target = target.strip()

    if " " in target and not target.startswith("<"):
        target = target.split(" ", maxsplit=1)[0]

    target = target.strip("<>")
    target = target.split("#", maxsplit=1)[0]
    return unquote(target)


def check_file(markdown_file: Path, repo_root: Path) -> list[str]:
    errors = []
    text = markdown_file.read_text(encoding="utf-8")

    for raw_target in LINK_PATTERN.findall(text):
        target = clean_target(raw_target)

        if not target:
            continue

        if is_external_link(target):
            continue

        candidate = (markdown_file.parent / target).resolve()

        if not candidate.exists():
            relative_file = markdown_file.relative_to(repo_root)
            errors.append(f"{relative_file}: missing link target -> {raw_target}")

    return errors


def main() -> int:
    repo_root = Path.cwd()
    markdown_files = sorted(repo_root.rglob("*.md"))

    markdown_files = [
        path
        for path in markdown_files
        if ".venv" not in path.parts
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
    ]

    errors = []

    for markdown_file in markdown_files:
        errors.extend(check_file(markdown_file, repo_root))

    if errors:
        print("Broken Markdown links found:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("All local Markdown links look valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())