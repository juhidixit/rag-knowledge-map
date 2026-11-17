"""Deploy the demo to a Hugging Face Space (Gradio SDK on free ZeroGPU hardware).

Builds a staging folder from space/ plus src/ragmap, fills the paper attributions into the Space
README, creates the Space if needed and uploads the folder. Needs HF_TOKEN (a write token).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
HARDWARE = "zero-a10g"  # the only free hardware for Gradio Spaces on personal accounts


def stage(dest: Path) -> None:
    space = ROOT / "space"
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(space / name, dest / name)
    shutil.copytree(space / "library", dest / "library", ignore=shutil.ignore_patterns(".gitkeep"))
    shutil.copytree(ROOT / "src" / "ragmap", dest / "src" / "ragmap", ignore=shutil.ignore_patterns("__pycache__"))
    meta = space / "papers.json"
    papers = json.loads(meta.read_text()) if meta.exists() else []
    if papers:
        shutil.copy2(meta, dest / "papers.json")
    lines = "\n".join(f"- {p['attribution']}" for p in papers) or "- None. Upload your own PDF."
    readme = dest / "README.md"
    readme.write_text(readme.read_text().replace("<!-- PAPERS -->", lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="juhidixit/ragmap-demo")
    ap.add_argument("--dry-run", action="store_true", help="stage only, print the file list")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        stage(dest)
        files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
        print(f"staged {len(files)} files")
        if args.dry_run:
            print("\n".join(files))
            return

        api = HfApi(token=os.environ["HF_TOKEN"])
        # Free personal accounts may only create Gradio Spaces on ZeroGPU, so ask for it at creation.
        api.create_repo(
            args.space, repo_type="space", space_sdk="gradio", space_hardware=HARDWARE, private=False, exist_ok=True
        )
        runtime = api.get_space_runtime(args.space)
        if runtime.requested_hardware != HARDWARE and runtime.hardware != HARDWARE:
            # Raises if the account may not use ZeroGPU; the workflow then fails with that message.
            api.request_space_hardware(args.space, HARDWARE)
        commit = api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=dest,
            commit_message=f"Deploy from GitHub {os.environ.get('GITHUB_SHA', 'local')[:7]}",
            delete_patterns=["src/**", "library/**", "*.py", "*.json"],
        )
        print(f"uploaded: {commit.commit_url}")
        print(f"space: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
