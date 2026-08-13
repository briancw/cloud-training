#!/usr/bin/env python3
"""Run an already-provisioned AI Toolkit image inside a Hugging Face Job."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def main() -> int:
    token = required_env("HF_TOKEN")
    config_path = Path(required_env("AI_TOOLKIT_CONFIG"))
    output_root = Path("/mnt/outputs")
    if not config_path.is_file():
        raise RuntimeError(f"AI Toolkit config mount is unavailable: {config_path}")
    if not output_root.is_dir():
        raise RuntimeError(f"output bucket mount is unavailable: {output_root}")

    # Preserve the exact config that produced the output alongside checkpoints.
    shutil.copy2(config_path, output_root / "ai-toolkit-config.yaml")
    subprocess.run(["python", "run.py", str(config_path)], check=True)

    repo_id = required_env("MODEL_REPO")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=output_root,
        commit_message="Successful AI Toolkit HF Job",
    )
    print(f"Published successful training artifacts to https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
