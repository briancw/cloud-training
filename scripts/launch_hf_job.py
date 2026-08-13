#!/usr/bin/env python3
"""Launch a containerized AI Toolkit Hugging Face Job from a TOML sidecar.

The image contains a pinned AI Toolkit installation. The local AI Toolkit YAML
is mounted unchanged and the HF token is sent to the Job as an encrypted secret.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs"
CONTAINER_ENTRYPOINT = "/opt/job/run_ai_toolkit_container.py"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def assert_bucket_access(bucket: str) -> None:
    try:
        HfApi(token=os.environ["HF_TOKEN"]).bucket_info(bucket)
    except Exception as error:
        raise RuntimeError(
            f"Cannot access HF Storage Bucket {bucket!r}. Check the namespace, "
            "that the bucket exists, and that HF_TOKEN has write access."
        ) from error


def require_string(table: dict, key: str, config: Path) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{config}: [{key}] must be a non-empty string")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to a job TOML config")
    parser.add_argument("--dry-run", action="store_true", help="Print the command only")
    args = parser.parse_args()

    config_path = args.config.resolve()
    try:
        config = tomllib.loads(config_path.read_text())
        job = config["job"]
        data = config["data"]
        ai_toolkit = config["ai_toolkit"]
        image = config["image"]
        outputs = config["outputs"]
        publish = config["publish"]
    except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError) as error:
        parser.error(f"invalid config: {error}")

    try:
        name = require_string(job, "name", config_path)
        flavor = require_string(job, "flavor", config_path)
        timeout = require_string(job, "timeout", config_path)
        bucket = require_string(data, "bucket", config_path)
        remote_path = require_string(data, "remote_path", config_path).strip("/")
        mount_path = require_string(data, "mount_path", config_path)
        output_path = require_string(outputs, "remote_path", config_path).strip("/")
        output_mount = require_string(outputs, "mount_path", config_path)
        ai_config = (ROOT / require_string(ai_toolkit, "config", config_path)).resolve()
        image_ref = require_string(image, "reference", config_path)
        model_repo = require_string(publish, "model_repo", config_path)
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    if not ai_config.is_file():
        parser.error(f"AI Toolkit config does not exist: {ai_config}")
    try:
        ai_config_relative_path = ai_config.relative_to(CONFIGS_DIR)
    except ValueError:
        parser.error(f"AI Toolkit config must live under {CONFIGS_DIR}: {ai_config}")
    if not mount_path.startswith("/"):
        parser.error("[data].mount_path must be absolute")
    if not output_mount.startswith("/"):
        parser.error("[outputs].mount_path must be absolute")

    load_dotenv(ROOT / ".env")
    if not os.environ.get("HF_TOKEN"):
        parser.error("HF_TOKEN is required; put it in .env or export it")
    # Prefer the CLI installed alongside the interpreter running this script.
    # That makes `./venv/bin/python scripts/launch_hf_job.py ...` work without
    # requiring the caller to activate the virtual environment first.
    hf_executable = shutil.which("hf")
    venv_hf = Path(sys.executable).parent / "hf"
    if hf_executable is None and venv_hf.is_file():
        hf_executable = str(venv_hf)
    if not args.dry_run and hf_executable is None:
        parser.error("`hf` was not found; install requirements.txt first")
    try:
        assert_bucket_access(bucket)
    except RuntimeError as error:
        parser.error(str(error))

    command = [
        hf_executable or "hf", "jobs", "run",
        "--name", name,
        "--flavor", flavor,
        "--timeout", timeout,
        "--secrets", "HF_TOKEN",
        "--volume", f"hf://buckets/{bucket}/{remote_path}:{mount_path}:ro",
        "--volume", f"hf://buckets/{bucket}/{output_path}:{output_mount}",
        # Local config files are uploaded and mounted read-only by HF Jobs. The
        # YAML remains the exact AI Toolkit config we run remotely.
        "--volume", f"{CONFIGS_DIR}:/mnt/config:ro",
        "--env", f"AI_TOOLKIT_CONFIG=/mnt/config/{ai_config_relative_path.as_posix()}",
        "--env", f"MODEL_REPO={model_repo}",
    ]
    for key, value in config.get("env", {}).items():
        if not isinstance(value, (str, int, float, bool)):
            parser.error(f"[env].{key} must be a scalar value")
        command.extend(["--env", f"{key}={value}"])
    if job.get("detach", False):
        command.append("--detach")
    command.extend([image_ref, "python", CONTAINER_ENTRYPOINT])

    # Do not print environment values or tokens.  The command only refers to the
    # secret by name, and HF stores it encrypted for the remote job.
    print("Launching:", " ".join(command))
    if args.dry_run:
        return 0
    # Jobs may sync local volumes with Xet before submission. Keep that staging
    # cache in the project rather than assuming ~/.cache is writable (common in
    # sandboxed shells and containers). These variables affect only the local
    # submission process; they are not passed into the remote Job.
    cache_dir = ROOT / ".hf-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    command_env = os.environ.copy()
    command_env.setdefault("HF_HOME", str(cache_dir))
    command_env.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    command_env.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
    command_env.setdefault("XDG_CACHE_HOME", str(cache_dir))
    subprocess.run(command, check=True, env=command_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
