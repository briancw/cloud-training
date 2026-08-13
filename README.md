# Ephemeral Hugging Face training jobs

This repository keeps three concerns separate:

- `training-data/` is local source data and is ignored by Git.
- The Hugging Face Storage Bucket `brianw/training-data` holds mutable datasets,
  checkpoints, and logs.
- `configs/` holds the actual AI Toolkit YAMLs; edit these directly for local or
  cloud training.
- `jobs/` holds only Hugging Face Jobs infrastructure settings: hardware,
  timeout, container image, bucket prefixes, and final model destination.
- `scripts/run_ai_toolkit_container.py` is a small runner mounted into each Job.
  It starts the YAML in AI Toolkit's official container, preserves the exact
  YAML with the outputs, and publishes successful artifacts to a model repo.

## Install and authenticate

```bash
python -m pip install -r requirements.txt
```

Put `HF_TOKEN=...` in `.env`. The scripts load it locally and never print it.
The token needs access to `brianw/training-data`; launching Jobs also requires
an HF account eligible for Jobs with available credit. No Docker build, GHCR
package, or GitHub token is required.

## First run

Review what would upload, then sync the dataset. The normal sync is additive:
it never deletes remote files unless `--delete` is explicitly passed.

```bash
python scripts/push_training_data.py \
  training-data/96yottea --dry-run

python scripts/push_training_data.py \
  training-data/96yottea
```

## Run an AI Toolkit job

The current test run has two files:

- `configs/flux2-klein-9b-500.yaml` — the normal AI Toolkit YAML. This is the
  source of truth for model, data paths, optimizer, batch size, steps, and every
  training-specific option. You can load and adjust it locally in AI Toolkit.
- `jobs/flux2-klein-9b-500.toml` — the cloud-only settings.

The launcher mounts the local `configs/` directory and small runner read-only
into the Job. It runs the referenced YAML unchanged in the official
`ostris/aitoolkit:latest` image, copies that exact YAML to the output bucket,
and publishes it with successful model artifacts. AI Toolkit and its
Python/CUDA dependencies are already in the image, so the Job does not clone
the toolkit or resolve packages at startup.

Before launching, accept the FLUX.2 model gate with the same HF account/token.
Then review and launch it:

```bash
python scripts/launch_hf_job.py jobs/flux2-klein-9b-500.toml --dry-run
python scripts/launch_hf_job.py jobs/flux2-klein-9b-500.toml
```

The short run uses batch size 2, 500 steps, and `prodigy8bit` on an A100 80GB.
AI Toolkit coerces Prodigy LRs below `0.1` to `1.0`, so `lr: 1.0` is the effective
value; a requested `0.001` would be silently changed by the trainer anyway.

`configs/krea2-500.yaml` and `jobs/krea2-500.toml` provide the equivalent Krea
2 Raw test. It uses AI Toolkit's `convrot8` (int8 ConvRot) model quantization
and enables `torch.compile`:

```bash
python scripts/launch_hf_job.py jobs/krea2-500.toml --dry-run
python scripts/launch_hf_job.py jobs/krea2-500.toml
```

Accept the Krea-2-Raw model gate before launching. `torch.compile` has a
one-time startup cost, so this short job is principally a compatibility test.

For a new run, copy both files, update paths inside the AI Toolkit YAML to use
`/mnt/training-data` and `/mnt/outputs`, then point the new job TOML's
`[ai_toolkit].config` field at it. The default image tracks the official AI
Toolkit `latest` image; replace `[image].reference` with an upstream fixed tag
or digest if you later need exact reproducibility.
