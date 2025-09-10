## Overview

This repository trains a reinforcement-learning agent that generates high-quality job offer descriptions in XML. It fine-tunes a base model (e.g., `Qwen/Qwen3-0.6B`) using the [ART](https://github.com/openpipe/art) reinforcement learning framework and learns from reward signals computed by an automated judge.

The agent runs on a GPU cluster provisioned via [SkyPilot](https://skypilot.readthedocs.io/). Training data ("job contexts") is loaded from an S3 bucket and includes a job title, target language, and optional skills for each sample.

### The Task

Given a job title, a target language (e.g., `en`, `fr`), and optional skills, the agent must generate a complete job offer in valid XML following a fixed structure: `<job_offer>`, `<title>`, `<overview>`, `<responsibilities>`, `<skills>`, `<nice_to_have>`.

The reward is a weighted average of five criteria computed by an automated judge:
- Language consistency with the requested language
- XML validity and presence of required tags
- Inclusion of provided skills (with de-duplication penalty)
- Relevance of any new skills added by the model
- Skill completeness (penalizes missing essential skills)

## Getting Started

### 1. Install dependencies

If you haven't already, install `uv` by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

Then install the project dependencies by running `uv sync`.

### 2. Install SkyPilot/RunPod

We'll be using `SkyPilotBackend` to manage the GPU that your model will be trained on. In order for the backend to work, you'll need to have SkyPilot installed on your machine and provide it with the credentials to spin up machines on at least one infra provider.

We recommend using RunPod because of their ease of use, but any infra provider that SkyPilot [supports](https://docs.skypilot.co/en/latest/overview.html#bringing-your-infra) will work.

Follow RunPod's **Getting Started** guide [here](https://docs.runpod.io/integrations/skypilot/). You'll have to provide a credit card to use RunPod, but you'll only pay for the time your GPUs are running.

### 3. Configure environment variables

Create a `.env` file at the root with the following:

- Azure OpenAI judge (required)
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_ENDPOINT` (e.g., `https://<your-resource>.openai.azure.com/`)
  - `AZURE_DEPLOYMENT_NAME` (e.g., `gpt-4o-mini` or your deployed chat model)

- AWS for S3 dataset and optional model backups (required for training data)
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_REGION`
  
  The code expects a dataset at: `s3://job-offer-generation/datasets/job_offer_dataset.json`.
  You can change this by editing `bucket_name` and `file_key` in `src/job-offer/load_documents.py`.

- Optional
  - `OPENPIPE_API_KEY` to log completions/metrics to OpenPipe

### 4. Run the training script

```bash
uv run python src/job-offer/train_with_setup.py
```

The first run will:

- Provision a cluster (default `L4:1` GPU) and install dependencies via a custom setup script
- Register the model with ART (`Qwen/Qwen3-0.6B` by default)
- Load job contexts from S3
- Train over batched rollouts, logging metrics and pushing improved checkpoints to S3

Notes:
- To change cluster resources, edit `resources = sky.Resources(...)` in `src/job-offer/train_with_setup.py`.
- Subsequent runs can use `src/job-offer/train.py` once the cluster is set up.

### 5. Shutting down the cluster

When you're done training and running benchmarks, you can shut down the cluster in two ways:

Through the CLI:

```bash
uv run sky down <cluster-name>
```

or through code:

```python
DESTROY_AFTER_RUN = True

if DESTROY_AFTER_RUN:
    await backend.down()
```

However, since spinning up clusters is a time-intensive process, we recommend keeping clusters alive until you're sure you won't be using them in the near future.

### Evaluation and Observability

- Reward shaping and metrics are implemented in `src/job-offer/rollout.py`.
- Judge calls are made via Azure OpenAI in `src/job-offer/get_judge_completion.py`.
- If `OPENPIPE_API_KEY` is set, completions and metrics are reported to OpenPipe for observability.

Project structure highlights:
- `src/job-offer/train_with_setup.py`: training entrypoint (provisions cluster with dependencies)
- `src/job-offer/train.py`: training entrypoint (assumes cluster dependencies already installed)
- `src/job-offer/custom_backend.py`: custom SkyPilot backend that installs dependencies on the cluster
- `src/job-offer/rollout.py`: rollout logic, scoring, and reward computation
- `src/job-offer/load_documents.py`: loads job contexts from S3
- `src/job-offer/get_judge_completion.py`: Azure OpenAI async judge client
- `setup_cluster.sh`: alternative setup script containing the dependency list
