"""
nexari.pipeline.trainer
────────────────────────
Step 4: Fine-tune the selected backbone on the chosen dataset.

Backends:
  local      — HF Trainer on Codespaces CPU (default, free)
  sagemaker  — AWS SageMaker HuggingFace estimator (GPU, ~$0.15/run)

Returns a local path to the saved model ready for deployment.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from rich.console import Console

from nexari.agent.interpreter import TaskDefinition
from nexari.agent.selector import BackboneSelection
from nexari.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_MAX_TRAIN_SAMPLES,
)

console = Console()


class TrainBackend(str, Enum):
    LOCAL = "local"
    SAGEMAKER = "sagemaker"
    # VERTEX = "vertex"      # future
    # AZURE_ML = "azure_ml"  # future


def train(
    task: TaskDefinition,
    dataset_id: str,
    backbone: BackboneSelection,
    output_dir: str | None = None,
    backend: TrainBackend | None = None,
) -> str:
    """
    Fine-tune backbone on dataset. Returns path to saved model directory.
    Backend defaults to NEXARI_TRAIN_BACKEND env var, fallback to local.
    """
    if backend is None:
        backend = TrainBackend(os.getenv("NEXARI_TRAIN_BACKEND", "local"))

    console.print(f"  [dim]Backend: {backend.value}[/]")

    if backend == TrainBackend.LOCAL:
        return _train_local(task, dataset_id, backbone, output_dir)
    elif backend == TrainBackend.SAGEMAKER:
        return _train_sagemaker(task, dataset_id, backbone, output_dir)
    else:
        raise ValueError(f"Unknown training backend: {backend}")


# ── Local Backend ─────────────────────────────────────────────────────────────

def _train_local(
    task: TaskDefinition,
    dataset_id: str,
    backbone: BackboneSelection,
    output_dir: str | None = None,
) -> str:
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    import numpy as np
    import evaluate

    output_path = Path(output_dir or f"./models/{backbone.model_id.replace('/', '_')}")
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(f"  [dim]Loading dataset {dataset_id}...[/]")
    raw = load_dataset(dataset_id)
    split = raw["train"] if "train" in raw else list(raw.values())[0]
    if len(split) > DEFAULT_MAX_TRAIN_SAMPLES:
        split = split.select(range(DEFAULT_MAX_TRAIN_SAMPLES))
        console.print(f"  [dim]Capped to {DEFAULT_MAX_TRAIN_SAMPLES} samples[/]")

    text_col = _detect_text_column(split.column_names)
    label_col = _detect_label_column(split.column_names)
    console.print(f"  [dim]Text: {text_col} | Label: {label_col}[/]")

    from datasets import ClassLabel
    if not isinstance(split.features[label_col], ClassLabel):
        unique_labels = sorted(set(str(x) for x in split[label_col]))
        label2id = {l: i for i, l in enumerate(unique_labels)}
        id2label = {i: l for l, i in label2id.items()}
        split = split.map(lambda x: {"labels": label2id[str(x[label_col])]})
    else:
        label2id = {split.features[label_col].names[i]: i for i in range(split.features[label_col].num_classes)}
        id2label = {i: l for l, i in label2id.items()}
        split = split.map(lambda x: {"labels": x[label_col]})

    num_labels = len(label2id)
    console.print(f"  [dim]{num_labels} labels: {list(label2id.keys())[:5]}[/]")

    split = split.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    tokenizer = AutoTokenizer.from_pretrained(backbone.tokenizer_id)
    def tokenize(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=128)

    keep = ["labels", "input_ids", "attention_mask", "token_type_ids"]
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=[c for c in train_ds.column_names if c not in keep])
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=[c for c in eval_ds.column_names if c not in keep])

    model = AutoModelForSequenceClassification.from_pretrained(
        backbone.model_id, num_labels=num_labels,
        id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True,
    )

    metric = evaluate.load("f1" if task.suggested_metric == "f1" else "accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        if task.suggested_metric == "f1":
            return metric.compute(predictions=preds, references=labels, average="weighted")
        return metric.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=DEFAULT_EPOCHS,
        per_device_train_batch_size=DEFAULT_BATCH_SIZE,
        per_device_eval_batch_size=DEFAULT_BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=task.suggested_metric,
        logging_steps=50,
        report_to="none",
        fp16=False,
        no_cuda=True,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    console.print(f"  [dim]Training {DEFAULT_EPOCHS} epoch(s) on {len(train_ds)} samples...[/]")
    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    _save_metadata(output_path, task, dataset_id, backbone, label2id, id2label, text_col)

    eval_results = trainer.evaluate()
    score = eval_results.get(f"eval_{task.suggested_metric}", "n/a")
    if isinstance(score, float):
        console.print(f"  [green]✓[/] {task.suggested_metric}: {score:.4f}")

    return str(output_path)


# ── SageMaker Backend ─────────────────────────────────────────────────────────

def _train_sagemaker(
    task: TaskDefinition,
    dataset_id: str,
    backbone: BackboneSelection,
    output_dir: str | None = None,
) -> str:
    import json
    import tarfile
    import tempfile
    import boto3
    from datasets import load_dataset

    role_arn = os.getenv("SAGEMAKER_ROLE_ARN")
    s3_bucket = os.getenv("NEXARI_S3_BUCKET")
    region = os.getenv("AWS_REGION", "us-east-1")

    if not role_arn:
        raise ValueError("SAGEMAKER_ROLE_ARN not set.")
    if not s3_bucket:
        raise ValueError("NEXARI_S3_BUCKET not set.")

    try:
        import sagemaker
        from sagemaker.pytorch import PyTorch
    except ImportError:
        raise ImportError("SageMaker backend requires: pip install sagemaker")

    console.print(f"  [dim]Preparing dataset for SageMaker...[/]")

    raw = load_dataset(dataset_id)
    split = raw["train"] if "train" in raw else list(raw.values())[0]
    if len(split) > DEFAULT_MAX_TRAIN_SAMPLES:
        split = split.select(range(DEFAULT_MAX_TRAIN_SAMPLES))

    text_col = _detect_text_column(split.column_names)
    label_col = _detect_label_column(split.column_names)

    from datasets import ClassLabel
    if not isinstance(split.features[label_col], ClassLabel):
        unique_labels = sorted(set(str(x) for x in split[label_col]))
        label2id = {l: i for i, l in enumerate(unique_labels)}
        id2label = {i: l for l, i in label2id.items()}
        split = split.map(lambda x: {"labels": label2id[str(x[label_col])]})
    else:
        label2id = {split.features[label_col].names[i]: i for i in range(split.features[label_col].num_classes)}
        id2label = {i: l for l, i in label2id.items()}
        split = split.map(lambda x: {"labels": x[label_col]})

    split = split.train_test_split(test_size=0.1, seed=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        split["train"].save_to_disk(str(tmppath / "train"))
        split["test"].save_to_disk(str(tmppath / "test"))

        meta = {
            "text_col": text_col,
            "model_id": backbone.model_id,
            "tokenizer_id": backbone.tokenizer_id,
            "num_labels": len(label2id),
            "label2id": label2id,
            "id2label": id2label,
            "metric": task.suggested_metric,
            "epochs": DEFAULT_EPOCHS,
            "batch_size": DEFAULT_BATCH_SIZE,
        }
        (tmppath / "metadata.json").write_text(json.dumps(meta))

        s3 = boto3.client("s3", region_name=region)
        s3_prefix = f"nexari/training/{task.domain.replace(' ', '-')}"
        for f in tmppath.rglob("*"):
            if f.is_file():
                key = f"{s3_prefix}/data/{f.relative_to(tmppath)}"
                s3.upload_file(str(f), s3_bucket, key)

        data_s3_uri = f"s3://{s3_bucket}/{s3_prefix}/data"
        console.print(f"  [dim]Dataset uploaded to {data_s3_uri}[/]")

    repo_root = Path(__file__).resolve().parents[2]

    console.print(f"  [dim]Submitting SageMaker job (spot g4dn.xlarge)...[/]")
    sess = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    estimator = PyTorch(
        entry_point="nexari/pipeline/scripts/sagemaker_train.py",
        source_dir=str(repo_root),
        dependencies=[],
        role=role_arn,
        
        framework_version="2.1",
        py_version="py310",
        instance_type="ml.c5.2xlarge",
        instance_count=1,
        use_spot_instances=False,
        
        max_run=1800,
        sagemaker_session=sess,
        environment={"HF_TOKEN": os.getenv("HF_TOKEN", "")},
        hyperparameters={
            "model_id": backbone.model_id,
            "epochs": DEFAULT_EPOCHS,
            "batch_size": DEFAULT_BATCH_SIZE,
        },
        output_path=f"s3://{s3_bucket}/{s3_prefix}/output",
    )

    estimator.fit({"training": data_s3_uri}, wait=True, logs="Training")
    console.print(f"  [green]✓[/] SageMaker job complete")

    output_path = Path(output_dir or f"./models/{backbone.model_id.replace('/', '_')}")
    output_path.mkdir(parents=True, exist_ok=True)

    model_uri = estimator.model_data
    console.print(f"  [dim]Downloading model from {model_uri}...[/]")
    bucket, key = model_uri.replace("s3://", "").split("/", 1)
    s3 = boto3.client("s3", region_name=region)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        s3.download_file(bucket, key, tmp.name)
        with tarfile.open(tmp.name, "r:gz") as tar:
            tar.extractall(str(output_path))

    _save_metadata(output_path, task, dataset_id, backbone, label2id, id2label, text_col)
    console.print(f"  [green]✓[/] Model saved to {output_path}")
    return str(output_path)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _save_metadata(output_path, task, dataset_id, backbone, label2id, id2label, text_col):
    import json
    metadata = {
        "task_type": task.task_type.value,
        "domain": task.domain,
        "dataset_id": dataset_id,
        "backbone": backbone.model_id,
        "num_labels": len(label2id),
        "label2id": label2id,
        "id2label": id2label,
        "metric": task.suggested_metric,
        "text_column": text_col,
    }
    (Path(output_path) / "nexari_metadata.json").write_text(json.dumps(metadata, indent=2))


def _detect_text_column(columns: list[str]) -> str:
    preferred = ["text", "content", "message", "description", "ticket", "body", "input", "sentence"]
    for p in preferred:
        for c in columns:
            if p in c.lower():
                return c
    return next((c for c in columns if c not in ("label", "labels", "id", "idx")), columns[0])


def _detect_label_column(columns: list[str]) -> str:
    preferred = ["label", "labels", "category", "urgency", "priority", "intent", "class", "target"]
    for p in preferred:
        for c in columns:
            if p in c.lower():
                return c
    return columns[-1]