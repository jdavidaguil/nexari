
import os, json, argparse, numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--training_dir", default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    parser.add_argument("--output_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = parser.parse_args()

    meta = json.loads((Path(args.training_dir) / "metadata.json").read_text())
    text_col = meta["text_col"]
    label2id = meta["label2id"]
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    num_labels = meta["num_labels"]
    metric_name = meta.get("metric", "f1")

    train_ds = load_from_disk(str(Path(args.training_dir) / "train"))
    eval_ds = load_from_disk(str(Path(args.training_dir) / "test"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    def tokenize(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=128)

    keep = ["labels", "input_ids", "attention_mask", "token_type_ids"]
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=[c for c in train_ds.column_names if c not in keep])
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=[c for c in eval_ds.column_names if c not in keep])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=num_labels,
        id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True,
    )

    metric = evaluate.load("f1" if metric_name == "f1" else "accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        if metric_name == "f1":
            return metric.compute(predictions=preds, references=labels, average="weighted")
        return metric.compute(predictions=preds, references=labels)

    TrainingArguments_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=False,
        no_cuda=True,
        report_to="none",
    )
    Trainer(
        model=model,
        args=TrainingArguments(**TrainingArguments_kwargs),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    ).train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
