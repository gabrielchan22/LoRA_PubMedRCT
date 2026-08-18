"""
Data loading & preprocessing — PubMed-RCT (20k) sentence classification.

Dataset: armanc/pubmed-rct20k on the HF Hub (177k train / 29.7k validation
/ 29.6k test rows). Columns: abstract_id, label (str, 5 classes), text,
sentence_id.

NOTE: this file requires network access to huggingface.co to actually run
(to download the dataset and tokenizer). It's written and structurally
correct but untested-by-execution in this sandbox, which can't reach HF's
servers — run it in Colab to verify end to end.
"""

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

LABELS = ["background", "objective", "methods", "results", "conclusions"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def build_dataloaders(model_name: str, max_length: int = 128, batch_size: int = 32, num_workers: int = 2):
    raw = load_dataset("armanc/pubmed-rct20k")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_batch(examples):
        enc = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        enc["labels"] = [LABEL2ID[label] for label in examples["label"]]
        return enc

    tokenized = raw.map(
        tokenize_batch,
        batched=True,
        remove_columns=raw["train"].column_names,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(
        tokenized["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        tokenized["validation"], batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        tokenized["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders("roberta-base")
    batch = next(iter(train_loader))
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("labels:", batch["labels"].shape)
    print("label distribution in batch:", batch["labels"].bincount())
