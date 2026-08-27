"""
Model construction — shared by the full fine-tune and LoRA experiments.

NOTE: like data.py, this needs huggingface.co access to actually download
roberta-base.
"""

from transformers import AutoModelForSequenceClassification
from lora_layers import inject_lora

MODEL_NAME = "roberta-base"  # or "microsoft/deberta-v3-base"
NUM_LABELS = 5


def build_full_finetune_model():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)


def build_lora_model(
    r: int = 8,
    alpha: int = 16,
    target_modules=("query", "value"),
    scaling_mode: str = "alpha_over_r",
):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    for p in model.parameters():
        p.requires_grad = False

    inject_lora(
        model,
        target_modules=target_modules,
        r=r,
        alpha=alpha,
        scaling_mode=scaling_mode,
    )

    # RobertaForSequenceClassification (and BERT-style equivalents) expose
    # the classification head as `model.classifier`. It has no pretrained
    # weights to preserve, so it stays fully trainable regardless of LoRA.
    # If you swap MODEL_NAME to a different architecture family, confirm
    # this attribute name still holds via print(model) first.
    for p in model.classifier.parameters():
        p.requires_grad = True

    return model


if __name__ == "__main__":
    model = build_lora_model(r=8, alpha=16)
    print(model)
