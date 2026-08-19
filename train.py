"""
One shared training loop used by BOTH the full fine-tune baseline and
every LoRA run — same loop, same eval code, only the trainable parameter
set (and learning rate) differs between calls.
"""

import os
import torch
from tqdm.auto import tqdm
from metrics import (
    TrainingRunTracker,
    compute_classification_metrics,
    get_checkpoint_size_mb,
    count_trainable_parameters,
)


def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def evaluate(model, dataloader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(batch["labels"].cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_classification_metrics(all_preds, all_labels)
    model.train()
    return metrics


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs: int = 3,
    lr: float = 2e-5,
    run_name: str = "run",
    checkpoint_dir: str = "checkpoints",
    is_lora: bool = False,
    verbose: bool = True,
    use_amp: bool = True,
) -> dict:
    model.to(device)
    optimizer = torch.optim.AdamW(get_trainable_params(model), lr=lr)

    total_steps = max(1, len(train_loader) * epochs)
    try:
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
    except ImportError:
        scheduler = None

    # Mixed precision only makes sense (and is only supported) on CUDA.
    amp_enabled = use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    tracker = TrainingRunTracker()
    tracker.start()

    model.train()
    for epoch in range(epochs):
        progress = tqdm(
            train_loader,
            desc=f"[{run_name}] epoch {epoch + 1}/{epochs}",
            leave=False,
            disable=not verbose,
        )
        running_loss = 0.0
        for step, batch in enumerate(progress, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            if step % 20 == 0:
                progress.set_postfix(avg_loss=f"{running_loss / step:.4f}")

        val_metrics = evaluate(model, val_loader, device)
        if verbose:
            print(
                f"[{run_name}] epoch {epoch + 1}/{epochs} "
                f"val_accuracy={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

    tracker.stop()
    test_metrics = evaluate(model, test_loader, device)

    os.makedirs(checkpoint_dir, exist_ok=True)
    if is_lora:
        ckpt_path = os.path.join(checkpoint_dir, f"{run_name}.pt")
        adapter_state = {
            k: v
            for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k or "classifier" in k
        }
        torch.save(adapter_state, ckpt_path)
    else:
        ckpt_path = os.path.join(checkpoint_dir, run_name)
        model.save_pretrained(ckpt_path)

    result = {
        **test_metrics,
        **tracker.summary(),
        "checkpoint_size_mb": get_checkpoint_size_mb(ckpt_path),
        **count_trainable_parameters(model),
    }
    return result
