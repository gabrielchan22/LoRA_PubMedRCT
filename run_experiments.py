"""
Top-level script: runs the full fine-tune baseline, then the LoRA ablation
sweep (rank x scaling convention), and assembles everything into one
comparison table.

Run this in Colab (needs a GPU and Hugging Face Hub access for the model
and dataset download).
"""

import torch
import pandas as pd

from data import build_dataloaders
from model import build_full_finetune_model, build_lora_model, MODEL_NAME
from train import train_and_evaluate

RESULTS = []


def run_baseline(device, train_loader, val_loader, test_loader, epochs=3, lr=2e-5):
    model = build_full_finetune_model()
    result = train_and_evaluate(
        model, train_loader, val_loader, test_loader, device,
        epochs=epochs, lr=lr, run_name="full_finetune", is_lora=False,
    )
    result.update({"method": "full_finetune", "rank": None, "alpha_mode": None})
    RESULTS.append(result)
    print("baseline done:", result)


def run_lora_sweep(
    device, train_loader, val_loader, test_loader,
    ranks=(4, 8, 16, 32, 64),
    alpha_modes=("alpha_over_r", "alpha_over_sqrt_r"),
    epochs=2,
    lr=1e-3,
):
    for r in ranks:
        for alpha_mode in alpha_modes:
            model = build_lora_model(r=r, alpha=2 * r, scaling_mode=alpha_mode)
            run_name = f"lora_r{r}_{alpha_mode}"
            result = train_and_evaluate(
                model, train_loader, val_loader, test_loader, device,
                epochs=epochs, lr=lr, run_name=run_name, is_lora=True,
            )
            result.update({"method": "lora", "rank": r, "alpha_mode": alpha_mode})
            RESULTS.append(result)
            print(f"{run_name} done:", result)


def build_results_table(output_path: str = "results.csv") -> pd.DataFrame:
    df = pd.DataFrame(RESULTS)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_loader, val_loader, test_loader = build_dataloaders(MODEL_NAME)

    run_baseline(device, train_loader, val_loader, test_loader)
    run_lora_sweep(device, train_loader, val_loader, test_loader)

    df = build_results_table()
    print(df)
