from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from falcon.conversion.config import ConversionTrainingConfig
from falcon.conversion.data import build_dense_eval_grid


def build_optimizer(model, config):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for optimizer.")
    return optim.Adam(trainable_params, lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)


def build_scheduler(optimizer, config):
    return optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler.gamma)


def evaluate_dense_grid(model, config):
    x_values, y_values = build_dense_eval_grid(config)
    device = torch.device(config.trainer.device)
    with torch.no_grad():
        predictions = model(torch.from_numpy(x_values).to(device=device)).cpu().numpy()
    mse = float(((predictions - y_values) ** 2).mean())
    mae = float(abs(predictions - y_values).mean())
    return {"x": x_values, "target": y_values, "prediction": predictions, "mse": mse, "mae": mae}


def train_conversion_model(model, train_loader, config: ConversionTrainingConfig) -> dict[str, Any]:
    criterion = nn.MSELoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    best_loss = float("inf")
    best_path = config.output_dir / config.checkpoint.dir / "best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    device = torch.device(config.trainer.device)

    for epoch in range(config.trainer.max_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            num_batches += 1

        epoch_loss = running_loss / max(num_batches, 1)
        history.append({"epoch": float(epoch + 1), "loss": epoch_loss})

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            if config.checkpoint.save_best:
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1, "best_loss": best_loss}, best_path)

        if (epoch + 1) % config.scheduler.step_every == 0:
            scheduler.step()
        if (epoch + 1) % config.trainer.log_every == 0:
            print(f"  Epoch {epoch + 1}/{config.trainer.max_epochs}, loss={epoch_loss:.6f}")

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])

    eval_metrics = evaluate_dense_grid(model, config)
    metrics = {"best_loss": best_loss, "final_mse": eval_metrics["mse"], "final_mae": eval_metrics["mae"], "epochs": config.trainer.max_epochs}

    metrics_path = config.output_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {"history": history, "metrics": metrics, "eval": eval_metrics}
