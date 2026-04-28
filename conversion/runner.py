from __future__ import annotations

from typing import Any

import torch

from conversion.config import ConversionTrainingConfig, load_conversion_training_config
from conversion.data import build_conversion_dataloader, build_dense_eval_grid
from conversion.lti import run_lti_search, apply_lti_to_model, configure_lti_finetune_from_result
from conversion.neuron import TrainableMBENeuron
from conversion.train import train_conversion_model
from utils.seeds import set_deterministic_seed


def build_conversion_model(config: ConversionTrainingConfig) -> TrainableMBENeuron:
    model = TrainableMBENeuron(
        T=config.model.T,
        num_basis=config.model.num_basis,
        init=config.model.init,
        fold_abs_input_for_polarities=config.model.fold_abs_input_for_polarities,
    )
    return model.to(device=config.trainer.device, dtype=torch.float32)


def run_conversion_training(
    config_name: str,
    *,
    use_lti: bool = False,
    lti_budget: int = 64,
    lti_gamma_range: tuple[float, float] = (0.01, 0.5),
    label: str | None = None,
) -> dict[str, Any]:
    """
    Run conversion training with optional LTI initialization.

    Args:
        config_name: name of the training YAML config
        use_lti: if True, run LTI search before training
        lti_budget: number of LTI candidates to screen
        lti_gamma_range: gamma search bounds
        label: optional label for logging

    Returns:
        dict with training results including history, metrics, eval
    """
    config = load_conversion_training_config(config_name)

    # Modify output dir based on mode
    if use_lti:
        import dataclasses
        new_artifacts = dataclasses.replace(config.artifacts, root_dir=f"{config.artifacts.root_dir}_lti")
        config = dataclasses.replace(config, name=f"{config.name}_lti", artifacts=new_artifacts)

    set_deterministic_seed(config.trainer.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_conversion_dataloader(config)
    model = build_conversion_model(config)

    tag = label or ("LTI" if use_lti else "Manual")
    print(f"\n{'='*60}")
    print(f"[{tag}] Starting conversion training: {config.run_name}")
    print(f"{'='*60}")

    if use_lti:
        # Run LTI search
        x, y = build_dense_eval_grid(config)
        lti_result = run_lti_search(
            config, x, y,
            budget=lti_budget,
            gamma_range=lti_gamma_range,
            seed=config.trainer.seed,
        )
        apply_lti_to_model(model, lti_result, config.model.init)
        configure_lti_finetune_from_result(model, lti_result)

    result = train_conversion_model(model, train_loader, config)

    print(f"[{tag}] Done: best_loss={result['metrics']['best_loss']:.6f}, "
          f"final_mse={result['metrics']['final_mse']:.6f}, "
          f"final_mae={result['metrics']['final_mae']:.6f}")

    return result
