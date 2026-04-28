from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from falcon.configs import coerce_dtype, load_project_defaults, load_training_spec, DEFAULT_SEED


@dataclass(frozen=True)
class TargetSpec:
    name: str
    x_min: float
    x_max: float


@dataclass(frozen=True)
class DataSpec:
    num_samples: int
    batch_size: int
    shuffle: bool
    sampling: str
    eval_num_samples: int


@dataclass(frozen=True)
class RangeSpec:
    min: float
    max: float


@dataclass(frozen=True)
class ScheduleInitSpec:
    alpha: RangeSpec
    lambda_: RangeSpec


@dataclass(frozen=True)
class PolarityInitSpec:
    threshold: ScheduleInitSpec
    reset: ScheduleInitSpec
    output: ScheduleInitSpec


@dataclass(frozen=True)
class ModelInitSpec:
    positive: PolarityInitSpec
    negative: PolarityInitSpec


@dataclass(frozen=True)
class ModelSpec:
    T: int
    num_basis: int
    init: ModelInitSpec
    fold_abs_input_for_polarities: bool


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    lr: float
    weight_decay: float


@dataclass(frozen=True)
class SchedulerSpec:
    name: str
    gamma: float
    step_every: int


@dataclass(frozen=True)
class TrainerSpec:
    max_epochs: int
    seed: int
    device: str
    dtype: torch.dtype
    log_every: int


@dataclass(frozen=True)
class CheckpointSpec:
    dir: str
    save_best: bool
    save_latest: bool
    resume_from: str | None


@dataclass(frozen=True)
class ArtifactSpec:
    root_dir: str
    save_plots: bool
    save_predictions: bool


@dataclass(frozen=True)
class ExportSpec:
    export_runtime_config: bool
    export_model_name: str
    export_to_yaml: bool


@dataclass(frozen=True)
class ConversionTrainingConfig:
    name: str
    target: TargetSpec
    data: DataSpec
    model: ModelSpec
    optimizer: OptimizerSpec
    scheduler: SchedulerSpec
    trainer: TrainerSpec
    checkpoint: CheckpointSpec
    artifacts: ArtifactSpec
    export: ExportSpec

    @property
    def run_name(self) -> str:
        return self.name

    @property
    def output_dir(self) -> Path:
        return Path(self.artifacts.root_dir) / self.run_name


def _load_range(spec, *, key, lower_bound=None, upper_bound=None):
    minimum, maximum = float(spec["min"]), float(spec["max"])
    if minimum > maximum:
        raise ValueError(f"{key}.min must be <= {key}.max")
    if lower_bound is not None and minimum <= lower_bound:
        raise ValueError(f"{key}.min must be > {lower_bound}")
    if upper_bound is not None and maximum >= upper_bound:
        raise ValueError(f"{key}.max must be < {upper_bound}")
    return RangeSpec(min=minimum, max=maximum)


def _load_schedule_init(spec, *, key):
    return ScheduleInitSpec(
        alpha=_load_range(spec["alpha"], key=f"{key}.alpha", lower_bound=0.0),
        lambda_=_load_range(spec["lambda"], key=f"{key}.lambda", lower_bound=0.0, upper_bound=1.0),
    )


def _load_polarity_init(spec, *, key):
    return PolarityInitSpec(
        threshold=_load_schedule_init(spec["threshold"], key=f"{key}.threshold"),
        reset=_load_schedule_init(spec["reset"], key=f"{key}.reset"),
        output=_load_schedule_init(spec["output"], key=f"{key}.output"),
    )


def _load_model_init(spec):
    return ModelInitSpec(
        positive=_load_polarity_init(spec["positive"], key="model.init.positive"),
        negative=_load_polarity_init(spec["negative"], key="model.init.negative"),
    )


def load_conversion_training_config(name: str) -> ConversionTrainingConfig:
    defaults = load_project_defaults()
    runtime_defaults = defaults["runtime"]
    p = load_training_spec(name)

    return ConversionTrainingConfig(
        name=str(p.get("name", name)),
        target=TargetSpec(name=str(p["target"]["name"]), x_min=float(p["target"]["domain"]["x_min"]), x_max=float(p["target"]["domain"]["x_max"])),
        data=DataSpec(
            num_samples=int(p["data"]["num_samples"]), batch_size=int(p["data"]["batch_size"]),
            shuffle=bool(p["data"].get("shuffle", True)), sampling=str(p["data"].get("sampling", "linspace")),
            eval_num_samples=int(p["data"].get("eval_num_samples", p["data"]["num_samples"])),
        ),
        model=ModelSpec(
            T=int(p["model"]["T"]),
            num_basis=int(p["model"]["num_basis"]),
            init=_load_model_init(p["model"]["init"]),
            fold_abs_input_for_polarities=bool(p["model"].get("fold_abs_input_for_polarities", False)),
        ),
        optimizer=OptimizerSpec(name=str(p["optimizer"].get("name", "adam")), lr=float(p["optimizer"]["lr"]), weight_decay=float(p["optimizer"].get("weight_decay", 0.0))),
        scheduler=SchedulerSpec(name=str(p["scheduler"].get("name", "exponential")), gamma=float(p["scheduler"].get("gamma", 0.99)), step_every=int(p["scheduler"].get("step_every", 10))),
        trainer=TrainerSpec(
            max_epochs=int(p["trainer"]["max_epochs"]), seed=int(p["trainer"].get("seed", DEFAULT_SEED)),
            device=str(p["trainer"].get("device", runtime_defaults["device"])),
            dtype=coerce_dtype(p["trainer"].get("dtype"), runtime_defaults["dtype"]),
            log_every=int(p["trainer"].get("log_every", 10)),
        ),
        checkpoint=CheckpointSpec(
            dir=str(p["checkpoint"].get("dir", "checkpoints")), save_best=bool(p["checkpoint"].get("save_best", True)),
            save_latest=bool(p["checkpoint"].get("save_latest", True)),
            resume_from=str(p["checkpoint"]["resume_from"]) if p["checkpoint"].get("resume_from") not in (None, "") else None,
        ),
        artifacts=ArtifactSpec(root_dir=str(p["artifacts"].get("root_dir", "outputs/conversion")), save_plots=bool(p["artifacts"].get("save_plots", True)), save_predictions=bool(p["artifacts"].get("save_predictions", True))),
        export=ExportSpec(export_runtime_config=bool(p["export"].get("export_runtime_config", True)), export_model_name=str(p["export"].get("export_model_name", name)), export_to_yaml=bool(p["export"].get("export_to_yaml", True))),
    )
