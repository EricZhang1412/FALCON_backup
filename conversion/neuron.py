from __future__ import annotations

import torch
import torch.nn as nn

from conversion.config import ModelInitSpec, PolarityInitSpec


class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_scaled):
        ctx.save_for_backward(v_scaled)
        return (v_scaled > 0).to(v_scaled.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (v_scaled,) = ctx.saved_tensors
        return grad_output * torch.clamp(1.0 - torch.abs(v_scaled), min=0.0)


class TrainableMBENeuron(nn.Module):
    def __init__(
        self,
        *,
        T: int,
        num_basis: int,
        init: ModelInitSpec,
        fold_abs_input_for_polarities: bool = False,
    ) -> None:
        super().__init__()
        if num_basis % 2 != 0:
            raise ValueError("num_basis must be even.")

        self.T = T
        self.num_basis = num_basis
        self.num_pos_basis = num_basis // 2
        self.num_neg_basis = num_basis // 2
        self.fold_abs_input_for_polarities = bool(fold_abs_input_for_polarities)

        self.log_alpha_v = nn.Parameter(self._build_log_alpha(init, "threshold"))
        self.log_alpha_r = nn.Parameter(self._build_log_alpha(init, "reset"))
        self.log_alpha_d = nn.Parameter(self._build_log_alpha(init, "output"))
        self.log_lambda_v = nn.Parameter(self._build_logit_lambda(init, "threshold"))
        self.log_lambda_r = nn.Parameter(self._build_logit_lambda(init, "reset"))
        self.log_lambda_d = nn.Parameter(self._build_logit_lambda(init, "output"))

        self.fc = nn.Linear(num_basis, 1, bias=False)
        with torch.no_grad():
            self.fc.weight[:, :self.num_pos_basis].normal_(mean=0.2, std=0.05)
            self.fc.weight[:, self.num_pos_basis:].normal_(mean=-0.2, std=0.05)

    def _build_log_alpha(self, init: ModelInitSpec, field: str) -> torch.Tensor:
        pos = getattr(init.positive, field)
        neg = getattr(init.negative, field)
        return torch.cat([
            torch.log(self._linspace(pos.alpha.min, pos.alpha.max, self.num_pos_basis)),
            torch.log(self._linspace(neg.alpha.min, neg.alpha.max, self.num_neg_basis)),
        ])

    def _build_logit_lambda(self, init: ModelInitSpec, field: str) -> torch.Tensor:
        pos = getattr(init.positive, field)
        neg = getattr(init.negative, field)
        return torch.cat([
            torch.logit(self._linspace(pos.lambda_.min, pos.lambda_.max, self.num_pos_basis)),
            torch.logit(self._linspace(neg.lambda_.min, neg.lambda_.max, self.num_neg_basis)),
        ])

    def _linspace(self, lo: float, hi: float, n: int) -> torch.Tensor:
        if n == 1:
            return torch.tensor([(lo + hi) * 0.5], dtype=torch.float32)
        return torch.linspace(lo, hi, steps=n, dtype=torch.float32)

    def _alpha(self, log_alpha):
        return torch.exp(log_alpha).reshape(1, -1)

    def _lambda(self, logit_lambda):
        return torch.sigmoid(logit_lambda).reshape(1, -1)

    def _signed_membrane(self, x):
        if self.fold_abs_input_for_polarities:
            x_abs = torch.abs(x)
            x_pos = x_abs
            x_neg = -x_abs
        else:
            x_pos = torch.clamp_min(x, 0.0)
            x_neg = -torch.clamp_min(-x, 0.0)
        return torch.cat([x_pos.repeat(1, self.num_pos_basis), x_neg.repeat(1, self.num_neg_basis)], dim=1)

    def _signed_threshold(self, threshold):
        return torch.cat([threshold[:, :self.num_pos_basis], -threshold[:, self.num_pos_basis:]], dim=1)

    def _signed_reset(self, reset):
        return torch.cat([reset[:, :self.num_pos_basis], -reset[:, self.num_pos_basis:]], dim=1)

    def _signed_vscaled(self, membrane, threshold):
        pos = membrane[:, :self.num_pos_basis] - threshold[:, :self.num_pos_basis]
        neg = threshold[:, self.num_pos_basis:] - membrane[:, self.num_pos_basis:]
        return torch.cat([pos, neg], dim=1)

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        lambda_v = self._lambda(self.log_lambda_v)
        lambda_r = self._lambda(self.log_lambda_r)
        lambda_d = self._lambda(self.log_lambda_d)

        threshold = self._signed_threshold(self._alpha(self.log_alpha_v))
        reset = self._signed_reset(self._alpha(self.log_alpha_r))
        output = self._alpha(self.log_alpha_d)
        membrane = self._signed_membrane(x)
        basis_output = torch.zeros_like(membrane)

        for _ in range(self.T):
            spikes = ActFun.apply(self._signed_vscaled(membrane, threshold))
            basis_output = basis_output + spikes * output
            membrane = membrane - spikes * reset
            threshold = self._signed_threshold(torch.abs(threshold) * lambda_v)
            reset = self._signed_reset(torch.abs(reset) * lambda_r)
            output = output * lambda_d

        return self.fc(basis_output)

    @torch.no_grad()
    def export_model_spec(self, *, name: str) -> dict:
        alpha_v = torch.exp(self.log_alpha_v)
        alpha_r = torch.exp(self.log_alpha_r)
        alpha_d = torch.exp(self.log_alpha_d)
        lambda_v = torch.sigmoid(self.log_lambda_v)
        lambda_r = torch.sigmoid(self.log_lambda_r)
        lambda_d = torch.sigmoid(self.log_lambda_d)
        weights = self.fc.weight.detach().cpu().reshape(-1).tolist()

        basis_schedules = []
        for i in range(self.num_basis):
            pol = "positive" if i < self.num_pos_basis else "negative"
            gi = i if i < self.num_pos_basis else i - self.num_pos_basis
            basis_schedules.append({
                "name": f"basis_{pol[:3]}_{gi}", "polarity": pol,
                "alpha_v": float(alpha_v[i]), "lambda_v": float(lambda_v[i]),
                "alpha_r": float(alpha_r[i]), "lambda_r": float(lambda_r[i]),
                "alpha_d": float(alpha_d[i]), "lambda_d": float(lambda_d[i]),
            })
        return {"name": name, "T": self.T, "basis_schedules": basis_schedules, "basis_weights": weights}
