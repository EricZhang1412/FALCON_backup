# FALCON

FALCON is a prototype for ANN-to-SNN conversion at the operator level. It fits scalar ANN nonlinearities with trainable MBE neurons, then validates the converted operators inside Hugging Face causal language models.

The current focus is:

- single-op conversion for activations such as `gelu`, `silu`, `tanh`, `relu`, and `rsqrt`
- Log-Timescale Initialization (LTI) for faster and better MBE fitting
- LayerNorm `rsqrt(var + eps)` conversion with per-LayerNorm MBE banks
- HF model validation with perplexity and logit-alignment metrics

## Project Layout

```text
FALCON_backup/
  falcon/
    conversion/              # MBE conversion training, LTI, data, config parsing
    models/                  # MBE runtime, decoder, LayerNorm rsqrt replacement
    configs/                 # Python config loaders
    utils/                   # Shared utilities
  configs/
    yaml/                    # YAML experiment configs
  scripts/
    single_op_conversion/    # Single-op training, comparison, spike analysis
    layernorm_conversion/    # Per-LN rsqrt bank build/train/pipeline scripts
    model_validation/        # HF model validation and WikiText sampling scripts
  data/                      # Small local evaluation text files
```

Library imports use the `falcon.*` package namespace:

```python
from falcon.conversion.runner import run_conversion_training
from falcon.models import MBENeuronDecoder
```

The old top-level script/module paths are intentionally not kept.

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

The default YAML runtime device is configured in `configs/yaml/project/defaults.yaml`. Most scripts also expose `--device` so you can override it from the command line.

## Single-Op Conversion

Train one target function with the standard manual initialization:

```bash
uv run python -m scripts.single_op_conversion.train_conversion gelu_conversion
```

Train with LTI initialization:

```bash
uv run python -m scripts.single_op_conversion.train_conversion gelu_conversion --lti
```

Compare manual initialization against LTI:

```bash
uv run python -m scripts.single_op_conversion.compare_initializers gelu_conversion --epochs 300 --budget 64
```

Analyze spike statistics for a trained checkpoint:

```bash
uv run python -m scripts.single_op_conversion.analyze_spikes gelu_conversion --comparison-mode lti
```

Available single-op configs live in `configs/yaml/training/`, for example:

- `gelu_conversion`
- `silu_conversion`
- `tanh_conversion`
- `relu_conversion`
- `rsqrt_conversion`

## LayerNorm `rsqrt` Conversion

LayerNorm conversion keeps mean, variance, and affine parameters exact, and replaces only `rsqrt(var + eps)` with an MBE approximator.

Build one rsqrt fitting config per LayerNorm module:

```bash
uv run python -m scripts.layernorm_conversion.build_rsqrt_bank \
  --hf-model gpt2 \
  --text-file data/eval_texts_wikitext.txt \
  --manifest-out outputs/ln_rsqrt_bank/gpt2_manifest.json
```

Train the generated bank:

```bash
uv run python -m scripts.layernorm_conversion.train_rsqrt_bank \
  --manifest outputs/ln_rsqrt_bank/gpt2_manifest.json \
  --lti
```

Run the full build/train/evaluate pipeline:

```bash
uv run python -m scripts.layernorm_conversion.run_rsqrt_pipeline \
  --hf-model gpt2 \
  --text-file data/eval_texts_wikitext.txt
```

For a larger WikiText evaluation example, see:

```bash
bash scripts/model_validation/ppl_test.sh
```

## HF Validation

Validate a Hugging Face causal LM after replacing the target activation with manual and LTI MBE checkpoints:

```bash
uv run python -m scripts.model_validation.validate_mbe_inference \
  --hf-model gpt2 \
  --conversion-config gelu_conversion \
  --comparison-root outputs/comparison/gelu_conversion \
  --text-file data/eval_texts_wikitext.txt \
  --max-samples 64 \
  --max-length 512 \
  --stride 256 \
  --device auto \
  --dtype float32
```

Validate with a per-LayerNorm `rsqrt` bank:

```bash
uv run python -m scripts.model_validation.validate_mbe_inference \
  --hf-model gpt2 \
  --conversion-config gelu_conversion \
  --comparison-root outputs/comparison/gelu_conversion \
  --rsqrt-bank-manifest outputs/ln_rsqrt_bank/gpt2_manifest.json \
  --text-file data/eval_texts_wikitext.txt \
  --max-samples 64
```

Sample evaluation text from WikiText:

```bash
uv run python -m scripts.model_validation.sample_wikitext --num-samples 500
```

## Outputs

Common output locations:

- `outputs/conversion/`: manual single-op conversion checkpoints and metrics
- `outputs/conversion_lti/`: LTI single-op conversion checkpoints and metrics
- `outputs/comparison/`: manual vs LTI comparison plots and summaries
- `outputs/ln_rsqrt_bank/`: per-LayerNorm rsqrt bank manifests
- `outputs/validation/`: HF validation reports

These directories are experiment artifacts and can be regenerated.

## Adding a New Target Function

1. Add the target function to `TARGETS` in `falcon/conversion/data.py`.
2. Add a YAML config in `configs/yaml/training/<name>_conversion.yaml`.
3. Run a quick comparison:

```bash
uv run python -m scripts.single_op_conversion.compare_initializers <name>_conversion --epochs 50 --budget 8
```

## Development Checks

After structural edits, run:

```bash
uv run python -m compileall falcon scripts
uv run python -c "import falcon; from falcon.conversion.runner import run_conversion_training; from falcon.models import MBENeuronDecoder"
uv run python -c "from falcon.conversion.config import load_conversion_training_config; print(load_conversion_training_config('gelu_conversion').name)"
uv run python -m scripts.single_op_conversion.compare_initializers relu_conversion --epochs 1 --budget 2
```
