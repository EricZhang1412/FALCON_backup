uv run run_ln_rsqrt_pipeline.py \
  --hf-model gpt2-medium \
  --wikitext-subset wikitext-2-raw-v1 \
  --wikitext-split test \
  --max-samples 0 --batch-size 20 \
  --lti-budget 64 \
  --gamma-lo 0.01 --gamma-hi 0.5 \
  --device cuda:6 --dtype float32 \
  --skip-build --skip-train \
  --max-length 1024 --stride 512