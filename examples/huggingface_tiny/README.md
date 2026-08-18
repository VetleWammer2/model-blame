# Offline recorded Hugging Face demo

This example constructs a tiny `GPT2LMHeadModel` from the local
[`config.json`](model/config.json); it never downloads model weights or code.
It then runs actual CPU training through ModelBlame's Hugging Face adapter,
audits an unchanged replay, reduces a real occurrence-ablation counterfactual,
builds the normal evidence bundle, and verifies the bundle by re-executing its
final patch in an isolated process.

Install the supported Hugging Face extra and run:

```console
python -m pip install -e ".[huggingface]"
python examples/huggingface_tiny/run_demo.py --output hf-demo-output
```

The output directory must not already exist. `demo-result.json` reports the
grade and measurements produced by that execution; the repository does not
ship precomputed benchmark output. The model is deliberately tiny, but the
demo performs multiple training and replay passes and can take a few minutes
on a CPU.
