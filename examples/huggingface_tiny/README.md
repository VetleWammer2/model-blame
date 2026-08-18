# Offline recorded Hugging Face demo

The demo builds a tiny `GPT2LMHeadModel` from the local
[`config.json`](model/config.json). It downloads no model weights and no code.
Then it trains on CPU through ModelBlame's Hugging Face adapter, audits an
unchanged replay, reduces an occurrence-ablation counterfactual, builds the
ordinary evidence bundle, and verifies that bundle by re-executing its final
patch in an isolated process.

Install the Hugging Face extra and run:

```console
python -m pip install -e ".[huggingface]"
python examples/huggingface_tiny/run_demo.py --output hf-demo-output
```

The output directory must not already exist. `demo-result.json` reports the
grade and measurements from that execution. No precomputed benchmark output
ships with the repository. The model is tiny, but the demo makes several
training and replay passes and can take a few minutes on a CPU.
