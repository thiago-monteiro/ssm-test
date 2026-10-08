# SSM experiments

Requires Python 3.11 or newer.

For the small experiments and optimizer benchmark:

```sh
python -m pip install -r requirements.txt
python scripts/run_expA.py --device cpu --quick
python scripts/run_expB.py --help
python scripts/run_expC.py --help
python scripts/run_colab_normopt.py --device cpu --steps 100 --seeds 2
```

Results go into `results/`. The optimizer benchmark supports CPU, CUDA, and
TPU (`--device tpu`, with `torch_xla` installed in the TPU runtime).

The large Mamba experiments use a separate Linux x86_64 CUDA environment.
`requirements-mamba.txt` contains the existing Python 3.11 / PyTorch 2.4
CUDA wheels; create the pinned environment before running them:

```sh
conda env create -f environment.yml
conda activate ssm-test-plan
python scripts/run_large_mamba.py --help
```

`scripts/plot_all.py` plots the small experiment results, and
`scripts/compute_cis.py` computes their confidence intervals.
