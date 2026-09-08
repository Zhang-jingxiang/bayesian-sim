# Bayesian SIM

This is an independent reproduction and experimental codebase for the Bayesian structured illumination microscopy (Bayesian SIM) method. The project is intended for algorithm research and reproducible experiments. 

## Implementation

The current implementation includes:

- SIM forward model
- Independent Gaussian noise likelihood and a Gaussian image prior with Laplacian regularization
- Gamma / Jeffreys hyperpriors and Gibbs sampling
- Solving the image conditional distribution through perturbation-optimization and preconditioned conjugate gradients
- Phase-shifted fringe patterns and calibrated per-frame `kx/ky/phase` patterns
- Optional 2x latent-variable reconstruction grid
- Optional shared smooth measurement-grid background and its posterior output

## Environment and Installation

Python `>=3.10` is required. Runtime dependencies are declared in [`pyproject.toml`](pyproject.toml).

```bash
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Testing requires an additional installation of `pytest`:

```bash
python -m pip install pytest
```

## Quick Verification

First verify the command-line entry point and the synthetic configuration:

```bash
python scripts/run_simulation.py --help
python scripts/run_simulation.py \
  --config configs/paper_test_pattern.json \
  --dry-run
```

`--dry-run` loads the configuration, generates patterns and simulated measurements, and checks their shapes, but does not run Gibbs sampling or write results. Once this succeeds, run the complete synthetic example:

```bash
python scripts/run_simulation.py \
  --config configs/paper_test_pattern.json
```

The default output directory is `results/paper_test_pattern/`. You can also use `--output` to override the output directory in the configuration.

## Configuration and Experiments

All JSON paths are resolved relative to the directory containing the configuration file, rather than relative to the current shell directory.

| Configuration | Type | Input requirements | Current status |
| --- | --- | --- | --- |
| `configs/paper_test_pattern.json` | synthetic simulation | No external data | Can be directly dry-run or run |
| `configs/sample1_512_sim9.json` | image simulation | `data/sample1-512.tif` | This file is not provided in the current repository |
| `configs/locksim_mito_real9.json` | real-data reconstruction | External LockSIM TIFF and OTF | Local paths must be prepared and modified first |
| `configs/cos7_first9_hr_background.json` | calibrated real-data reconstruction | COS7 TIFF in the repository, external OTF | The COS7 TIFF is provided; the OTF path in the configuration must be modified |

For example, the command to run COS7 is:

```bash
python scripts/run_simulation.py \
  --config configs/cos7_first9_hr_background.json
```

This configuration currently resolves the OTF to the external path `/mnt/SIM4Expt/OTF.tif`; if the file does not exist, the run will report an explicit error. Change `optics.otf_path` to an existing local `.tif` or `.npy` OTF file. The LockSIM configuration likewise references external data and an OTF; `sample1_512_sim9.json` first requires the TIFF specified in the configuration to be provided.

The frame-count sweep scripts also depend on the corresponding real data:

```bash
python scripts/run_locksim_frame_sweep.py
python scripts/run_cos7_frame_sweep.py --frame-counts 5 6 7 8 --workers 1
```

## Directory Structure

```text
bayesim2012/                         Core algorithm package
configs/                             Experiment JSON configurations
data/                                Currently provided input data and GT
doc/notebooks/                       Tutorial notebooks
doc/snapshots/                       Small versionable experiment summaries
scripts/                             Command-line entry points and frame-count sweep scripts
tests/                               Automated tests
2012_Bayesian_.../                   Paper text and small figures
pyproject.toml                       Python package and dependency declarations
```

## Tutorials

Tutorial instructions are available in [`doc/README.md`](doc/README.md), and the notebooks are located in [`doc/notebooks/`](doc/notebooks/). After installing Jupyter, launch them from the repository root:

```bash
python -m pip install jupyter
jupyter lab doc/notebooks
```

The complete COS7 reconstruction in the tutorials requires an external OTF and substantial computing resources; reading the notebooks and running some data inspection cells does not mean that the complete experiment will run automatically.

## Citation

If this project or its implementation helps your research, please cite the following papers:

> Jingxiang Zhang, Tianyu Zhao, Manming Shu, Keru Mou, Zheming Zhang, Yihan Sun,
> Mengrui Wang, Yansheng Liang, Shaowei Wang, and Ming Lei, “A unified
> reconstruction algorithm for reduced-frame structured illumination microscopy,”
> arXiv:2608.12964, 2026.
> [arXiv:2608.12964](https://arxiv.org/abs/2608.12964)

> François Orieux, Eduardo Sepulveda, Vincent Loriette, Benoit Dubertret, and
> Jean-Christophe Olivo-Marin, “Bayesian Estimation for Optimized Structured
> Illumination Microscopy,” *IEEE Transactions on Image Processing*, 2012.
> DOI: [10.1109/TIP.2011.2162741](https://doi.org/10.1109/TIP.2011.2162741)

## License

We claim an Apache licence for this project.
