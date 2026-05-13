# Discrete Spectrum Tracking for direct nonlinear Fourier transform (NFT) – Reproducible Notebook

This repository contains a single Jupyter notebook that implements two methods for tracking discrete eigenvalues of the Zakharov–Shabat scattering problem along the longitudinal coordinate `z`:
1) a deterministic tracker with distance-based gating and the Hungarian algorithm, and  
2) a Kalman-filter-based tracker with χ² gating and assignment.

The notebook renders all figures inline, so the results can be viewed directly on GitHub or in Jupyter without extra steps.

---

## Associated paper

This notebook accompanies the paper:

> Chekhovskoy, I. S., Shtyrina, O. V., and Fedoruk, M. P. *Nonlinear Fourier Transform as a Tool for Analyzing the Soliton Dynamics in Systems Obeying the Haus–Ginzburg–Landau Equation*. **Bulletin of the Lebedev Physics Institute** 52, Suppl. 11, S1151–S1160 (2025). https://doi.org/10.3103/S1068335625604571

The repository provides the reproducible notebook and example dataset used for the discrete-spectrum tracing demonstration discussed in the paper.

---

## Structure

```
.
├── README.md                  # this file
├── LICENSE
├── ds_tracker.ipynb           # reproducible notebook
└── NFT_DiscreteSpectrum.dat   # example dataset
```

> The dataset file is included alongside the notebook, so the workflow is reproducible out of the box.

---

## Quick start

1. **Create an environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   python -m pip install -U pip
   python -m pip install numpy pandas scipy plotly jupyter
   ```

2. **Launch Jupyter and open the notebook**
   ```bash
   jupyter lab       # or: jupyter notebook
   ```

3. **Run all cells**  
   The notebook reads the dataset **from the file placed next to it**:
   ```
   NFT_DiscreteSpectrum.dat
   ```
   To use another data file, either rename it to the same filename or set the `filename` variable at the top of the notebook to your path.

---

## What’s inside the notebook

- **Data loading**: reads columns (`z`, eigenvalue parts `x/h` → `Re(ζ)/Im(ζ)`, scattering coefficient parts `Re(r)/Im(r)`, etc.) into a `pandas` DataFrame.
- **Method A – deterministic tracker**:
  - linear extrapolation of `(Re(ζ), Im(ζ))` and `(Re(r), Im(r))`;
  - adaptive local thresholds;
  - cost matrix as a weighted sum of Euclidean distances;
  - assignment solved by the Hungarian algorithm via `scipy.optimize.linear_sum_assignment`.
- **Method B – Kalman + χ² gating + assignment**:
  - constant-velocity state for ζ and r (positions and velocities);
  - χ² gating using the Mahalanobis distance;
  - adaptive noise tuning from innovations;
  - Hungarian assignment as above.
- **Visualization**: 2D plots over `z` and 3D trajectories with Plotly; figures are embedded inline.

---

## Exporting figures (optional)

The notebook already displays Plotly figures inline. To also **save static images** (PNG/SVG/PDF) locally, install Kaleido and use Plotly’s static export:
```bash
python -m pip install -U kaleido
```

Then call, for example:
```python
fig.write_image("figure.png", scale=2)  # high-DPI export
```

---

## Requirements

Core libraries:
- `numpy`, `pandas`
- `scipy` (Hungarian algorithm for assignment)
- `plotly` (interactive and 3D figures)
- `jupyter` or `jupyterlab` (to run the notebook locally)

Optional:
- `kaleido` for static Plotly figure export

---

## Reproducing the paper result

The provided dataset file is the snapshot used to reproduce the discrete-spectrum tracing example shown in the paper. Keep the file next to the notebook and run all cells — no path edits are required. The notebook produces the same branch-tracking visuals for the included dataset.

---

## Citing

If you use this code, dataset, or figures in your research, please cite:

```bibtex
@article{Chekhovskoy2025NFT_HGLE,
  author  = {Chekhovskoy, I. S. and Shtyrina, O. V. and Fedoruk, M. P.},
  title   = {Nonlinear Fourier Transform as a Tool for Analyzing the Soliton Dynamics in Systems Obeying the Haus--Ginzburg--Landau Equation},
  journal = {Bulletin of the Lebedev Physics Institute},
  volume  = {52},
  number  = {Suppl. 11},
  pages   = {S1151--S1160},
  year    = {2025},
  doi     = {10.3103/S1068335625604571}
}
```

---

## License

MIT.
