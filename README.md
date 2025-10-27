# Discrete Spectrum Tracking (NFT) – Reproducible Notebook

This repository contains a single Jupyter notebook that implements two methods for tracking discrete eigenvalues of the Zakharov–Shabat scattering problem across longitudinal coordinate `z`:
1) a deterministic tracker with distance-based gating and the Hungarian algorithm, and  
2) a Kalman-filter-based tracker with χ² gating and assignment.

The notebook renders all figures inline so you can view the results on GitHub or in Jupyter without extra steps.

---

## Structure

```
.
├── README.md                  # this file
├── requirements.txt           # pinned Python dependencies
├── ds_tracker.ipynb
└── NFT_DiscreteSpectrum.dat   # example dataset
```

> The dataset file is included alongside the notebook so the workflow is fully reproducible out of the box.

---

## Quick start

1. **Create an environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install -r requirements.txt
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
   If you want to use your own data file, either rename it to the same filename or set the `filename` variable at the top of the notebook to your path.

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

The notebook already displays Plotly figures inline. If you also want to **save static images** (PNG/SVG/PDF) locally, install Kaleido and use Plotly’s static export:
```bash
pip install -U kaleido
```
Then call, for example:
```python
fig.write_image("figure.png", scale=2)  # high-DPI export
```

---

## Requirements

See `requirements.txt` for pinned versions. Core libraries:
- `numpy`, `pandas`
- `scipy` (Hungarian algorithm for assignment)
- `plotly` (interactive and 3D figures; optional `kaleido` for static export)

---

## Reproducing the paper result

The provided dataset file is the exact snapshot used to reproduce the mode shown in the paper. Simply keep the file next to the notebook (same directory) and run all cells—no path edits required. The notebook produces the same branch tracking visuals and summary plots as in the manuscript.

---

## Citing

If you use this code or figures in your research, please cite the associated paper (add your bibliographic entry here).

---

## License

MIT.
