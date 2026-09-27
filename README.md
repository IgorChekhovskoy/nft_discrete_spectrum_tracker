# Discrete Spectrum Tracking for direct nonlinear Fourier transform (NFT) – Reproducible Notebook

This repository contains a single Jupyter notebook that implements two methods for tracking discrete eigenvalues of the Zakharov–Shabat scattering problem along the longitudinal coordinate `z`:
1) a deterministic tracker with distance-based gating and optional assignment, and
2) a Kalman-filter-based tracker with χ² gating and assignment.

The notebook renders all figures inline, so the results can be viewed directly on GitHub or in Jupyter without extra steps.

---

## Associated paper

This notebook accompanies the paper:

> Chekhovskoy, I. S., Shtyrina, O. V., and Fedoruk, M. P. *Nonlinear Fourier Transform as a Tool for Analyzing the Soliton Dynamics in Systems Obeying the Haus–Ginzburg–Landau Equation*. **Bulletin of the Lebedev Physics Institute** 52, Suppl. 11, S1151–S1160 (2025). https://doi.org/10.3103/S1068335625604571

The repository includes the example dataset for the discrete-spectrum tracing demonstration described in the paper.

---

## Structure

```
.
├── README.md                  # this file
├── LICENSE
├── test_trackers.py           # regression tests and synthetic trajectories
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
   To use another data file, either rename it to the same filename or set the `FILENAME` variable at the top of the notebook to your path.

---

## What’s inside the notebook

- **Data loading**: reads columns (`z`, eigenvalue parts `x/h` → `Re(ζ)/Im(ζ)`, scattering coefficient parts `Re(r)/Im(r)`, etc.) into a `pandas` DataFrame.
- **Method A – deterministic tracker**:
  - linear extrapolation of `(Re(ζ), Im(ζ))` and `(Re(r), Im(r))`;
  - adaptive local thresholds;
  - weighted distances normalized by their gate radii;
  - assignment via `scipy.optimize.linear_sum_assignment`, with an explicit unmatched option.
- **Method B – Kalman + χ² gating + assignment**:
  - constant-velocity state for ζ and r (positions and velocities);
  - χ² gating using the Mahalanobis distance;
  - continuous-time process covariance for nonuniform z steps;
  - adapted measurement covariance with prediction uncertainty subtracted;
  - optional assignment as above, fixed normalization, and two-point velocity initialization.
- **Visualization**: 2D plots over `z` and 3D trajectories with Plotly; figures are embedded inline.

---

## Tests

From this directory, run:

```bash
python -m unittest -v test_trackers
```

The tests execute the functions from the notebook itself, without launching a kernel or displaying plots. They cover empty frames, births and gaps, disabled channels, preservation of raw observations, assignment rejection, covariance updates, nonuniform steps, synthetic identities, and the bundled dataset. Only NumPy, pandas and SciPy are required in addition to the Python standard library.

To evaluate both trackers on synthetic trajectories:

```bash
python -c "import json; from test_trackers import benchmark_synthetic; print(json.dumps(benchmark_synthetic(), indent=2))"
```

`benchmark_synthetic` accepts an optional notebook path. The default series contains ten cases with twenty seeds each. Identities are used only for scoring. These artificial trajectories test association logic, not the accuracy of a direct Zakharov–Shabat solver or admissibility of scattering data.

## Data, parameters and output

Pass a sorted, strictly increasing `z_levels` array containing all analyzed slices, including slices with no detections. `np.sort(df['z'].unique())` cannot recover completely empty slices from a detections-only file. In the bundled file the coordinate is labeled `z/L`; use the same coordinate and units throughout the calculation.

`max_gap` is the maximum number of missing frames that an established track may bridge; zero permits adjacent frames only. Unconfirmed Kalman tracks expire after `birth_confirm` consecutive misses. A zero `weight_r` or `meas_w_r` removes the discrete-amplitude channel from both gating and assignment. For the Kalman tracker, `meas_w_zeta=0` similarly disables the eigenvalue channel; at least one channel must be active.

For the deterministic tracker, `birth_threshold` is the initial and minimum gate radius. For the Kalman tracker it bounds the first eigenvalue displacement before a velocity is measured. Discrete amplitudes use their covariance gate, so their faster rotation need not split the eigenvalue track. The same positional tolerance defines the floor for covariance adaptation. `R_init` is an initial estimate, not that lower bound. `Q_init` represents process-noise intensity per unit z and is integrated over each actual step. Covariance values must be reconsidered when coordinate units or data scales change. The numerical settings are configured for the bundled dataset.

`groups` / `groups_local` retain raw observations, including short and unconfirmed tracks. Missing detections are not filled with filter predictions. `states_log` contains filter-velocity estimates in internally normalized coordinates, including prediction-only frames; these are not independent spectral derivatives. `filtered_groups` selects tracks for visualization by length and preserves the coordinates and multiplicity of their observations.

The input `Re_r` and `Im_r` columns are discrete amplitude data supplied by the exporter. They must not be confused with the continuous-spectrum reflection coefficient. A new or interrupted track does not by itself establish physical creation or disappearance of a soliton. Inspect ambiguous associations near close branches, long gaps, or abrupt changes in motion.

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

## Example dataset

The bundled dataset contains the discrete-spectrum observations used in the associated paper. Run the notebook from the repository directory to load this file and plot the tracked branches. To use another dataset, set `FILENAME` in each tracker cell.

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
