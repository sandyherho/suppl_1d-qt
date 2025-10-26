# Supplementary Materials: "`1d-qt-ideal-solver`: 1D idealized quantum tunneling solver with absorbing boundaries"

[![DOI](https://zenodo.org/badge/1083497967.svg)](https://doi.org/10.5281/zenodo.17446362)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains supplementary Python code for statistical analysis and visualization of quantum tunneling case studies. The analysis scripts generate comprehensive statistical reports comparing rectangular and Gaussian barrier scenarios.

## Contents

- **`script/basic_stats.py`**: Probability distribution analysis with entropy measures and hypothesis testing
- **`script/phase_space.py`**: Phase space analysis (Re(ψ) vs Im(ψ)) with information-theoretic metrics
- **`script/basic_dynamic_plot.py`**: Publication-quality visualization of quantum tunneling dynamics
- **`stats/*.txt`**: Generated statistical analysis reports

## Data Access

The NetCDF data files, animations, and publication-quality figures referenced by these scripts are available at:

**OSF Repository**: [https://doi.org/10.17605/OSF.IO/RVDQ2](https://doi.org/10.17605/OSF.IO/RVDQ2)

## Main Solver

For the source code of the 1D quantum tunneling solver itself, visit:

**GitHub Repository**: [https://github.com/sandyherho/1d-qt-ideal-solver](https://github.com/sandyherho/1d-qt-ideal-solver)

## Usage

1. Download NetCDF data files from the OSF repository
2. Place them in `raw_data/` directory
3. Run analysis scripts:
   ```bash
   python script/basic_stats.py
   python script/phase_space.py
   python script/basic_dynamic_plot.py
   ```
4. Results will be saved to `stats/` and `figs/` directories

## License

MIT License (see LICENSE file)
