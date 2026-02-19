CatalMOF
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/catalmof/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/catalmof/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/CatalMOF/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/CatalMOF/branch/main)


We are generating DFT-ready SBU clusters for stable MOFs with catalytic potential.

### Text mining

CatalMOF does not ship a manuscript CSV. You provide it and point config at it via `paths.manuscript_data_csv`. The CSV must have columns **name** (MOF refcode, matching stable MOF names), **doi**, and **title**.

- **Title-only** (`text_mining_title_only: true`): Provide the manuscript CSV with names, DOIs, and titles. Use the title fetcher to fill titles from a DOI list (same CSV format as the downloader):

  ```bash
  pip install catalmof[text_mining_titles]
  python -m catalmof.text_mining_tools.title_fetcher --doi-csv dois.csv --output-csv manuscript_data_w_titles.csv
  ```

  If your DOI CSV includes a **name** column (e.g. from a CSD export), the output is ready for `manuscript_data_csv`.

- **Full-paper**: Provide the same manuscript CSV and a directory of article HTML/XML files (`paths.text_mining_html_dir`). Set `paths.text_mining_pickle_dir` for pickler output. CatalMOF runs the internal pickler on your HTMLs (pickles are not assumed pre-provided; existing pickles are skipped by default). To download HTML/XML by DOI first:

  ```bash
  python -m catalmof.text_mining_tools.downloader --doi-csv dois.csv --output-dir /path/to/html/corpus
  ```

  Downloader CSV: `doi` column required; optional `mode` column. For Elsevier DOIs you need an API key (obtain separately; use `--elsevier-key` or `CATALMOF_ELSEVIER_API_KEY`). ACS is skipped (violation of ACS policy). Wiley is supported.

### R-factor check (SBU analysis)

If you set `run_rfactor_check: true`, you must provide a CSV with MOF refcodes and R-factors (`paths.core_rfactors_csv`). CatalMOF does not ship this file. You can obtain R-factor data separately using the **CCDC/CSD API** (a CSD license is required). The CSV must have columns: **name** (MOF refcode), **R-factor** (numeric).

### Zeo++ (geometric analysis)

CatalMOF does not ship Zeo++. For geometric featurization (pore diameter, surface area, pore volume, etc.) you must download and build Zeo++ yourself, then set `zeo_network` in your config to the path of the `network` binary. Use **version 0.3** to match the code paths used in CatalMOF.

- **Download:** [Zeo++ — Download](https://www.zeoplusplus.org/download.html)

### Copyright

Copyright (c) 2024, Husain Adamji


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
