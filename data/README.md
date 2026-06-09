# Data

COVID-19 data were downloaded from "A time-resolved proteomic and prognostic map of COVID-19" (Demichev et al. 2021): https://doi.org/10.1016/j.cels.2021.05.005.

The sepsis data were generated in-house and are currently being uploaded.

## Repository Contents

- `covid/raw/`: source COVID-19 tables used by the preprocessing notebook.
- `covid/quant.csv` and `covid/meta.csv`: processed COVID-19 abundance and metadata tables generated from the raw source files.
- `covid/covid_uniprot_annotations.csv`: cached UniProt annotation table used to make protein labels reproducible without repeated API calls.
- `processed/`: processed sepsis abundance, metadata, and annotation tables.
