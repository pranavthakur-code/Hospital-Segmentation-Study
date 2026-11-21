# Hospital Segmentation Study (Hyderabad community clinics)

Segment patients into actionable groups and design one action, with cost and a measurable target, for each.

> **Data are simulated.** The brief describes a fictional network and supplies no files, so `generate_data.py` creates a seeded, reproducible dataset (4 hospitals, 140 beds, ~510,000 rows, 2024-2025) that reproduces the problems named in the brief: free-text diagnosis, age as integer *or* date of birth, GST counted twice. Costs and effect sizes in the action plan are planning assumptions in INR, not measurements.

## Result in brief
* **8 segments**, each >= 7% of patients. Two inpatient segments are 15% of patients but 76% of revenue and 95% of bed-days.
* **Final method: k-means (k = 8)** on log-scaled behavioural features, k chosen by elbow + silhouette + a 5% minimum size + outcome-KPI separation. It beats **RFM** on bed-day / length-of-stay separation, and RFM has 3 of 6 segments with too few admissions to report any LOS or readmission rate. RFM is kept as a score field.
* **Stability (2024 vs 2025):** the same 8 profiles reappear independently (noise floor < 1%), but ~66% of patients seen in both years change segment (RFM: 69%). Re-score quarterly; trigger programmes on events.
* **Actions:** ~Rs 26 lakh cost, ~Rs 50 lakh expected benefit (low-high Rs 25-75 lakh), to be confirmed by a pilot with hold-out groups.

## Files
| File | What it is |
|---|---|
| `Hospital_Segmentation.ipynb` | Executed notebook: audit, cleaning, features, EDA, RFM, k-means, profiles, actions, stability, migration |
| `outputs/Hospital_Segmentation_Workbook.xlsx` | Segment profile table, actions with cost/benefit, assumptions, stability, migration matrix, data-quality log, patient-level segments |
| `outputs/segment_migration_chart.png` | 2024 -> 2025 segment migration |
| `outputs/segment_profile_table.csv`, `outputs/actions_cost_benefit.csv` | Same tables as CSV |
| `generate_data.py` | Synthetic data generator (seed 2025) |
| `prep.py` | Cleaning (age parsing, diagnosis mapping, GST fix, refunds, duplicates) and feature engineering |
| `seg.py` | RFM, k-means helpers, KPI profiling, matching and stability |

## Run
```
pip install pandas numpy scipy scikit-learn matplotlib openpyxl nbformat jupyter
python generate_data.py data
jupyter nbconvert --to notebook --execute --inplace Hospital_Segmentation.ipynb
```
To use real data, drop files with the same columns into `data/` (see `generate_data.py` for the schema) and re-run.

## Key design choices
* **Clustering inputs are behavioural only** (frequency, recency, value ex-GST, growth, seasonality, age, chronic share, labs per visit). LOS, bed-days, occupancy and 30-day readmission are outcome KPIs for profiling.
* **Caveat:** bed-days and readmission are mechanically tied to the admission count (a readmission needs two admissions), so average LOS is the cleanest outcome check.
* **Revenue is the ex-GST base amount**, so double-counted GST cannot enter any value feature.
* **Limitations:** single-year behaviour is noisy; the simulator's archetypes were never used in the analysis (appendix check only); effect sizes are assumptions.
