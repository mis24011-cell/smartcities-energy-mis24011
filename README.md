# Κώδικας Διπλωματικής — ΝΕΟ τελικό δείγμα (5.000 κτίρια / 5 ζώνες / Copernicus)

Αναπαραγώγιμος κώδικας (Python) που παρήγαγε ΟΛΑ τα αποτελέσματα, σχήματα και πίνακες
της τελικής έκδοσης της διπλωματικής, στο νέο δείγμα:
**5.000 κτίρια (1.000/ζώνη × 5: 2A Houston, 3A Atlanta, 4A Baltimore, 5A Chicago, 5B Denver),
60% οικιστικά / 40% εμπορικά, 87.600 ωριαίες εγγραφές, κλιματικά σενάρια Copernicus
SSP2-4.5 & SSP5-8.5 για 2040/2050/2100.**

> Τα scripts τρέχουν και σε Jupyter: `%run <script>.py` σε ένα cell, ή επικόλληση του
> περιεχομένου σε cells. Είναι απλά .py αντί για .ipynb — η λογική/αποτελέσματα ταυτόσημα.

## Απαιτούμενες βιβλιοθήκες
```
pandas numpy matplotlib scipy scikit-learn xgboost optuna shap pyarrow lxml pillow
```
`pip install pandas numpy matplotlib scipy scikit-learn xgboost optuna shap pyarrow lxml pillow`

## Αρχεία δεδομένων (παράγονται από το 01, χρησιμοποιούνται από τα 02–05)
- `merged_master.parquet` — 87.600 γραμμές (10 προφίλ × 8.760h): κατανάλωση ανά καύσιμο,
  τελικές χρήσεις (eu.*), CO₂ ανά καύσιμο, zone, dataset_type, n_buildings.
- `scenarios_2x3_baseline.parquet` — προβολές σεναρίων (Baseline + SSP2-4.5/SSP5-8.5 ×
  2040/2050/2100 × 5 ζώνες): HDD/CDD, delta scaling, consumption_delta_pct.
- `comstock_metadata.parquet`, `resstock_metadata.parquet` — μεταδεδομένα NREL (τύποι κτιρίων,
  εμβαδόν, πολιτεία) για τα σχήματα χαρακτηριστικών. Δειγματοληψία: states TX/GA/MD/IL/CO → ζώνες 2A/3A/4A/5A/5B.

## Δομή / σειρά εκτέλεσης
### 01_data_acquisition — λήψη & σύνθεση δείγματος
- `download_master_v5.py` — κύριο: κατεβάζει NREL ResStock 2024 R2 + ComStock 2023 R2 (AMY2018),
  στρωματοποιημένη δειγματοληψία 5.000 κτιρίων, παράγει `merged_master.parquet`.
- `download_all_zones.py`, `download_enduse.py`, `column_selector.py` — βοηθητικά (ζώνες, τελικές χρήσεις, επιλογή στηλών).

### 02_eda_chapter3_figures — Διερευνητική ανάλυση & σχήματα Κεφ. 3
- `phase3_eda*.py` — βασικά EDA στατιστικά ανά ζώνη.
- `fig_sampling_funnel.py` → Σχ 3.1 (funnel δειγματοληψίας). `fig_zones.py`/`fig_maps.py`/`fig_map_*` → χάρτες ζωνών/καυσίμων/CO₂/βαθμοημερών.
- `fig_stock_res.py`/`fig_stock_com.py` → χαρακτηριστικά αποθέματος.
- **fig_v6_residential.py** → Σχ 3.5–3.8 (οικιστικά: καύσιμα, CO₂, ανά ζώνη, χρονικά).
- **fig_v6_commercial.py** + **fig_v6_symmetry.py** → Σχ 3.11–3.15 (εμπορικά, συμμετρικά με οικιστικά· + res χαρακτηριστικά με πίνακα m²).
- **fig_v6_52.py** → Σχ 5.2 (CO₂ ανά σενάριο, decoupling). `fig_emission_heatmap_allzones.py` → θερμικός χάρτης εκπομπών 5 ζωνών.

### 03_ml_chapter4 — Μηχανική Μάθηση (Ψηφιακό Δίδυμο)
- **ch4_run.py** — κύριο pipeline: features (31), χρονολογικός διαχωρισμός 70/15/15 (5B out-of-sample),
  Optuna (30 trials), XGBoost / Random Forest / Naive Seasonal / Prophet, SHAP. Παράγει Πίν 4.1–4.5.
  Αποτελέσματα: XGBoost R²=0,9991 · 5B OOS R²=0,9954.
- `ch4_district_esg.py` → Πίν 4.6 (ESG 5 ζωνών), Πίν 4.7 (climate stress), Σχ 4.1–4.3.
- `ch4_figures.py`, `ch4_pipeline.py`, `build_eval.py` — σχήματα/βοηθητικά ML.

### 04_climate_scenarios_chapter5 — Κλιματικά σενάρια (Copernicus)
- `climate_scenarios_v2.py` — delta scaling επί Copernicus CDS, εναρμόνιση βαθμοημερών (πραγματικές ASHRAE),
  παράγει `scenarios_2x3_baseline.parquet`. `recalibrate_and_regen.py` — επαναβαθμονόμηση.

### 05_esg_social_phase4 — ESG & Κοινωνικός πυλώνας
- `phase4_figures.py` — Σχ 5.1 (μεταβολή κατανάλωσης), τροχιά CO₂ (grid decarbonization
  GRID_DECARB 2018=100%→2100=10%· EF_ELEC ανά ζώνη· EF_GAS/OIL/PROPANE).
- `phase4_esg.py` → ESG dashboard/scores. `phase4_social.py` → energy burden / τρωτότητα.

### tools
- `splice_master.py` — εργαλείο ενσωμάτωσης σχημάτων/κειμένου στο .docx (XML surgery).

## Βασικά επαληθευμένα αποτελέσματα (για γρήγορο έλεγχο)
- Μείγμα καυσίμων οικιστικών: 48,0% ηλεκτρισμός / 47,6% αέριο (συμφωνεί με EIA RECS 2020).
- XGBoost R²=0,9991 · RMSE 0,486 · MAPE 5,94% · 5B OOS R²=0,9954.
- ESG: 4A καλύτερη (B), 2A χειρότερη (E). Climate stress 3A: +1,0% (2040) → +7,9% (2100, SSP5-8.5).
- CO₂ ανά σενάριο: −48% (2040) → −75% (2100) λόγω αποανθρακοποίησης δικτύου.

Επεξεργασία: Αριστοτέλης Ζυγούρης (2026) · NREL ResStock/ComStock · Copernicus CDS.
