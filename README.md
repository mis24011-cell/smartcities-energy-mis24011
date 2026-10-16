# Smart Cities energy analysis — MSc thesis MIS24011

**Εφαρμογή μεθόδων μηχανικής μάθησης σε δεδομένα μοντέλων προσομοίωσης κτιρίων για την ανάλυση της ενεργειακής κατανάλωσης σε έξυπνες πόλεις: κλιματικά σενάρια, δείκτες ESG και υποστήριξη αποφάσεων**

Διπλωματική εργασία — ΔΠΜΣ στα Πληροφοριακά Συστήματα, Πανεπιστήμιο Μακεδονίας (2026)<br>
Συγγραφέας: Αριστοτέλης Ζυγούρης · Επιβλέπων: Κωνσταντίνος Ταραμπάνης

> **English summary.** Reproducible Python pipeline of the thesis. It uses 5,000 individual buildings from NREL ResStock 2024 R2 and ComStock 2023 R2 (600 residential + 400 commercial in Houston 2A, Atlanta 3A, Baltimore 4A, Chicago 5A and Denver 5B; 43.8 M hourly records, weather AMY2018). It covers hourly forecasting (persistence, seasonal naive, Ridge, Random Forest, XGBoost with per-building scaling; Denver held out), statistical significance (cluster bootstrap and Diebold–Mariano), IPCC AR6 climate scenarios through a separate weather model, the diversity factor, CO₂ per m² and grid-decarbonisation what-if, household energy burden, a composite E+S index with weight sensitivity, and an interactive dashboard. Every number and figure of the thesis is produced by these scripts.

---

## Τι περιέχει

| Φάκελος | Περιεχόμενο |
|---|---|
| `v2/` | Ο κώδικας της ανάλυσης (βήματα 00a–10), `config.py` (όλες οι εξωτερικές τιμές με την πηγή τους), `common.py` |
| `v2/extra/` | Οι δύο χάρτες του Διαγράμματος 3.19 (πόλεις, κλιματικές ζώνες IECC) |
| `data/individual_final/` | Ο κατάλογος των 5.000 κτιρίων του δείγματος (`metadata.parquet`) και ο ωριαίος καιρός AMY2018 των πέντε κομητειών (`weather.parquet`) |
| `results/tables/` | Όλοι οι πίνακες αποτελεσμάτων (CSV) της τελικής εκτέλεσης, που τροφοδοτούν το κείμενο |
| `results/figures/` | Τα διαγράμματα της εργασίας |
| `results/data_checks/` | Οι έλεγχοι ποιότητας (`diagnostics.txt`), τα πλήθη (`data_summary.json`) και η σύγκριση με τα ετήσια σύνολα της NREL (`annual_check.csv`) |
| `dashboard/` | Η διαδραστική κονσόλα (`ΚΟΝΣΟΛΑ_dashboard.html`) — ανοίγει με διπλό κλικ σε οποιονδήποτε φυλλομετρητή |

Τα ωριαία δεδομένα των κτιρίων (μερικά GB) **δεν** περιλαμβάνονται. Κατεβαίνουν από το δημόσιο αποθετήριο OEDI της NREL (Amazon S3, χωρίς λογαριασμό) με το βήμα 00a, για ακριβώς τα ίδια 5.000 κτίρια.

## Εγκατάσταση

Python ≥ 3.10.

```bash
pip install -r requirements.txt
```

Σε Windows με Anaconda: `v2\INSTALL_PACKAGES.bat`.

## Εκτέλεση

Όλα τα βήματα εκτελούνται **από τον κεντρικό φάκελο του repository**, γιατί οι διαδρομές είναι σχετικές (`data/`, `outputs_v2/`).

```bash
# Windows
v2\RUN_ALL.bat
# Linux / macOS
bash v2/run_all.sh
```

Κάθε βήμα γράφει log στο `outputs_v2/log_<βήμα>.txt` και σταματά την εκτέλεση αν αποτύχει κάποιος έλεγχος. Η πλήρης εκτέλεση διαρκεί περίπου 4–5 ώρες σε σύγχρονο υπολογιστή, κυρίως για τη λήψη των δεδομένων και την εκπαίδευση των μοντέλων.

| Βήμα | Τι κάνει | Ενότητα της εργασίας | Κύρια έξοδος |
|---|---|---|---|
| `00a_REDOWNLOAD_v3` | Λήψη των 5.000 κτιρίων από τη NREL, 15λεπτα → ωριαία (σύμβαση «τέλος διαστήματος») | §3.2 | `data/individual_final/buildings_v3.parquet` |
| `00b_REPAIR_DATA` | Μοναδικός κωδικός κτιρίου, εμβαδόν σε m², ευθυγράμμιση καιρού, βάρη αποθέματος, 13 έλεγχοι ποιότητας | §3.3 | `data/v2/*`, `diagnostics.txt` |
| `00c_VERIFY_ANNUAL` | Σύγκριση των ωριαίων αθροισμάτων με τα ετήσια σύνολα της NREL | §3.3.6 | `annual_check.csv` |
| `01_FEATURES_v2` | 19 μεταβλητές χωρίς πληροφορία της ίδιας ώρας, υστερήσεις ανά κτίριο, διαχωρισμός εκπαίδευσης / επικύρωσης / ελέγχου | §3.4–3.5 | `ml_features_v2.parquet` |
| `02_ML_v2` | Persistence, Seasonal Naive, Ridge, Random Forest, XGBoost (Optuna στο σύνολο επικύρωσης, κλιμάκωση ανά κτίριο) | §4.1–4.4, §4.6 | `table_4_4*`, μοντέλα |
| `02c_SEASONAL_BY_TYPE_v2` | Skill ανά εποχή και τύπο κτιρίου | §4.6.3 | `table_4_4d` |
| `03_CLIMATE_v2` | Μοντέλο καιρού χωρίς υστερήσεις, σενάρια IPCC AR6 (SSP2-4.5, SSP5-8.5) | §5.1.2 | `table_5_0*`, `table_5_1` |
| `04_ENERGY_BURDEN_v2` | Ενεργειακή επιβάρυνση νοικοκυριών (όλα τα καύσιμα, τιμές EIA SEDS 2018) | §5.2 | `table_5_3*` |
| `05_EDA_v2` | Διερευνητική ανάλυση: πίνακες και διαγράμματα του Κεφ. 3 | §3.7 | `table_3_*`, `fig_3_*` |
| `06_ESG_DF_DECARB_v2` | Συντελεστής ετεροχρονισμού, σύνθετος δείκτης E+S, σενάρια δικτύου | §4.4.2, §4.5, §5.1.3 | `table_4_5`, `table_4_6*`, `table_5_2` |
| `07_FIGURES_v2` | Διαγράμματα των Κεφ. 4–5 από τους πίνακες | Κεφ. 4–5 | `fig_4_*`, `fig_5_*` |
| `08_SIGNIFICANCE_v2` | Στατιστική σημαντικότητα: bootstrap ανά κτίριο, έλεγχος Diebold–Mariano | §4.4.3 | `table_4_4e/f/g`, `fig_4_2b` |
| `09_DASHBOARD_v2` | Διαδραστική κονσόλα (ένα αρχείο HTML) | §6.2.2, Παράρτημα Α | `dashboard_standalone.html` |
| `10_SUPPLEMENT_v2` | Συμπληρωματικοί υπολογισμοί του κειμένου (προφίλ ανά m², σφάλματα ανά πόλη, νοικοκυριά Q1, δείκτης υπό σενάρια δικτύου) | §4.2.4, §4.4.1, §4.4.5, §5.2.2, §5.4.1 | `suppl_*` |

Οι χάρτες του Διαγράμματος 3.19 (προαιρετικά): `python v2/extra/map_cities.py` και `python v2/extra/map_zones.py` (απαιτούν `basemap`, `shapely`, `eeweather`).

## Αναπαραγωγιμότητα

- Κάθε εξωτερική τιμή (συντελεστές εκπομπών eGRID2018 και EPA, τιμές ενέργειας EIA, ΔT του IPCC, κατώφλια επιβάρυνσης) ορίζεται **μία φορά** στο `config.py`, με την πηγή της.
- Σταθερό seed (42) σε όλες τις τυχαίες επιλογές· 30 δοκιμές Optuna ανά μοντέλο.
- Κάθε πίνακας και διάγραμμα παράγεται από τα αρχεία αποτελεσμάτων· κανένας αριθμός δεν πληκτρολογείται με το χέρι.
- Τα αποτελέσματα της τελικής εκτέλεσης βρίσκονται στον φάκελο `results/`. Νέα εκτέλεση γράφει στο `outputs_v2/`, ώστε να μπορεί να συγκριθεί.

## Δεδομένα και άδειες

- **Δεδομένα:** NREL, *End-Use Load Profiles for the U.S. Building Stock* — ResStock 2024 Release 2 και ComStock 2023 Release 2, Open Energy Data Initiative (OEDI), ελεύθερα διαθέσιμα υπό τις άδειες ResStock και ComStock της NREL (βλ. [Registry of Open Data on AWS](https://registry.opendata.aws/nrel-pds-building-stock/)). Τα αρχεία του φακέλου `data/individual_final/` είναι απόσπασμα αυτών των δεδομένων (μεταδεδομένα και καιρός των 5.000 κτιρίων του δείγματος).
- **Όρια κλιματικών ζωνών** (μόνο για τον χάρτη): πακέτο `eeweather` (OpenEEmeter, Apache-2.0).
- **Κώδικας:** άδεια MIT (αρχείο `LICENSE`).

## Παραπομπή

Ζυγούρης, Α. (2026), *Εφαρμογή μεθόδων μηχανικής μάθησης σε δεδομένα μοντέλων προσομοίωσης κτιρίων για την ανάλυση της ενεργειακής κατανάλωσης σε έξυπνες πόλεις: κλιματικά σενάρια, δείκτες ESG και υποστήριξη αποφάσεων*, Διπλωματική εργασία, ΔΠΜΣ στα Πληροφοριακά Συστήματα, Πανεπιστήμιο Μακεδονίας, Θεσσαλονίκη. Βλ. επίσης `CITATION.cff`.
