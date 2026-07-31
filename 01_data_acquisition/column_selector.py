# -*- coding: utf-8 -*-
# ============================================================================
# column_selector.py
# ----------------------------------------------------------------------------
# ΣΚΟΠΟΣ (Greek explanation):
#   Μείωση διαστατικότητας κατά τη λήψη (dimensionality reduction at ingestion).
#   Τα αρχεία metadata του NREL ResStock/ComStock περιέχουν ~550 στήλες ανά
#   κτίριο. Πολλές ΔΕΝ έχουν προγνωστική αξία για την ενεργειακή πρόβλεψη
#   (αναγνωριστικά, γεωγραφικά ονόματα, σταθερά metadata, ακατέργαστα strings
#   που υπάρχουν ήδη σε αριθμητική μορφή). Αφαιρώντας τες ΠΡΙΝ φορτωθεί ο κορμός
#   των δεδομένων, ελαφρύνουμε το αρχείο και χωράνε ΠΕΡΙΣΣΟΤΕΡΕΣ εγγραφές (κτίρια)
#   στην ίδια μνήμη — κρίσιμο τώρα που ΔΕΝ γίνεται αναβάθμιση υλικού.
#
# ΑΝΑΦΟΡΕΣ (academic basis):
#   - Wilson, E.J.H. et al. (2022), End-Use Load Profiles for the U.S. Building
#     Stock, NREL/TP-5500-80889. (τεκμηρίωση δομής ResStock/ComStock metadata)
#   - Guyon, I. and Elisseeff, A. (2003), "An introduction to variable and
#     feature selection", JMLR, Vol. 3, pp. 1157-1182. (θεωρία feature selection)
#
# ΑΝΑΠΑΡΑΓΩΓΙΜΟΤΗΤΑ: όλες οι επιλογές ελέγχονται από παραμέτρους + RANDOM_SEED.
# ============================================================================

from __future__ import annotations
import argparse
import pandas as pd
import numpy as np

RANDOM_SEED = 42  # για αναπαραγώγιμη δειγματοληψία

# ----------------------------------------------------------------------------
# 1) Κανόνες αναγνώρισης "άχρηστων" στηλών
#    Κάθε keyword δηλώνει στήλη χωρίς προγνωστική αξία (identifier/geo/meta).
#    Τα ονόματα στηλών ResStock ξεκινούν συνήθως με "in." (inputs) ή "out." (outputs).
# ----------------------------------------------------------------------------
ID_KEYWORDS = [
    "_id", "bldg_id", "building_id", "sample", "weight",  # αναγνωριστικά & βάρη
    "puma", "county", "census", "tract", "fips",           # γεωγραφικοί κωδικοί
    "name", "label", "_desc", "address", "street",         # ονόματα/διευθύνσεις
    "metadata_index", "applicability",                     # τεχνικά metadata
]
# Σταθερές/τεχνικές στήλες προσομοίωσης που είναι ίδιες σε ΟΛΕΣ τις εγγραφές
CONSTANT_META_KEYWORDS = [
    "simulation", "upgrade", "version", "release", "schedule_scale",
]

def _matches(col: str, keywords: list[str]) -> bool:
    """Επιστρέφει True αν το όνομα στήλης περιέχει κάποιο keyword (case-insensitive)."""
    c = col.lower()
    return any(kw in c for kw in keywords)

# ----------------------------------------------------------------------------
# 2) Profiling: για κάθε στήλη υπολογίζουμε στατιστικά & προτεινόμενη ενέργεια
# ----------------------------------------------------------------------------
def profile_columns(df: pd.DataFrame,
                    missing_threshold: float = 60.0,
                    cardinality_cap: int = 50) -> pd.DataFrame:
    """
    Δημιουργεί αναφορά ανά στήλη με: dtype, ποσοστό κενών, πλήθος μοναδικών,
    δείκτη σταθερότητας και ΠΡΟΤΕΙΝΟΜΕΝΗ ΕΝΕΡΓΕΙΑ (action).

    Παράμετροι
    ----------
    missing_threshold : στήλες με % κενών > threshold -> DROP
    cardinality_cap   : object στήλες με n_unique < cap -> ENCODE (categorical)
    """
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        n_unique = int(s.nunique(dropna=True))
        missing_pct = float(s.isna().mean() * 100.0)
        is_constant = n_unique <= 1
        dtype = str(s.dtype)

        # --- Λογική απόφασης (κατά σειρά προτεραιότητας) ---
        if is_constant:
            action = "DROP_constant"           # σταθερή -> μηδενική πληροφορία
        elif _matches(col, CONSTANT_META_KEYWORDS):
            action = "DROP_sim_metadata"        # τεχνικό metadata προσομοίωσης
        elif _matches(col, ID_KEYWORDS):
            action = "DROP_identifier"          # αναγνωριστικό/γεωγραφικό/όνομα
        elif missing_pct > missing_threshold:
            action = "DROP_high_missing"        # υπερβολικά κενά
        elif (dtype == "object" or pd.api.types.is_string_dtype(s)) and n_unique <= cardinality_cap:
            action = "ENCODE_categorical"       # κατηγορική -> προς encoding
        elif pd.api.types.is_numeric_dtype(s):
            action = "KEEP_numeric"             # αριθμητικό feature
        else:
            action = "REVIEW_manual"            # χειροκίνητος έλεγχος

        rows.append(dict(column=col, dtype=dtype,
                        missing_pct=round(missing_pct, 1),
                        n_unique=n_unique, is_constant=is_constant,
                        action=action))
    return pd.DataFrame(rows).sort_values(["action", "column"]).reset_index(drop=True)

# ----------------------------------------------------------------------------
# 3) Επιλογή στηλών που κρατάμε (KEEP + ENCODE)
# ----------------------------------------------------------------------------
def select_columns(profile: pd.DataFrame, extra_drops: list[str] | None = None) -> list[str]:
    """Επιστρέφει τη λίστα στηλών προς διατήρηση, με προαιρετικά χειροκίνητα drops."""
    keep = profile["action"].isin(["KEEP_numeric", "ENCODE_categorical"])
    selected = profile.loc[keep, "column"].tolist()
    if extra_drops:
        selected = [c for c in selected if c not in set(extra_drops)]
    return selected

# ----------------------------------------------------------------------------
# 4) Εκτίμηση κέρδους μνήμης (πίνακας "πριν/μετά" για το §3.2 της διπλωματικής)
# ----------------------------------------------------------------------------
def memory_report(df: pd.DataFrame, kept_cols: list[str]) -> dict:
    """Υπολογίζει μνήμη πριν/μετά και εκτίμηση για το πόσες φορές περισσότερα
    κτίρια χωράνε στην ίδια μνήμη μετά τη μείωση στηλών."""
    mem_before = df.memory_usage(deep=True).sum() / 1e6            # MB
    mem_after = df[kept_cols].memory_usage(deep=True).sum() / 1e6  # MB
    reduction_pct = (1 - mem_after / mem_before) * 100 if mem_before else 0.0
    capacity_gain = (mem_before / mem_after) if mem_after else float("inf")
    return dict(
        cols_before=df.shape[1], cols_after=len(kept_cols),
        mem_before_mb=round(mem_before, 2), mem_after_mb=round(mem_after, 2),
        reduction_pct=round(reduction_pct, 1),
        capacity_gain_x=round(capacity_gain, 2),  # π.χ. 6.0x = 6πλάσια κτίρια
    )

# ----------------------------------------------------------------------------
# 5) Main: profiling -> report -> επιλογή -> "slim" αρχείο
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ResStock/ComStock column selector")
    ap.add_argument("input", help="parquet ή csv με metadata (baseline_metadata_only)")
    ap.add_argument("--out-report", default="results/column_profile.csv")
    ap.add_argument("--out-slim", default="results/metadata_slim.parquet")
    ap.add_argument("--missing", type=float, default=60.0)
    args = ap.parse_args()

    # --- Φόρτωση (parquet ή csv) ---
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    print(f"Φορτώθηκε: {df.shape[0]:,} εγγραφές × {df.shape[1]} στήλες")

    # --- Profiling + report ---
    profile = profile_columns(df, missing_threshold=args.missing)
    profile.to_csv(args.out_report, index=False)
    print("\nΣύνοψη ενεργειών ανά στήλη:")
    print(profile["action"].value_counts().to_string())

    # --- Επιλογή & αποθήκευση slim ---
    kept = select_columns(profile)
    mem = memory_report(df, kept)
    print("\nΠΙΝΑΚΑΣ ΠΡΙΝ/ΜΕΤΑ (για §3.2):")
    for k, v in mem.items():
        print(f"  {k:>16}: {v}")

    df[kept].to_parquet(args.out_slim, index=False)
    print(f"\nΑποθηκεύτηκε slim dataset: {args.out_slim}  ({len(kept)} στήλες)")

if __name__ == "__main__":
    main()
