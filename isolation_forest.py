"""
Isolation Forest - Détection d'anomalies sur spectres de masse
==============================================================
Pipeline :
1. Chargement des spectres CSV
2. Interpolation sur une grille commune (m/z)
3. Extraction des features (std, sum, skew) sur chaque fenêtre de pic
4. Séparation train / val / test via split_manifest.json
5. Entraînement de l'Isolation Forest sur le jeu d'entraînement
6. Évaluation sur les jeux de validation et de test
7. Export des scores (decision_function) en JSON pour le méta-classifieur
"""

import sys
import time
import json
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score, precision_recall_curve, roc_curve
)
from scipy.interpolate import interp1d
from scipy.stats import skew

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "backup_big_data"

DIRS = {
    "reference":     DATA_DIR / "reference_spectrum",
    "low_intensity": DATA_DIR / "low_intensity",
    "mass_shift":    DATA_DIR / "mass_shift",
    "other":         DATA_DIR / "other",
}

FENETRES_FILE       = SCRIPT_DIR / "fenetres_4.txt"
SPLIT_MANIFEST_FILE = SCRIPT_DIR / "split_manifest.json"
REPORT_FILE         = SCRIPT_DIR / "isolation_forest_report.txt"
SCORES_VAL_JSON     = SCRIPT_DIR / "isolation_forest_scores_val.json"
SCORES_TEST_JSON    = SCRIPT_DIR / "isolation_forest_scores_test.json"

MZ_MIN, MZ_MAX, MZ_STEP = 0.0, 500.0, 0.01

IF_PARAMS = dict(
    n_estimators=200,
    contamination=0.10,
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)

# ---------------------------------------------------------------------------
# UTILITAIRES
# ---------------------------------------------------------------------------

def vstack_safe(arrays):
    valid_arrays = [a for a in arrays if a.shape[0] > 0]
    return np.vstack(valid_arrays) if valid_arrays else np.empty((0, arrays[0].shape[1] if arrays else 0))

def concat_safe(arrays):
    valid_arrays = [a for a in arrays if len(a) > 0]
    return np.concatenate(valid_arrays) if valid_arrays else np.array([])

# ---------------------------------------------------------------------------
# TRAITEMENT SPECTRAL
# ---------------------------------------------------------------------------

def load_spectrum(filepath: Path):
    try:
        df = pd.read_csv(filepath, sep=None, engine="python", header=None,
                         comment="#", skip_blank_lines=True)
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna()
        mz  = df.iloc[:, 0].values.astype(float)
        ity = df.iloc[:, 1].values.astype(float)
        order = np.argsort(mz)
        return mz[order], ity[order]
    except Exception as e:
        print(f"  [WARN] Impossible de charger {filepath.name} : {e}")
        return None, None

def build_common_grid():
    return np.arange(MZ_MIN, MZ_MAX + MZ_STEP, MZ_STEP)

def interpolate_spectrum(mz, ity, grid):
    mask = (grid >= mz[0]) & (grid <= mz[-1])
    result = np.zeros(len(grid), dtype=float)
    if mask.sum() > 0:
        f = interp1d(mz, ity, kind="linear", bounds_error=False, fill_value=0.0)
        result[mask] = f(grid[mask])
    return np.clip(result, 0, None)

def log_transform(spectrum):
    return np.log1p(spectrum)

def load_windows(filepath: Path):
    windows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.replace(",", ".").split(";")
            if len(parts) >= 2:
                try:
                    windows.append((min(float(parts[0]), float(parts[1])), max(float(parts[0]), float(parts[1]))))
                except ValueError:
                    pass
    return windows

def extract_window_features(spectrum, grid, windows):
    features = []
    for (a, b) in windows:
        idx = np.where((grid >= a) & (grid <= b))[0]
        if len(idx) == 0:
            features.extend([0.0, 0.0, 0.0])
            continue
        seg_ity = spectrum[idx]
        std_val  = np.std(seg_ity)
        sum_val  = np.sum(seg_ity)
        skew_val = skew(seg_ity) if std_val > 0 else 0.0
        features.extend([std_val, sum_val, skew_val])
    return np.array(features, dtype=float)

def load_and_process_folder(folder: Path, grid, windows, label: int):
    csvs = list(folder.glob("*.csv")) + list(folder.glob("*.CSV"))
    if not csvs:
        return np.empty((0, len(windows)*3)), np.array([]), []
    X_list, names = [], []
    print(f"  Chargement de {len(csvs)} spectres depuis '{folder.name}'...")
    for fp in sorted(csvs):
        mz, ity = load_spectrum(fp)
        if mz is None: continue
        spec = log_transform(interpolate_spectrum(mz, ity, grid))
        feat = extract_window_features(spec, grid, windows)
        X_list.append(feat)
        names.append(fp.name)

    if not X_list: return np.empty((0, len(windows)*3)), np.array([]), []
    return np.vstack(X_list), np.full(len(X_list), label), np.array(names)

def apply_split(X, y, names, target_stems):
    """Garde uniquement les spectres dont le nom figure dans target_stems."""
    idx = [i for i, n in enumerate(names) if Path(n).stem in target_stems]
    if not idx:
        return np.empty((0, X.shape[1] if len(X.shape)>1 else 0)), np.array([]), np.array([])
    return X[idx], y[idx], names[idx]

# ---------------------------------------------------------------------------
# RAPPORT TXT
# ---------------------------------------------------------------------------

class ReportWriter:
    SEP1 = "=" * 72
    SEP2 = "-" * 72

    def __init__(self):
        self.lines = []

    def _add(self, text=""): self.lines.append(text)
    def title(self, text): self._add(self.SEP1); self._add(f"  {text}"); self._add(self.SEP1)
    def section(self, text): self._add(""); self._add(self.SEP2); self._add(f"  {text}"); self._add(self.SEP2)
    def line(self, text=""): self._add(text)
    def blank(self): self._add("")
    def kv(self, key, value, indent=2): self._add(f"{'':>{indent}}{key:<38}: {value}")
    def table_header(self, cols, widths):
        self._add("  " + "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths)))
        self._add("  " + "  ".join("-" * w for w in widths))
    def table_row(self, cells, widths):
        self._add("  " + "  ".join(f"{str(c):<{w}}" for c, w in zip(cells, widths)))
    def save(self, path: Path):
        with open(path, "w", encoding="utf-8") as f: f.write("\n".join(self.lines) + "\n")
        print(f"  [OK] Rapport sauvegardé : {path.name}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    rw = ReportWriter()

    rw.title("RAPPORT  –  Isolation Forest  –  Détection d'anomalies")
    rw.blank()
    rw.kv("Date d'exécution", datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
    rw.kv("Script", str(Path(__file__).resolve()))

    # 1. Vérifications
    for key, path in DIRS.items():
        if not path.exists():
            print(f"[ERREUR] Dossier introuvable : {path}")
            sys.exit(1)
    if not FENETRES_FILE.exists() or not SPLIT_MANIFEST_FILE.exists():
        print("[ERREUR] Fichier de fenêtres ou split_manifest.json introuvable.")
        sys.exit(1)

    # 2. Chargement du split
    with open(SPLIT_MANIFEST_FILE, "r") as f:
        manifest = json.load(f)
    train_stems = set(manifest.get("train", []))
    val_stems   = set(manifest.get("val", []))
    test_stems  = set(manifest.get("test", []))

    grid    = build_common_grid()
    windows = load_windows(FENETRES_FILE)

    rw.section("CONFIGURATION")
    rw.kv("Grille m/z", f"{grid[0]:.2f} – {grid[-1]:.2f}  (pas {MZ_STEP}, {len(grid)} points)")
    rw.kv("Fenêtres de pics", f"{len(windows)} intervalles  (source : {FENETRES_FILE.name})")
    rw.kv("Features par spectre", f"{len(windows) * 3} (std_val, sum_val, skew_val)")
    rw.kv("Split Manifest", f"Train: {len(train_stems)} | Val: {len(val_stems)} | Test: {len(test_stems)}")
    rw.blank()
    for k, v in IF_PARAMS.items(): rw.kv(f"  IsolationForest.{k}", str(v))

    # 3. Chargement des spectres et répartition
    print("\n--- Chargement et répartition des données ---")

    X_ref_raw, y_ref_raw, names_ref_raw = load_and_process_folder(DIRS["reference"], grid, windows, label=1)
    X_low_raw, y_low_raw, names_low_raw = load_and_process_folder(DIRS["low_intensity"], grid, windows, label=-1)
    X_ms_raw,  y_ms_raw,  names_ms_raw  = load_and_process_folder(DIRS["mass_shift"], grid, windows, label=-1)
    X_oth_raw, y_oth_raw, names_oth_raw = load_and_process_folder(DIRS["other"], grid, windows, label=-1)

    # Répartition train / val / test pour chaque type
    X_ref_tr, y_ref_tr, names_ref_tr = apply_split(X_ref_raw, y_ref_raw, names_ref_raw, train_stems)
    X_ref_val, y_ref_val, names_ref_val = apply_split(X_ref_raw, y_ref_raw, names_ref_raw, val_stems)
    X_ref_test, y_ref_test, names_ref_test = apply_split(X_ref_raw, y_ref_raw, names_ref_raw, test_stems)

    X_low_tr, y_low_tr, names_low_tr = apply_split(X_low_raw, y_low_raw, names_low_raw, train_stems)
    X_low_val, y_low_val, names_low_val = apply_split(X_low_raw, y_low_raw, names_low_raw, val_stems)
    X_low_test, y_low_test, names_low_test = apply_split(X_low_raw, y_low_raw, names_low_raw, test_stems)

    X_ms_tr, y_ms_tr, names_ms_tr = apply_split(X_ms_raw, y_ms_raw, names_ms_raw, train_stems)
    X_ms_val, y_ms_val, names_ms_val = apply_split(X_ms_raw, y_ms_raw, names_ms_raw, val_stems)
    X_ms_test, y_ms_test, names_ms_test = apply_split(X_ms_raw, y_ms_raw, names_ms_raw, test_stems)

    X_oth_tr, y_oth_tr, names_oth_tr = apply_split(X_oth_raw, y_oth_raw, names_oth_raw, train_stems)
    X_oth_val, y_oth_val, names_oth_val = apply_split(X_oth_raw, y_oth_raw, names_oth_raw, val_stems)
    X_oth_test, y_oth_test, names_oth_test = apply_split(X_oth_raw, y_oth_raw, names_oth_raw, test_stems)

    # Matrice d'entraînement
    X_train = vstack_safe([X_ref_tr, X_low_tr, X_ms_tr, X_oth_tr])

    if X_train.shape[0] == 0:
        print("[ERREUR] Aucun spectre trouvé pour l'entraînement.")
        sys.exit(1)

    # 4. Normalisation
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    def safe_sc(X): return scaler.transform(X) if X.shape[0] > 0 else X

    # 5. Entraînement
    print("\n--- Entraînement Isolation Forest ---")
    clf = IsolationForest(**IF_PARAMS)
    clf.fit(X_train_sc)

    rw.section("ENTRAÎNEMENT")
    rw.kv("Spectres d'entraînement totaux", str(X_train_sc.shape[0]))
    rw.kv("Seuil de décision (offset_)", f"{clf.offset_:.6f}")
    rw.line("  Note : score < seuil  →  ANOMALIE  |  score ≥ seuil  →  SAIN")

    # ---------------------------------------------------------------------------
    # FONCTIONS D'ÉVALUATION
    # ---------------------------------------------------------------------------
    def predict(X_sc):
        if X_sc.shape[0] == 0: return np.array([], dtype=int), np.array([])
        return clf.predict(X_sc), clf.score_samples(X_sc)

    def type_metrics(preds, y_true, scores):
        if len(preds) == 0: return {}
        n = len(preds)
        pred_anom = int(np.sum(preds == -1))
        pred_sain = int(np.sum(preds == 1))
        true_label = int(y_true[0])
        correct = pred_sain if true_label == 1 else pred_anom
        return {
            "n": n, "pred_anom": pred_anom, "pred_sain": pred_sain,
            "correct": correct, "rate": correct / n * 100,
            "score_mean": float(scores.mean()), "score_std":  float(scores.std()),
            "score_min":  float(scores.min()),  "score_max":  float(scores.max()),
            "true_label": true_label,
        }

    def write_eval_block(phase_name, X_ref, y_ref, n_ref, X_low, y_low, n_low, X_ms, y_ms, n_ms, X_oth, y_oth, n_oth):
        # Prédictions
        p_ref, s_ref = predict(safe_sc(X_ref))
        p_low, s_low = predict(safe_sc(X_low))
        p_ms,  s_ms  = predict(safe_sc(X_ms))
        p_oth, s_oth = predict(safe_sc(X_oth))

        m_ref = type_metrics(p_ref, y_ref, s_ref)
        m_low = type_metrics(p_low, y_low, s_low)
        m_ms  = type_metrics(p_ms,  y_ms,  s_ms)
        m_oth = type_metrics(p_oth, y_oth, s_oth)

        all_p = concat_safe([p_ref, p_low, p_ms, p_oth])
        all_y = concat_safe([y_ref, y_low, y_ms, y_oth])
        all_s = concat_safe([s_ref, s_low, s_ms, s_oth])

        if len(all_y) == 0:
            rw.section(f"ÉVALUATION : {phase_name.upper()} (Aucune donnée)")
            return all_s, all_y, all_p

        tn, fp, fn, tp = confusion_matrix(all_y, all_p, labels=[1, -1]).ravel()
        acc = (tp + tn) / len(all_y) * 100
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)

        y_bin = (all_y == -1).astype(int)
        try: auc = float(roc_auc_score(y_bin, -all_s))
        except ValueError: auc = None

        rw.section(f"RÉSULTATS PAR TYPE : {phase_name.upper()}")
        rw.blank()
        cw2 = [22, 5, 11, 11, 9, 10, 10, 10, 10]
        rw.table_header(["Type", "N", "Préd.SAIN", "Préd.ANOM", "Taux(%)", "Scr.moy", "Scr.std", "Scr.min", "Scr.max"], cw2)

        type_rows = [
            ("Référence", m_ref), ("Low Intensity", m_low),
            ("Mass Shift", m_ms), ("Other", m_oth)
        ]
        for tname, m in type_rows:
            if not m: rw.table_row([tname] + ["N/A"] * 8, cw2)
            else:
                rw.table_row([
                    tname, m["n"], m["pred_sain"], m["pred_anom"], f"{m['rate']:.1f}",
                    f"{m['score_mean']:.4f}", f"{m['score_std']:.4f}", f"{m['score_min']:.4f}", f"{m['score_max']:.4f}"
                ], cw2)

        rw.section(f"MÉTRIQUES GLOBALES : {phase_name.upper()}")
        rw.blank()
        rw.line(f"  Total évalués : {len(all_y)}")
        rw.line(f"  Faux Positifs (Sain prédit Anom) : {fp}")
        rw.line(f"  Faux Négatifs (Anom non détectée): {fn}")
        rw.kv("Accuracy", f"{acc:.2f}%")
        rw.kv("AUC-ROC", f"{auc:.4f}" if auc else "N/A")

        return all_s, all_y, all_p

    # Évaluation validation
    write_eval_block("Validation", X_ref_val, y_ref_val, names_ref_val,
                                   X_low_val, y_low_val, names_low_val,
                                   X_ms_val, y_ms_val, names_ms_val,
                                   X_oth_val, y_oth_val, names_oth_val)

    # Évaluation test
    test_scores, test_y, test_preds = write_eval_block("Test", X_ref_test, y_ref_test, names_ref_test,
                                                               X_low_test, y_low_test, names_low_test,
                                                               X_ms_test, y_ms_test, names_ms_test,
                                                               X_oth_test, y_oth_test, names_oth_test)

    # Pied de rapport
    rw.blank()
    rw.line(ReportWriter.SEP1)
    rw.line(f"  Fin du rapport  |  généré le {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rw.line(ReportWriter.SEP1)
    rw.save(REPORT_FILE)


    # ---------------------------------------------------------------------------
    # EXPORT JSON POUR LE MÉTA-CLASSIFIEUR
    # ---------------------------------------------------------------------------
    print("\n--- Export JSON des scores (decision_function) ---")

    def export_split_json(X_ref, n_ref, X_low, n_low, X_ms, n_ms, X_oth, n_oth, filepath):
        X_all = vstack_safe([X_ref, X_low, X_ms, X_oth])
        names_all = concat_safe([n_ref, n_low, n_ms, n_oth])

        if len(names_all) > 0:
            X_all_sc = safe_sc(X_all)
            decision_scores = clf.decision_function(X_all_sc)
            scores_dict = {name: round(float(score), 4) for name, score in zip(names_all, decision_scores)}
        else:
            scores_dict = {}

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(scores_dict, f, indent=4)
        print(f"  [OK] Export {filepath.name}")

    export_split_json(X_ref_val, names_ref_val, X_low_val, names_low_val, X_ms_val, names_ms_val, X_oth_val, names_oth_val, SCORES_VAL_JSON)
    export_split_json(X_ref_test, names_ref_test, X_low_test, names_low_test, X_ms_test, names_ms_test, X_oth_test, names_oth_test, SCORES_TEST_JSON)


    # ---------------------------------------------------------------------------
    # VISUALISATIONS (jeu de test)
    # ---------------------------------------------------------------------------
    print("\n--- Génération des graphiques (jeu de test) ---")

    # Scores individuels du jeu de test
    _, s_ref_test = predict(safe_sc(X_ref_test))
    _, s_low_test = predict(safe_sc(X_low_test))
    _, s_ms_test  = predict(safe_sc(X_ms_test))
    _, s_oth_test = predict(safe_sc(X_oth_test))

    colors = {
        "Référence (Test)": "#4CAF50",
        "Low Intensity":    "#FF9800",
        "Mass Shift":       "#F44336",
        "Other":            "#9C27B0",
    }
    datasets = [
        ("Référence (Test)", s_ref_test, y_ref_test),
        ("Low Intensity",    s_low_test, y_low_test),
        ("Mass Shift",       s_ms_test,  y_ms_test),
        ("Other",            s_oth_test, y_oth_test),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Isolation Forest – Évaluation sur le jeu de TEST (contamination={IF_PARAMS['contamination']})", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    for dname, sc, _ in datasets:
        if len(sc) > 0: ax.hist(sc, bins=30, alpha=0.6, label=dname, color=colors[dname], density=True)
    ax.axvline(x=clf.offset_, color="black", linestyle="--", lw=1.5, label=f"Seuil ({clf.offset_:.4f})")
    ax.set_xlabel("Score d'anomalie"); ax.set_ylabel("Densité")
    ax.set_title("Distribution des scores par type"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    bd = [sc for _, sc, _ in datasets if len(sc) > 0]
    bl = [n  for n, sc, _ in datasets if len(sc) > 0]
    if bd:
        bp = ax.boxplot(bd, labels=bl, patch_artist=True)
        for patch, lbl in zip(bp["boxes"], bl): patch.set_facecolor(colors[lbl]); patch.set_alpha(0.7)
    ax.axhline(y=clf.offset_, color="black", linestyle="--", lw=1.5, label="Seuil")
    ax.set_ylabel("Score"); ax.set_title("Boxplot des scores"); ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=15)

    ax = axes[0, 2]
    try: auc_test = float(roc_auc_score((test_y == -1).astype(int), -test_scores))
    except (ValueError, TypeError): auc_test = None

    if auc_test is not None:
        fpr_c, tpr_c, _ = roc_curve((test_y == -1).astype(int), -test_scores)
        ax.plot(fpr_c, tpr_c, color="#2196F3", lw=2, label=f"AUC = {auc_test:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Courbe ROC Test"); ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "AUC N/A", ha="center", va="center", transform=ax.transAxes); ax.set_title("Courbe ROC Test")

    ax = axes[1, 0]
    if auc_test is not None:
        prec_c, rec_c, _ = precision_recall_curve((test_y == -1).astype(int), -test_scores)
        ax.plot(rec_c, prec_c, color="#E91E63", lw=2)
        ax.set_xlabel("Recall"); ax.set_ylabel("Précision"); ax.set_title("Courbe Précision-Rappel Test")
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes); ax.set_title("Courbe Précision-Rappel Test")

    ax = axes[1, 1]
    x_off = 0
    for dname, sc, _ in datasets:
        if len(sc) == 0: continue
        ax.scatter(np.arange(x_off, x_off + len(sc)), sc, c=colors[dname], s=10, alpha=0.7, label=dname)
        x_off += len(sc)
    ax.axhline(y=clf.offset_, color="black", linestyle="--", lw=1.5, label="Seuil")
    ax.set_xlabel("Index"); ax.set_ylabel("Score"); ax.set_title("Scores individuels Test"); ax.legend(fontsize=8)

    ax = axes[1, 2]
    if len(test_y) > 0:
        tn, fp, fn, tp = confusion_matrix(test_y, test_preds, labels=[1, -1]).ravel()
        cm_arr = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Prédit Sain", "Prédit Anom."])
        ax.set_yticklabels(["Réel Sain", "Réel Anom."])
        for i in range(2):
            for j in range(2):
                color = "white" if cm_arr[i, j] > cm_arr.max() / 2 else "black"
                ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", fontsize=16, color=color)
    ax.set_title("Matrice de confusion Test")

    plt.tight_layout()
    fig_path = SCRIPT_DIR / "isolation_forest_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  [OK] Graphiques globaux sauvegardés : {fig_path.name}")

    plt.show()
    print("\n[DONE]")

if __name__ == "__main__":
    main()
