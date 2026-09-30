import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DOSSIER = Path(__file__).parent

# (fichier, couleur matplotlib)
FICHIERS = [
    (DOSSIER / "spectrum_example1.csv", "g"),
    (DOSSIER / "spectrum_example2.csv", "r"),
]

plt.figure(figsize=(12, 7))

for path, couleur in FICHIERS:
    if not path.exists():
        print(f"Fichier introuvable : {path.name}")
        continue

    df = pd.read_csv(path, sep=None, engine="python")
    mz = df.iloc[:, 0].to_numpy()
    intensite_log = np.log10(df.iloc[:, 1].to_numpy() + 1)

    plt.plot(mz, intensite_log, label=path.name, color=couleur, lw=0.6, alpha=0.8)

plt.title("Comparaison de spectres (échelle logarithmique)", fontsize=14)
plt.xlabel("m/z")
plt.ylabel("log10(Intensité + 1)")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
