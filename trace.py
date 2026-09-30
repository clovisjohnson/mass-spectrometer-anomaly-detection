import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Usage : python trace.py [fichier.csv]
DOSSIER = Path(__file__).parent
fichier = Path(sys.argv[1]) if len(sys.argv) > 1 else DOSSIER / "spectrum_example1.csv"

df = pd.read_csv(fichier)
mz = df.iloc[:, 0].to_numpy()
intensite_log = np.log10(df.iloc[:, 1].to_numpy() + 1)  # +1 pour éviter log(0)

plt.figure(figsize=(12, 6))
plt.plot(mz, intensite_log, color="#16a085", lw=0.5)
plt.title(f"Spectre en échelle logarithmique : {fichier.name}")
plt.xlabel("m/z")
plt.ylabel("log10(Intensité + 1)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
