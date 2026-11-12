# Mise à jour du dépôt — réconciliation README / résultats

Ton dépôt GitHub contient déjà les **nouveaux résultats** (`outputs/results.json`,
`outputs/metrics_summary.csv`, graphiques régénérés) mais l'ancien README, et il
lui manque des fichiers essentiels. Ce dossier contient les 7 fichiers corrigés
à copier dans ta copie locale (`/Users/vannickyakam/Desktop/DL Based Stock Trading`
— adapte le chemin si besoin).

## Contenu et rôle de chaque fichier

| Fichier | Pourquoi |
|---|---|
| `README.md` | Tableau des résultats réécrit depuis `outputs/results.json` (LSTM +67.6 %, etc.), fenêtre de test Nov 2025 → Nov 2026, nouveau nom de repo, narration honnête (dépendance à la fenêtre, lecture sceptique du Sharpe 4.26) |
| `docs/methodology.md` | Dates/split à jour (2022-01-03 → 2026-11-12, 976/245), provenance du checkpoint LSTM, nouvelle §7 « Result variability and reproducibility » avec la comparaison des deux fenêtres |
| `.gitignore` | Corrige `models/` → `/models/` (le package `src/dl_trading/models/` était exclu du dépôt GitHub !) et permet de committer `artifacts/` |
| `pyproject.toml` | Les 3 URLs pointent vers le nouveau nom de repo (et la branche `master`) |
| `CONTRIBUTING.md` | URL de clone à jour |
| `requirements.txt` | `tensorflow-cpu` n'existe pas sur macOS arm64 : marqueurs d'environnement ajoutés |
| `.github/workflows/ci.yml` | Le workflow CI avait disparu du nouveau repo (badge CI cassé) — ruff + pytest sur Python 3.10/3.11/3.12 |

## Application (commandes)

```bash
cd "/Users/vannickyakam/Desktop/DL Based Stock Trading"

# 1. Copier les fichiers corrigés (depuis ce dossier décompressé)
cp README.md .gitignore pyproject.toml CONTRIBUTING.md requirements.txt .
cp docs/methodology.md docs/
mkdir -p .github/workflows && cp .github/workflows/ci.yml .github/workflows/

# 2. Supprimer le fichier vide oublié à la racine
rm "réécrit"

# 3. CRUCIAL — committer le package models (absent de GitHub, le repo cloné ne peut pas tourner)
git add src/dl_trading/models

# 4. Publier les modèles pré-entraînés (le README annonce qu'ils sont inclus)
git add artifacts

# 5. Committer les fichiers corrigés
git add README.md docs/methodology.md .gitignore pyproject.toml CONTRIBUTING.md requirements.txt .github/workflows/

# 6. Optionnel — rafraîchir le vieux snapshot JSON (il date de l'ancien modèle : -3.0 %)
dl-trading backtest --symbol AMZN --strategy lstm --output-json outputs/AMZN_lstm.json
git add outputs/

git commit -m "Sync README with regenerated results; fix .gitignore (models/), restore CI workflow"
git push
```

## Remarques

- Le badge CI restera gris (« no status ») jusqu'au premier run du workflow,
  c'est-à-dire juste après le push.
- `docs/assets/` et `outputs/results.json` / `metrics_summary.csv` sont déjà
  à jour dans ton repo — rien à refaire de ce côté.
- Si tu préfères **ne pas publier** `artifacts/` (poids ou confidentialité du
  checkpoint), ne fais pas l'étape 4 et préviens-moi : le README dit
  « ships with pre-trained models », il faudra reformuler 2-3 phrases.
- Qui écrit quoi (pour mémoire) :
  - `outputs/results.json` + `outputs/metrics_summary.csv` → uniquement
    `python scripts/generate_report.py` ;
  - `outputs/AMZN_<stratégie>.json` → `dl-trading backtest` **uniquement**
    quand on passe `--output-json <chemin>` ;
  - `outputs/max_drawdown/…` → suivi de record du backtester ;
  - `quickstart.py` → PNG uniquement, jamais de JSON.
