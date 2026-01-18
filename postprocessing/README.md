# EKG-NaN-Imputation: Postprocessing Repair Net

Implementierung eines **Temporal Convolutional Network (TCN)** zur Imputation fehlender Werte (NaNs) in digitalisierten Papier-EKG-Zeitreihen.

## Problem

- **Input**: 12-Lead-EKG-Zeitreihen vom ECG-Digitizer mit fehlenden Werten (NaNs)
- **Ursachen**: Tracing-Abbrüche, Überlagerungen, schlechte Bildqualität
- **Output**: Reparierte Zeitreihen mit sinnvoll ergänzten fehlenden Segmenten

## Architektur

### Datenformat
- **x_filled**: EKG mit NaNs als 0 gefüllt, Shape: `(12, T)`
- **mask**: Binary Maske (1=vorhanden, 0=fehlt), Shape: `(12, T)`
- **y_target**: Ground Truth (nur zum Training/Evaluation), Shape: `(12, T)`

### Modell
- **Eingabe**: `concat(x_filled, mask)` → Shape: `(24, T)`
- **Modell**: TCN mit exponentiellen Dilationen
- **Ausgabe**: Residual `Δ` → Shape: `(12, T)`
- **Finale Vorhersage**: `y_hat = x_filled + Δ`

### Loss-Funktion
- **Primär**: MSE/Huber Loss **nur auf fehlenden Stellen** (`mask=0`)
- **Sekundär**: Kleiner MSE Loss auf vorhandenen Stellen (`mask=1`) zur Stabilisierung

## Installation

```bash
# Im Root-Verzeichnis des Projekts
cd /home/paulw/projects/ekg_ml

# Virtuelle Umgebung erstellen (nur einmalig)
python3 -m venv venv
source venv/bin/activate

# Dependencies installieren
pip install -r requirements.txt
```

## Workflow: Von Rohdaten zum trainierten Modell

### Schritt 1: Datensplit (60/20/20)

```bash
cd /home/paulw/projects/ekg_ml
source venv/bin/activate

python3 postprocessing/split_data.py \
  --source-dir ../EKG_DATA/train \
  --output-dir postprocessing/data_split \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --test-ratio 0.2 \
  --seed 42
```

**Output:**
```
Gefunden: 7200 EKG-Dateien
Split abgeschlossen!
Train: 4320 Dateien (60%) → postprocessing/data_split/train
Val:   1440 Dateien (20%) → postprocessing/data_split/val
Test:  1440 Dateien (20%) → postprocessing/data_split/test
```

### Schritt 2: Training

```bash
cd /home/paulw/projects/ekg_ml/postprocessing

python3 train.py \
  --train-dir data_split/train \
  --val-dir data_split/val \
  --batch-size 32 \
  --window-size 1000 \
  --overlap 0.5 \
  --damage-rate 0.1 \
  --hidden-channels 64 \
  --num-blocks 4 \
  --learning-rate 0.001 \
  --num-epochs 50 \
  --checkpoint-dir checkpoints
```

**Während Training:** Real-time Outputs für Epoch, Loss, MAE, RMSE
```
Epoch 1/50
  Training...
    Batch 20/100: loss=0.001234
    Batch 40/100: loss=0.001156
    ...
  Validation...
    Loss: 0.001567
    MAE (missing): 0.0456
    RMSE (missing): 0.0789
    ✓ Best model saved: checkpoints/best_model.pt
```

**Nach Training:** Training-History in `checkpoints/training_history.json`

### Schritt 3: Test-Evaluation

```bash
python3 evaluate.py \
  --test-dir data_split/test \
  --checkpoint checkpoints/best_model.pt \
  --output-dir results
```

**Output:**
```
======================================================================
TEST EVALUATION RESULTS
======================================================================

Overall Performance:
  MAE:  0.012345 ± 0.006789
  RMSE: 0.023456 ± 0.012345

Grouped Performance:
  Limb Leads (I, II, III, aVR, aVL, aVF):
    MAE:  0.011234
    RMSE: 0.022345
  Chest Leads (V1-V6):
    MAE:  0.013456
    RMSE: 0.024567

Per-Lead Performance:
  I  : MAE=0.010123, RMSE=0.020234
  II : MAE=0.010456, RMSE=0.021345
  ...
======================================================================

Results saved to: results/test_results.json
```

### Schritt 4: Training fortsetzen (optional)

Falls Training unterbrochen wird:

```bash
python3 train.py \
  --train-dir data_split/train \
  --val-dir data_split/val \
  --num-epochs 100 \
  --resume-checkpoint checkpoints/best_model.pt
```

## Datenformat

Die EKG-CSV-Dateien sollten folgendes Format haben:

```csv
I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6
0.04,0.1,0.06,,,,,,,,,
0.04,0.1,0.06,,,,,,,,,
...
```

**Wichtig**:
- 12 Spalten entsprechend den Standard-Leads
- Reihenfolge: `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`
- Leere Zellen (,,) sind erlaubt und werden intern gehandhabt
- Zeitreihen beliebiger Länge T (werden resampled auf 10000 Samples)

## Hyperparameter

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `--window-size` | 1000 | Zeitfenster in Samples (2s bei 500Hz) |
| `--overlap` | 0.5 | Fenster-Overlap (50%) |
| `--damage-rate` | 0.1 | Anteil synthetischer NaN-Werte (10%) |
| `--hidden-channels` | 64 | TCN Hidden Channels |
| `--num-blocks` | 4 | Anzahl TCN Blöcke |
| `--dropout` | 0.2 | Dropout Rate |
| `--learning-rate` | 0.001 | Adam Lernrate |
| `--batch-size` | 32 | Batch-Größe |
| `--num-epochs` | 50 | Anzahl Trainingsepochen |
| `--loss-type` | 'mse' | Loss-Funktion: 'mse' oder 'huber' |
| `--w-present` | 0.01 | Gewicht für Loss auf vorhandenen Werten |

## Projekt-Struktur

```
postprocessing/
├── data.py              # Datenladen, Windowing, Damage
├── model.py             # TCN Architektur
├── loss.py              # Masked Loss Functions
├── eval.py              # Evaluation & Metriken
├── train.py             # Training Loop
├── evaluate.py          # Test-Set Evaluation
├── split_data.py        # Datensplit (Train/Val/Test)
├── requirements.txt
├── README.md            # Diese Datei
├── checkpoints/         # Modell-Checkpoints (nach Training)
│   ├── best_model.pt
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_2.pt
│   └── training_history.json
├── data_split/          # Split nach split_data.py
│   ├── train/           # 60% der Daten
│   ├── val/             # 20% der Daten
│   └── test/            # 20% der Daten
└── results/             # Test-Ergebnisse
    └── test_results.json
```

## Troubleshooting

### Keine CSV-Dateien gefunden
```
FileNotFoundError: Keine CSV-Dateien in ... gefunden
```
**Lösungen:**
1. Stelle sicher, dass Verzeichnisstruktur korrekt ist: `train/[id]/[id].csv`
2. Alternativ: Flache Struktur mit `train/*.csv` funktioniert auch
3. Überprüfe, dass die Dateien `.csv` Extension haben

### Out of Memory (OOM)
```
CUDA out of memory
```
**Lösungen:**
1. Reduziere `--batch-size` (z.B. 16)
2. Reduziere `--window-size` (z.B. 512)
3. Reduziere `--hidden-channels` (z.B. 32)
4. Erhöhe `--num-workers 0` (falls nicht bereits 0)

### Training konvergiert nicht (Loss bleibt hoch)
**Lösungen:**
1. Erhöhe `--learning-rate` (z.B. 0.01)
2. Reduziere `--learning-rate` (z.B. 0.0001)
3. Erhöhe `--damage-rate` (z.B. 0.2) für schwierigeres Training
4. Prüfe Datenqualität: Sind CSVs valide?
5. Erhöhe `--num-epochs` für längeres Training

### CUDA nicht verfügbar
```
Device: cpu
```
**Info:** Training funktioniert auch auf CPU (aber langsamer)
**Für GPU:** Installiere `torch[cuda]` statt `torch`

## Performance-Tipps

1. **Always activate venv** bevor Training startet
2. **Seed setzen** für Reproduzierbarkeit: `--seed 42` in split_data.py
3. **Größere Datasets** → bessere Performance (Minimum: 1000 EKGs empfohlen)
4. **Monitoring**: Schau auf MAE statt Loss (interpretierbar in physikalischen Einheiten)
5. **Checkpoint-Speicher**: Jedes Checkpoint ~100 MB
6. **GPU Memory**: Bei OOM → batch_size reduzieren (wichtiger als num_workers!)

## Output

### Checkpoints
Nach jedem Epoch werden folgende Dateien gespeichert:
- `checkpoint_epoch_N.pt` — Modell nach Epoch N
- `best_model.pt` — Bestes Modell (nach Validation Loss)
- `training_history.json` — Trainingshistorie (Loss, Metriken)

### Metrics

Das Modell wird anhand folgender Metriken evaluiert:
- **MAE (Mean Absolute Error)** auf fehlenden Stellen
- **RMSE (Root Mean Squared Error)** auf fehlenden Stellen
- **RMSE Global** auf allen Stellen

Auswertung getrennt nach:
- **Limb Leads**: I, II, III, aVR, aVL, aVF
- **Chest Leads**: V1, V2, V3, V4, V5, V6

## Beispiel-Workflow

```python
import torch
from model import build_model
from data import create_dataloader
from eval import ECGMetrics

# 1. Daten laden
train_loader = create_dataloader(
    data_dir='./data/train',
    batch_size=32,
    window_size=1000,
    damage_rate=0.1,
)

# 2. Modell laden
model = build_model(hidden_channels=64, num_blocks=4)

# 3. Batch verarbeiten
batch = next(iter(train_loader))
x_filled = batch['x_filled']
mask = batch['mask']
y_target = batch['y_target']

# 4. Vorhersage
y_hat = model.predict(x_filled, mask)

# 5. Metriken berechnen
mae = ECGMetrics.mae_on_missing(
    y_hat[0].numpy(),
    y_target[0].numpy(),
    mask[0].numpy()
)
print(f"MAE: {mae:.6f}")
```

## Modell-Checkpoints laden

```python
import torch
from model import build_model

# Modell initialisieren
model = build_model()

# Best Model laden
checkpoint = torch.load('./checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Vorhersage
with torch.no_grad():
    y_hat = model.predict(x_filled, mask)
```

## Debugging & Tipps

1. **Daten laden fehlgeschlagen**: 
   - CSV-Datei-Format überprüfen (Spaltenreihenfolge)
   - Pfad muss existierende CSV-Dateien enthalten

2. **CUDA Out of Memory**:
   - `--batch-size` reduzieren
   - `--window-size` reduzieren

3. **Training zu langsam**:
   - `--num-blocks` reduzieren
   - `--hidden-channels` reduzieren
   - `--num-workers` erhöhen (wenn genug RAM)

4. **Loss stagniert**:
   - `--learning-rate` reduzieren
   - `--damage-rate` erhöhen
   - `--dropout` erhöhen

## Referenzen

- Projekt-Anforderungen: `project_requirements.md`
- Paper: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (Bai et al., 2018)
