# Training Convergence Issues - Analysis & Solutions

## 🔴 Identifizierte Probleme

### 1. **Zero Initialization des Output Layer** ✅ FIXED
**Problem:** Der Output-Layer wurde mit Nullen initialisiert:
```python
nn.init.zeros_(self.output_proj.weight)
nn.init.zeros_(self.output_proj.bias)
```
Dies führt dazu, dass das Modell am Anfang immer `Δ = 0` ausgibt, was zu sehr schwachen Gradienten führt.

**Lösung:** Kaiming-Uniform Initialization mit kleiner Skalierung verwenden.

---

### 2. **Fehlende Residual Connections** ✅ FIXED
**Problem:** Das TCN-Netzwerk ohne Skip-Connections kann unter "Vanishing Gradients" leiden.

**Lösung:** Residual Connection nach der Eingabe-Projektion hinzugefügt:
```python
residual_x = x  # Nach input_proj
for block in self.tcn_blocks:
    x = block(x)
x = x + residual_x  # Skip Connection
```

---

### 3. **Learning Rate zu hoch** ✅ FIXED
**Problem:** Default `lr=1e-3` ist zu aggressiv für ein Residual-Netzwerk.

**Lösung:** 
- Learning Rate auf `5e-4` reduziert
- `AdamW` statt `Adam` (bessere Regularisierung)
- `ReduceLROnPlateau` mit `patience=3` (aggressivere Reduktion)
- `min_lr=1e-6` (verhindert zu starke Reduktion)

---

### 4. **Loss-Gewichtung unbalanciert** ✅ FIXED
**Problem:** `w_present = 0.01` war zu groß:
- Zu viel Fokus auf vorhandene Werte
- Zu wenig Fokus auf fehlende Werte (Hauptziel)

**Lösung:** `w_present = 0.001` (10x kleiner) für stärkeren Fokus auf fehlende Werte.

---

## 📊 Empfehlung zum Debugging

Führen Sie das folgende Debug-Skript aus:
```bash
python debug_convergence.py
```

Dieses überprüft:
- **Data Quality:** Sind ausreichend fehlende Werte vorhanden? (>5% empfohlen)
- **Gradient Flow:** Fließen Gradienten durch alle Layer?
- **Model Output Scale:** Sind die Residual-Ausgaben klein und variabel?

---

## 🎯 Nächste Schritte zum Training

### Option 1: Mit neuem Setup trainieren
```bash
python train.py \
    --train-dir ./data_split/train \
    --val-dir ./data_split/val \
    --batch-size 16 \
    --num-epochs 50 \
    --learning-rate 5e-4 \
    --hidden-channels 64 \
    --num-blocks 4 \
    --dropout 0.2
```

### Option 2: Mit besseren Hyperparametern
Falls immer noch Probleme:
```bash
python train.py \
    --train-dir ./data_split/train \
    --val-dir ./data_split/val \
    --batch-size 32 \
    --num-epochs 100 \
    --learning-rate 2e-4 \
    --weight-decay 1e-4 \
    --hidden-channels 128 \
    --num-blocks 5 \
    --dropout 0.3 \
    --grad-clip-norm 1.0
```

---

## 🔍 Andere potenzielle Ursachen (zu überprüfen)

Falls immer noch nicht konvergiert:

### A. **Daten-Problem**
- Zu wenig fehlende Werte in den synthetischen Daten
- Signal-Amplituden zu klein
- Schlechte Daten-Qualität

**Fix:**
```bash
python train.py --damage-rate 0.2 --damage-blocks-min 2
```

### B. **Modell-Komplexität**
- Netzwerk zu klein/groß
- Zu viel/wenig Dropout

**Fix:**
```bash
python train.py --hidden-channels 128 --num-blocks 5 --dropout 0.3
```

### C. **Numerische Instabilität**
- NaN/Inf in Gradienten
- Overflow/Underflow in Loss-Berechnung

**Fix:**
```bash
python train.py --grad-clip-norm 1.0 --learning-rate 1e-4
```

---

## 📈 Was man in der Trainingsausgabe erwarten sollte

### Gutes Training (konvergiert):
```
Epoch 1/50
  Training...
    Loss: 8.234e-02
  Validation...
    Loss: 7.956e-02
    MAE (missing): 0.045234
    RMSE (missing): 0.056782

Epoch 2/50
  Training...
    Loss: 7.123e-02  ← Loss sinkt
  Validation...
    Loss: 6.890e-02  ← Loss sinkt
```

### Schlechtes Training (konvergiert nicht):
```
Epoch 1/50
  Training...
    Loss: 5.234e+00  ← Viel zu groß
  Validation...
    Loss: 5.123e+00  ← Kein Fortschritt

Epoch 2/50
  Training...
    Loss: 5.198e+00  ← Keine Verbesserung
```

---

## ✅ Summary der Fixes

| Problem | Fix | Datei |
|---------|-----|-------|
| Zero initialization | Kaiming initialization | `model.py` |
| Vanishing gradients | Residual connections | `model.py` |
| Learning rate | 1e-3 → 5e-4 + AdamW | `train.py` |
| Loss unbalanciert | w_present: 0.01 → 0.001 | `train.py` |
| LR Scheduler | patience: 5 → 3 | `train.py` |

