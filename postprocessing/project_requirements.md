**Projekt-Zusammenfassung / Referenz für Coding-Agent**

**Ziel**
Implementierung eines *lern­basierten Postprocessing-Moduls* zur **Imputation fehlender Werte (NaNs)** in digitalisierten Papier-EKG-Zeitreihen.
Der bestehende **ECG-Digitizer wird als Blackbox** betrachtet und **nicht verändert**.

---

## Problemdefinition

* Input: Vom Digitizer extrahierte **12-Lead-EKG-Zeitreihen** mit **fehlenden Werten (NaNs)**.
* Ursache der NaNs: typische Digitizer-Fehler (Tracing-Abbrüche, Überlagerungen, schlechte Bildqualität).
* Output: **Reparierte Zeitreihen**, bei denen **nur die fehlenden Segmente** sinnvoll ergänzt wurden.
* Ground Truth: Saubere digitale 12-Lead-EKGs liegen vor und werden **nur für Training/Evaluation** genutzt.

---

## Scope (bewusst eingeschränkt)

* **Nur NaN-Imputation**, keine:

  * Rauschunterdrückung
  * Gain-/Baseline-Korrektur
  * Perspektiv- oder Bildverarbeitung
* Fokus ausschließlich auf **einen Pipeline-Schritt** nach der Digitalisierung.
* Keine medizinische Diagnose, nur **Signalrekonstruktion**.

---

## Datenannahmen

* Format pro Sample:

  * `x`: `(C=12, T)` Digitizer-Output mit NaNs
  * `y`: `(12, T)` Ground-Truth-Zeitreihe ohne NaNs
* Zusätzliche Ableitung:

  * `mask m = ~isnan(x)` → `1 = vorhanden`, `0 = fehlt`
* NaNs in `x` werden für das Modell durch `0` ersetzt, Information steckt in `m`.

---

## Modellannahmen

* **Modelltyp**: Temporal Convolutional Network (TCN)
* **Eingabe**: `concat(x_filled, m)` → `(24, T)`
* **Ausgabe**: Residual `Δ` → `(12, T)`
* **Finale Vorhersage**:
  `y_hat = x_filled + Δ`
* Motivation: Modell soll **nur korrigieren**, nicht neu halluzinieren.

---

## Trainingsstrategie

* Training auf echten Digitizer-Outputs **oder** synthetisch beschädigter Ground Truth (nur NaNs).
* **Loss nur auf fehlenden Stellen**:

  * Primärer Loss: MSE oder Huber
  * Optional: sehr kleiner Zusatz-Loss auf vorhandenen Stellen zur Stabilisierung.
* Windowing:

  * Zeitfenster (z. B. 4 s) mit Overlap, um mehr Trainingssamples zu erzeugen.

---

## Evaluation

* Metriken:

  * RMSE / MAE **nur auf NaN-Stellen**
  * optional zusätzlich globaler RMSE (darf sich nicht verschlechtern)
* Auswertung getrennt nach:

  * Limb Leads
  * Chest Leads (V1–V6)
* Qualitativ:

  * Overlays: Input (mit NaNs) vs. Prediction vs. Ground Truth

---

## Projektarchitektur (minimal)

* Eigenständiges Git-Repository (nicht im Digitizer-Repo).
* Digitizer-Outputs werden **nur gelesen**, keine Code-Abhängigkeit.
* Kernmodule:

  * `data.py` – Laden von `x`, `y`, Maskenerzeugung, Windowing
  * `model.py` – TCN-Repair-Net
  * `loss.py` – Masked Loss
  * `train.py` – Trainingsloop
  * `eval.py` – Metriken & Plots

---

## Nicht-Ziele

* Kein End-to-End Bild→Signal
* Keine Erweiterung des Digitizers
* Keine komplexen medizinischen Regeln
* Keine Generalisierung über andere Datensätze hinaus

---

## Leitgedanke

> *Das Modell soll lernen, wie ein vollständiges 12-Lead-EKG zeitlich konsistent aussieht,
> und diese Struktur nutzen, um lokal fehlende Werte im Digitizer-Output zu ergänzen.*

Diese Zusammenfassung dient als **verbindliche Referenz** für alle Coding-Entscheidungen.
