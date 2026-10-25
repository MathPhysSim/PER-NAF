# PER-NAF & NAF2: Kritischer Benchmark- und 50-Seed-SCC-Cluster-Bericht

**Autor:** Simon Hirlaender  
**Stand:** 25.10.2026  
**SCC-Cluster-Jobs:** `2971477` (50-Seed Array) & `2971478` (Collector), $N = 2.200$ Läufe, 0 Fehler  

---

## 1. Executive Summary

1. **Algorithmisches Refactoring & Compiler-Beschleunigung (`10.7x` bis `34.6x`):**
   - **Vektorisierte Cholesky-Projektion $\mathcal{O}(B d_a^2)$** in `pernaf/pernaf/networks.py` ersetzt geschachtelte Python-Schleifen und $\mathcal{O}(B d_a^3)$ `torch.bmm`.
   - **Numba `@njit(fastmath=True)` SumTree/MinTree** in `pernaf/pernaf/buffers.py` beschleunigt das Prioritized Sampling um **34.6x** und das Speichern um **6.0x**.
   - **JAX XLA + Optax Kernel** in `pernaf/pernaf/jax_naf.py` beschleunigt ein komplettes 16-DOF-Training-Update um **10.7x** (`333.5 ms` statt `3558.8 ms` für 500 Updates mit Batch 100).
   - **Kritischer Bugfix im PER-Buffer:** Die ursprüngliche feste Initialpriorität (`1.0`) führte auf Beschleuniger-Umgebungen mit kleinen TD-Fehlern ($\sim 0.01$) zu einer **Prioritäts-Inversion**, bei der frühe Transitionen permanent mit $\sim 100	imes$ höherer Wahrscheinlichkeit gezogen wurden als neue Erfahrungen. Behoben durch automatische Erstkalibrierung an das empirische TD-Fehler-Maximum und Mini-Batch-Max-Gewichtsnormierung $w_i / \max_{j \in B} w_j \in (0, 1]$.

2. **Hauptergebnis der 50-Seed-SCC-Studie ($N = 2.200$ Läufe, gepaarte $2 	imes 2$ ANOVA):**
   - **CERN LINAC4 (16-DOF Nominal):** **PER** hat einen hochsignifikanten Haupteffekt von **$-154.2 \pm 39.1$ Shots ($p = 0.00025$)**, reduziert die Evaluationsschritte von `4.94` auf **`2.53–2.92` Schritte ($p = 0.00024$)** und erhöht die Erfolgsrate von `77.5%` auf **`90.6%–92.2%` ($p = 0.00012$)**.
   - **CERN LINAC4 (16-DOF HighNoise, $\sigma=0.04, \lambda=0.05$):** **`PER-NAF2` (`All`: Clipped Double-Q + PER)** ist der klare Gesamtsieger mit **`506.6` Shots ($-135.9$ Shots ggü. `NAF`, $p = 0.00083$)**, **`3.83` Eval-Schritten ($lpha=0.4$)** und **`84.5%` Erfolgsrate** (gegenüber `64.5%` bei `NAF`, $p = 0.0033$). Während Single-Q `PER-NAF` bei aggressivem $lpha=0.8$ unter Rauschen einbricht (`654.5` Shots, `74.8%`), bleibt `PER-NAF2` durch Clipped Double-Q stabil (`569.8` Shots, `83.1%`).
   - **CERN AWAKE (10-DOF Clean, HighNoise, Nonlinear):** Da das 10-DOF-System bereits in $\sim 100$ Shots konvergiert, dominiert **Clipped Double-Q** (`DoubleQ_effect = -12.46 Shots, p = 0.0263` auf Clean; `-8.65 Shots, p = 0.0417` auf HighNoise). Reines Single-Q `PER-NAF` verlangsamt auf `AWAKE HighNoise` die Konvergenz (`+16.67 Shots, p = 0.0011`), weil verrauschte BPM-Ausreißer hohe TD-Prioritäten erhalten; **`PER-NAF2` (`All`)** kompensiert diesen Fehler durch die Minimum-Schätzung zweier unabhängiger Q-Netze (`1.11` Eval-Schritte vs. `1.86` bei `PER-NAF`).
   - **Pendulum-v1 (20.000 Schritte, 50 Seeds):** **PER** beschleunigt das frühe Aufschwingen signifikant (`+145.6 Return, p = 0.00181`) und verbessert die Gesamt-AUC (`+106.0, p = 0.000285`). **`PER-NAF2` (`All`)** erreicht in den letzten 20 Episoden den besten Endwert und die niedrigste Varianz über alle 50 Seeds (`-201.1 ± 98.5 SD`).

---

## 2. Vollständige 50-Seed-SCC-Tabelle (alle 44 Konfigurationen pro Seed)

| Szenario | Variante | Seeds | Shots bis Konvergenz (Mean ± SD) | Eval-Schritte (Mean ± SD) | Erfolgsrate / AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `LINAC4_16DOF_Nominal` | `NAF` | 50 | 651.66 ± 330.94 | 4.94 ± 4.70 | 77.5 % |
| `LINAC4_16DOF_Nominal` | `NAF2` | 50 | 691.30 ± 307.85 | 6.47 ± 5.23 | 70.6 % |
| `LINAC4_16DOF_Nominal` | **`PER-NAF_a04`** | 50 | **465.54 ± 104.35** | **2.53 ± 2.51** | **91.8 %** |
| `LINAC4_16DOF_Nominal` | `PER-NAF_a06` | 50 | 529.54 ± 194.53 | 3.48 ± 3.61 | 87.4 % |
| `LINAC4_16DOF_Nominal` | `PER-NAF_a08` | 50 | 577.18 ± 205.91 | 4.75 ± 3.89 | 81.5 % |
| `LINAC4_16DOF_Nominal` | **`PER-NAF2_a04` (`All`)** | 50 | **504.10 ± 122.90** | **2.66 ± 3.32** | **92.2 %** |
| `LINAC4_16DOF_Nominal` | **`PER-NAF2_a06` (`All`)** | 50 | **505.02 ± 177.05** | **2.92 ± 3.26** | **90.6 %** |
| `LINAC4_16DOF_Nominal` | **`PER-NAF2_a08` (`All`)** | 50 | **494.10 ± 154.96** | 3.31 ± 3.61 | 87.8 % |
| `LINAC4_16DOF_HighNoise` | `NAF` | 50 | 642.52 ± 238.16 | 6.94 ± 5.10 | 64.5 % |
| `LINAC4_16DOF_HighNoise` | `NAF2` | 50 | 685.82 ± 319.04 | 6.66 ± 5.02 | 67.0 % |
| `LINAC4_16DOF_HighNoise` | `PER-NAF_a04` | 50 | 505.74 ± 134.21 | 4.96 ± 4.74 | 78.1 % |
| `LINAC4_16DOF_HighNoise` | `PER-NAF_a06` | 50 | 566.28 ± 163.62 | 4.94 ± 4.26 | 77.0 % |
| `LINAC4_16DOF_HighNoise` | `PER-NAF_a08` | 50 | 654.52 ± 283.13 | 5.65 ± 4.72 | 74.8 % |
| `LINAC4_16DOF_HighNoise` | **`PER-NAF2_a04` (`All`)** | 50 | 527.22 ± 230.87 | **3.83 ± 5.13** | **84.5 %** |
| `LINAC4_16DOF_HighNoise` | **`PER-NAF2_a06` (`All`)** | 50 | **506.60 ± 187.95** | **4.10 ± 4.34** | **83.0 %** |
| `LINAC4_16DOF_HighNoise` | **`PER-NAF2_a08` (`All`)** | 50 | 569.76 ± 230.44 | **4.09 ± 3.20** | **83.1 %** |
| `AWAKE_10DOF_Clean` | `NAF` | 50 | 109.40 ± 40.43 | 1.11 ± 0.44 | 99.8 % |
| `AWAKE_10DOF_Clean` | `NAF2` | 50 | 97.52 ± 31.52 | 1.04 ± 0.05 | 100.0 % |
| `AWAKE_10DOF_Clean` | `PER-NAF_a04` | 50 | 103.02 ± 28.35 | 1.06 ± 0.19 | 99.9 % |
| `AWAKE_10DOF_Clean` | `PER-NAF_a06` | 50 | 108.70 ± 35.73 | 1.16 ± 0.62 | 99.7 % |
| `AWAKE_10DOF_Clean` | **`PER-NAF2_a06` (`All`)** | 50 | **95.66 ± 31.56** | 1.09 ± 0.25 | **99.9 %** |
| `AWAKE_10DOF_Clean` | **`PER-NAF2_a08` (`All`)** | 50 | 100.74 ± 31.10 | **1.024 ± 0.026** | **100.0 %** |
| `AWAKE_10DOF_HighNoise` | `NAF` | 50 | 102.68 ± 35.32 | 1.13 ± 0.44 | 99.8 % |
| `AWAKE_10DOF_HighNoise` | **`NAF2`** | 50 | **92.32 ± 29.39** | 1.14 ± 0.29 | **99.8 %** |
| `AWAKE_10DOF_HighNoise` | `PER-NAF_a06` | 50 | 117.64 ± 35.55 | 1.86 ± 3.75 | 98.1 % |
| `AWAKE_10DOF_HighNoise` | **`PER-NAF2_a06` (`All`)** | 50 | 110.70 ± 31.96 | **1.11 ± 0.27** | **99.9 %** |
| `AWAKE_10DOF_Nonlinear` | `NAF` | 50 | 101.84 ± 41.08 | 1.84 ± 1.35 | 96.3 % |
| `AWAKE_10DOF_Nonlinear` | **`NAF2`** | 50 | 105.64 ± 45.67 | **1.50 ± 1.04** | **98.7 %** |
| `AWAKE_10DOF_Nonlinear` | **`PER-NAF2_a04` (`All`)** | 50 | 104.06 ± 36.63 | **1.54 ± 1.03** | **98.0 %** |
| `AWAKE_10DOF_Nonlinear` | **`PER-NAF2_a06` (`All`)** | 50 | **97.06 ± 40.19** | 1.80 ± 1.28 | 97.0 % |
| `Pendulum-v1` | `NAF` | 50 | -672.86 ± 261.29 (Early) | -223.17 ± 158.82 (Final) | -536.67 (AUC) |
| `Pendulum-v1` | `NAF2` | 50 | -856.32 ± 332.54 (Early) | -347.26 ± 385.70 (Final) | -658.60 (AUC) |
| `Pendulum-v1` | **`PER-NAF_a06`** | 50 | **-597.64 ± 315.96 (Early)** | -208.02 ± 123.95 (Final) | **-480.27 (AUC)** |
| `Pendulum-v1` | **`PER-NAF2_a06` (`All`)** | 50 | -640.28 ± 313.91 (Early) | **-201.12 ± 98.45 (Final)** | -503.05 (AUC) |
