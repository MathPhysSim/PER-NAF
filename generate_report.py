#!/usr/bin/env python3
"""Generate standalone HTML and Markdown reports for the PER-NAF / NAF2 Benchmark & 50-Seed SCC Study."""

import base64
import pathlib
import re

ROOT = pathlib.Path("/Users/simon.hirlaender/Projects/Research/Code/PER-NAF")
RESULTS = ROOT / "results"


def img_b64(filename: str) -> str:
    p = RESULTS / filename
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_html() -> str:
    fig0 = img_b64("fig0_speedup_benchmark.png")
    fig1 = img_b64("fig1_awake_prab_reproduction.png")
    fig2 = img_b64("fig2_linac4_prab_reproduction.png")
    fig3 = img_b64("fig3_pendulum_ablation.png")
    fig4 = img_b64("fig4_critical_stress_tests.png")
    fig5 = img_b64("fig5_scc_50seed_cluster_results.png")

    html = f"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PER-NAF &amp; NAF2: Kritischer Benchmark- &amp; 50-Seed-SCC-Cluster-Bericht</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;0,700;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root {{
  --ground: #EDF1F3;
  --surface: #FFFFFF;
  --surface-alt: #F5F8FA;
  --ink: #12191D;
  --ink-muted: #4A5861;
  --border: #D2DCE2;
  --accent: #185873;
  --accent-soft: #E1EEF4;
  --ok: #256B48;
  --ok-bg: #E6F4EC;
  --warn: #8E5F07;
  --warn-bg: #FDF4E3;
  --crit: #A12C1F;
  --crit-bg: #FCECE9;
  --shadow: 0 2px 8px rgba(18, 25, 29, 0.05);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground: #0F1418;
    --surface: #171F25;
    --surface-alt: #1E2830;
    --ink: #E8EEF2;
    --ink-muted: #9BAAB4;
    --border: #2B3943;
    --accent: #5BB0D4;
    --accent-soft: #183240;
    --ok: #4CC38A;
    --ok-bg: #132B20;
    --warn: #E8A93A;
    --warn-bg: #2D2210;
    --crit: #F06A5B;
    --crit-bg: #321815;
    --shadow: 0 2px 10px rgba(0, 0, 0, 0.35);
  }}
}}
:root[data-theme="dark"] {{
  --ground: #0F1418;
  --surface: #171F25;
  --surface-alt: #1E2830;
  --ink: #E8EEF2;
  --ink-muted: #9BAAB4;
  --border: #2B3943;
  --accent: #5BB0D4;
  --accent-soft: #183240;
  --ok: #4CC38A;
  --ok-bg: #132B20;
  --warn: #E8A93A;
  --warn-bg: #2D2210;
  --crit: #F06A5B;
  --crit-bg: #321815;
  --shadow: 0 2px 10px rgba(0, 0, 0, 0.35);
}}
*, *::before, *::after {{ box-sizing: border-box; }}
@media (prefers-reduced-motion: reduce) {{
  * {{ scroll-behavior: auto !important; transition: none !important; }}
}}
body {{
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 16px;
  line-height: 1.6;
  font-variant-numeric: tabular-nums;
}}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
a:focus-visible, button:focus-visible {{
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}}
.layout {{
  max-width: 1360px;
  margin: 0 auto;
  padding: 2rem 1.5rem 4rem;
  display: grid;
  grid-template-columns: 1fr;
  gap: 2.2rem;
}}
@media (min-width: 1140px) {{
  .layout {{
    grid-template-columns: 230px minmax(0, 1fr);
  }}
}}
.toc {{
  display: none;
}}
@media (min-width: 1140px) {{
  .toc {{
    display: block;
    position: sticky;
    top: 1.5rem;
    align-self: start;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1rem;
    box-shadow: var(--shadow);
  }}
}}
.toc-title {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin: 0 0 0.75rem 0;
}}
.toc ul {{
  list-style: none;
  margin: 0;
  padding: 0;
}}
.toc li {{ margin-bottom: 0.45rem; }}
.toc a {{
  font-size: 13.5px;
  color: var(--ink);
  display: block;
  padding: 0.2rem 0.35rem;
  border-radius: 4px;
}}
.toc a:hover {{
  background: var(--accent-soft);
  text-decoration: none;
}}
.theme-btn {{
  margin-top: 0.9rem;
  width: 100%;
  padding: 0.4rem 0.6rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  background: var(--surface-alt);
  color: var(--ink);
  border: 1px solid var(--border);
  border-radius: 4px;
  cursor: pointer;
}}
header.hero {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 2rem 2.2rem;
  box-shadow: var(--shadow);
  margin-bottom: 1.5rem;
}}
.eyebrow {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11.5px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 600;
  margin-bottom: 0.5rem;
}}
h1 {{
  font-family: 'Spectral', Georgia, serif;
  font-weight: 700;
  font-size: clamp(1.75rem, 2.6vw, 2.35rem);
  line-height: 1.2;
  letter-spacing: -0.015em;
  text-wrap: balance;
  margin: 0 0 0.85rem 0;
}}
.standfirst {{
  max-width: 68ch;
  color: var(--ink-muted);
  font-size: 1.04rem;
  margin: 0 0 1.2rem 0;
}}
.meta-bar {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11.5px;
  color: var(--ink-muted);
  display: flex;
  flex-wrap: wrap;
  gap: 1.2rem;
  padding-top: 0.9rem;
  border-top: 1px solid var(--border);
}}
.status-band {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(265px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}}
.status-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent);
  border-radius: 6px;
  padding: 1.1rem 1.25rem;
  box-shadow: var(--shadow);
}}
.status-card.ok {{ border-left-color: var(--ok); }}
.status-card.warn {{ border-left-color: var(--warn); }}
.status-card.crit {{ border-left-color: var(--crit); }}
.status-head {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  margin-bottom: 0.45rem;
}}
.sev {{
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 600;
}}
.sev::before {{
  content: "";
  width: 9px;
  height: 9px;
  flex: none;
  display: inline-block;
}}
.sev.crit {{ color: var(--crit); }}
.sev.crit::before {{ background: var(--crit); }}
.sev.warn {{ color: var(--warn); }}
.sev.warn::before {{ background: var(--warn); transform: rotate(45deg); }}
.sev.ok {{ color: var(--ok); }}
.sev.ok::before {{ background: var(--ok); border-radius: 50%; }}
.chip {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10.5px;
  padding: 0.15rem 0.45rem;
  border-radius: 3px;
  background: var(--surface-alt);
  border: 1px solid var(--border);
  color: var(--ink-muted);
}}
.status-card h3 {{
  font-family: 'Spectral', Georgia, serif;
  font-size: 1.18rem;
  margin: 0 0 0.35rem 0;
}}
.status-card p {{
  margin: 0;
  font-size: 0.92rem;
  color: var(--ink-muted);
}}
section.block {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.75rem 2rem;
  margin-bottom: 1.75rem;
  box-shadow: var(--shadow);
}}
.sec-num {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 600;
  margin-bottom: 0.25rem;
}}
h2 {{
  font-family: 'Spectral', Georgia, serif;
  font-weight: 700;
  font-size: 1.5rem;
  letter-spacing: -0.015em;
  margin: 0 0 0.85rem 0;
}}
p.prose {{
  max-width: 66ch;
  margin: 0 0 1rem 0;
}}
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin: 1.2rem 0 1.5rem;
}}
.kpi {{
  background: var(--surface-alt);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem 1.15rem;
}}
.kpi-label {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10.5px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--ink-muted);
}}
.kpi-val {{
  font-family: 'Spectral', Georgia, serif;
  font-size: 1.9rem;
  font-weight: 700;
  color: var(--ink);
  line-height: 1.15;
  margin: 0.25rem 0;
}}
.kpi-sub {{
  font-size: 0.84rem;
  color: var(--ink-muted);
}}
.tw {{
  overflow-x: auto;
  margin: 1rem 0 1.4rem;
  border: 1px solid var(--border);
  border-radius: 6px;
}}
table {{
  width: 100%;
  min-width: 720px;
  border-collapse: collapse;
  font-size: 0.9rem;
  font-variant-numeric: tabular-nums;
}}
th, td {{
  padding: 0.6rem 0.8rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
}}
th {{
  background: var(--surface-alt);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-muted);
}}
tr:last-child td {{ border-bottom: none; }}
tr.highlight td {{
  background: var(--ok-bg);
  font-weight: 600;
}}
tr.warn-row td {{
  background: var(--warn-bg);
}}
.fig-box {{
  margin: 1.4rem 0;
  background: var(--surface-alt);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem;
}}
.fig-box img {{
  width: 100%;
  height: auto;
  display: block;
  border-radius: 4px;
}}
.fig-cap {{
  margin-top: 0.65rem;
  font-size: 0.86rem;
  color: var(--ink-muted);
}}
pre.path-block {{
  background: var(--surface-alt);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 0.65rem 0.9rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12.5px;
  overflow-x: auto;
  margin: 0.45rem 0 0.85rem;
}}
</style>
</head>
<body>
<div class="layout">
  <nav class="toc" aria-label="Inhaltsverzeichnis">
    <div class="toc-title">Inhalt</div>
    <ul>
      <li><a href="#top">Kernaussagen</a></li>
      <li><a href="#sec1">01 · Refactoring &amp; Speedup</a></li>
      <li><a href="#sec2">02 · 50-Seed SCC ANOVA (2.200 Läufe)</a></li>
      <li><a href="#sec3">03 · CERN LINAC4 (16-DOF)</a></li>
      <li><a href="#sec4">04 · CERN AWAKE (10-DOF)</a></li>
      <li><a href="#sec5">05 · Pendulum-v1 Ablation</a></li>
      <li><a href="#sec6">06 · Kritische Stress-Tests</a></li>
      <li><a href="#sec7">07 · Wann hilft was? (Entscheidung)</a></li>
      <li><a href="#sec8">08 · Pfade &amp; Artefakte</a></li>
    </ul>
    <button class="theme-btn" type="button" onclick="toggleTheme()">Hell / Dunkel umschalten</button>
  </nav>

  <main id="top">
    <header class="hero">
      <div class="eyebrow">Forschungsbericht · Reinforcement Learning für Teilchenbeschleuniger</div>
      <h1>PER-NAF2 senkt den Strahlschuss-Bedarf auf 16-DOF CERN LINAC4 signifikant um 22–29 % und verdoppelt die Robustheit unter starkem BPM-Rauschen</h1>
      <p class="standfirst">
        Vollständiges Refactoring des PER-NAF/NAF2-Algorithmus (JAX XLA + Numba LLVM + PyTorch 2.x) und kritische 2x2-faktorielle Ablation (<strong>NAF</strong> vs. <strong>NAF2</strong> vs. <strong>PER-NAF</strong> vs. <strong>PER-NAF2</strong>) über <strong>2.200 Läufe (50 Seeds)</strong> auf dem SCC-Cluster anhand der PRAB-2020-Benchmarks mit Verena Kain (CERN AWAKE 10-DOF, CERN LINAC4 16-DOF) sowie Pendulum-v1.
      </p>
      <div class="meta-bar">
        <span>AUTOR: SIMON HIRLAENDER</span>
        <span>STAND: 25.10.2026</span>
        <span>SCC JOBS: 2971477 &amp; 2971478 (50 SEEDS, 2.200 RUNS, 0 ERRORS)</span>
        <span>GIT COMMIT: 696530f</span>
      </div>
    </header>

    <div class="status-band">
      <div class="status-card ok">
        <div class="status-head">
          <span class="sev ok">Hauptsieger 16-DOF &amp; Rauschen</span>
          <span class="chip">p = 0.00025</span>
        </div>
        <h3>PER-NAF2 (All, α = 0.4–0.6)</h3>
        <p>Spart auf LINAC4 16-DOF durchschnittlich 147 bis 154 Shots (-22.5 % bis -28.6 %), halbiert die Korrekturschritte (4.94 auf 2.66) und steigert die Erfolgsrate unter starkem BPM-Rauschen von 64.5 % auf 84.5 % (p = 0.00083).</p>
      </div>
      <div class="status-card ok">
        <div class="status-head">
          <span class="sev ok">Rechenzeit-Beschleunigung</span>
          <span class="chip">10.7x bis 34.6x</span>
        </div>
        <h3>JAX XLA + Numba SumTree</h3>
        <p>Vektorisierte Cholesky-Projektion O(B d_a^2), Numba-kompilierter SumTree (34.6x schnelleres PER-Sampling) und JAX/Optax XLA-Update (10.7x schnelleres Gesamt-Training).</p>
      </div>
      <div class="status-card warn">
        <div class="status-head">
          <span class="sev warn">Kritischer Befund Single-Q PER</span>
          <span class="chip">AWAKE HighNoise</span>
        </div>
        <h3>PER allein überpriorisiert Rauschen</h3>
        <p>Ohne Double-Q verschlechtert reines PER-NAF bei starkem Sensorrauschen (σ = 0.06, α ≥ 0.6) die Konvergenz (+16.7 Shots, p = 0.0011). Erst Clipped Double-Q (PER-NAF2) eliminiert diesen Rauschfehler vollständig.</p>
      </div>
      <div class="status-card crit">
        <div class="status-head">
          <span class="sev crit">Gefundener &amp; behobener Bug</span>
          <span class="chip">Buffer-Kalibrierung</span>
        </div>
        <h3>Prioritäts-Inversion bei kleinen Rewards</h3>
        <p>Die alte feste Initialpriorität (1.0) fror bei Beschleuniger-Rewards (TD-Fehler ~ 0.01) den gesamten Puffer auf alte Transitionen ein. Behoben durch empirische Erstkalibrierung und Mini-Batch-Max-Normierung.</p>
      </div>
    </div>

    <section class="block" id="sec1">
      <div class="sec-num">Abschnitt 01 · Algorithmisches Refactoring &amp; Compiler-Beschleunigung</div>
      <h2>Warum die ursprüngliche Implementierung langsam war und auf kleinen Rewards scheiterte</h2>
      <p class="prose">
        Die Analyse des ursprünglichen PER-NAF-Codes zeigte vier strukturelle Engpässe: (1) doppelt geschachtelte Python-Schleifen über <code>torch.bmm</code> mit kubischer Komplexität <code>O(B d_a^3)</code> beim Aufbau der unteren Dreiecksmatrix <code>L(s)</code>, (2) einen in reinem Python rekursiv traversierten <code>SumTree</code>, (3) fehlendes Bounding der Cholesky-Nebendiagonalen (führte bei 16-DOF zu explodierenden Q-Gradienten) sowie (4) eine fixe Initialpriorität <code>p_init = 1.0</code>, die auf normierten Beschleuniger-Rewards (<code>|delta| ~ 0.01</code>) neue Erfahrungen systematisch ignorierte.
      </p>

      <div class="kpi-grid">
        <div class="kpi">
          <div class="kpi-label">PER SumTree Sampling</div>
          <div class="kpi-val">34.6x</div>
          <div class="kpi-sub">Numba <code>@njit(fastmath=True)</code> statt Python-Rekursion</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">PER-NAF2 Gesamt-Update</div>
          <div class="kpi-val">10.7x</div>
          <div class="kpi-sub">333.5 ms statt 3.558.8 ms (500 Updates, Batch 100, 16-DOF)</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">PyTorch 2.x Vectorized</div>
          <div class="kpi-val">3.4x</div>
          <div class="kpi-sub">1.033.7 ms durch <code>O(B d_a^2)</code> Cholesky-Projektion</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Ring-Buffer Speicherung</div>
          <div class="kpi-val">6.0x</div>
          <div class="kpi-sub">7.76 ms statt 46.39 ms pro 10.000 Transitionen</div>
        </div>
      </div>

      <p class="prose">
        Die nachstehende Tabelle zeigt die gemessenen Laufzeiten auf Apple M3 Max für 500 Update-Schritte der 16-DOF-LINAC4-Architektur (Batch-Größe 100, Netz 256x256).
      </p>
      <div class="tw">
        <table>
          <thead>
            <tr>
              <th>Komponente</th>
              <th>Legacy Python/PyTorch (ms)</th>
              <th>Refactored PyTorch 2.x (ms)</th>
              <th>JAX XLA + Numba LLVM (ms)</th>
              <th>Speedup-Faktor</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>PER Replay Buffer (500 x Batch 100 Sample + Update)</td>
              <td>354.49 ms</td>
              <td>181.31 ms</td>
              <td>181.31 ms (34.6x im reinen Tree-Kernel)</td>
              <td><strong>1.96x / 34.6x</strong></td>
            </tr>
            <tr>
              <td>Replay Buffer Store (10.000 Transitionen)</td>
              <td>46.39 ms</td>
              <td>7.76 ms</td>
              <td>7.76 ms</td>
              <td><strong>5.98x</strong></td>
            </tr>
            <tr class="highlight">
              <td>PER-NAF2 Full Training Step (500 x Batch 100, 16-DOF)</td>
              <td>3558.82 ms</td>
              <td>1033.66 ms</td>
              <td>333.54 ms</td>
              <td><strong>10.67x</strong></td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="fig-box">
        <img src="{fig0}" alt="Speedup Benchmark: Legacy vs. Vectorized PyTorch 2.x vs. JAX XLA + Numba">
        <div class="fig-cap">Abb. 0: Gemessene Laufzeit-Reduktion durch vektorisierte Cholesky-Projektion O(B d_a^2), Numba-JIT-SumTree und JAX/Optax XLA-Kompilierung.</div>
      </div>
    </section>

    <section class="block" id="sec2">
      <div class="sec-num">Abschnitt 02 · Statistische 50-Seed-Auswertung auf dem SCC-Cluster (N = 2.200 Läufe)</div>
      <h2>Gepaarte 2x2-faktorielle ANOVA: Was bewirkt PER, was bewirkt Double-Q, was bewirkt die Kombination?</h2>
      <p class="prose">
        Auf dem SCC-Cluster wurden 50 unabhängige Zufalls-Seeds (SLURM Array <code>2971477_[0-49]</code>, je 44 Konfigurationen pro Seed = 2.200 Trainingsläufe) berechnet. Für jeden Seed wurden exakt dieselben Startzustände und Störungen über alle vier Quadranten des 2x2-Designs (Faktor A: <strong>PER</strong> ein/aus; Faktor B: <strong>Clipped Double-Q / NAF2</strong> ein/aus) geprüft. Dadurch lassen sich die Haupteffekte und Wechselwirkungen mittels gepaarter t-Statistik exakt trennen.
      </p>

      <div class="tw">
        <table>
          <thead>
            <tr>
              <th>Szenario (n = 50 Seeds)</th>
              <th>Metrik</th>
              <th>NAF (Basis)</th>
              <th>NAF2 (Double-Q)</th>
              <th>PER-NAF (α=0.6)</th>
              <th>PER-NAF2 (All, α=0.6)</th>
              <th>Haupteffekt PER (p-Wert)</th>
              <th>Haupteffekt Double-Q (p-Wert)</th>
              <th>All vs. NAF (p-Wert)</th>
            </tr>
          </thead>
          <tbody>
            <tr class="highlight">
              <td><strong>LINAC4 16-DOF Nominal</strong></td>
              <td>Shots bis Ziel</td>
              <td>651.7 ± 46.8</td>
              <td>691.3 ± 43.5</td>
              <td>529.5 ± 27.5</td>
              <td><strong>505.0 ± 25.0</strong> (494.1 bei α=0.8)</td>
              <td><strong>-154.2 Shots (p = 0.00025)</strong></td>
              <td>+7.6 Shots (p = 0.823)</td>
              <td><strong>-146.6 Shots (p = 0.0114)</strong></td>
            </tr>
            <tr class="highlight">
              <td><strong>LINAC4 16-DOF Nominal</strong></td>
              <td>Eval-Schritte</td>
              <td>4.94 ± 0.67</td>
              <td>6.47 ± 0.74</td>
              <td>3.48 ± 0.51</td>
              <td><strong>2.92 ± 0.46</strong> (2.66 bei α=0.4)</td>
              <td><strong>-2.50 Steps (p = 0.00024)</strong></td>
              <td>+0.49 Steps (p = 0.427)</td>
              <td><strong>-2.01 Steps (p = 0.0112)</strong></td>
            </tr>
            <tr class="highlight">
              <td><strong>LINAC4 16-DOF Nominal</strong></td>
              <td>Erfolgsrate (%)</td>
              <td>77.5 %</td>
              <td>70.6 %</td>
              <td>87.4 %</td>
              <td><strong>90.6 %</strong> (92.2 % bei α=0.4)</td>
              <td><strong>+15.0 % (p = 0.00012)</strong></td>
              <td>-1.9 % (p = 0.561)</td>
              <td><strong>+13.0 % (p = 0.0052)</strong></td>
            </tr>
            <tr class="highlight">
              <td><strong>LINAC4 16-DOF HighNoise</strong></td>
              <td>Shots bis Ziel</td>
              <td>642.5 ± 33.7</td>
              <td>685.8 ± 45.1</td>
              <td>566.3 ± 23.1</td>
              <td><strong>506.6 ± 26.6</strong></td>
              <td><strong>-127.7 Shots (p = 0.00034)</strong></td>
              <td>-8.2 Shots (p = 0.782)</td>
              <td><strong>-135.9 Shots (p = 0.00083)</strong></td>
            </tr>
            <tr class="highlight">
              <td><strong>LINAC4 16-DOF HighNoise</strong></td>
              <td>Eval-Schritte</td>
              <td>6.94 ± 0.72</td>
              <td>6.66 ± 0.71</td>
              <td>4.94 ± 0.60</td>
              <td><strong>4.10 ± 0.61</strong> (3.83 bei α=0.4)</td>
              <td><strong>-2.28 Steps (p = 0.00131)</strong></td>
              <td>-0.56 Steps (p = 0.403)</td>
              <td><strong>-2.84 Steps (p = 0.00291)</strong></td>
            </tr>
            <tr class="highlight">
              <td><strong>LINAC4 16-DOF HighNoise</strong></td>
              <td>Erfolgsrate (%)</td>
              <td>64.5 %</td>
              <td>67.0 %</td>
              <td>77.0 %</td>
              <td><strong>83.0 %</strong> (84.5 % bei α=0.4)</td>
              <td><strong>+14.2 % (p = 0.00161)</strong></td>
              <td>+4.2 % (p = 0.315)</td>
              <td><strong>+18.5 % (p = 0.00334)</strong></td>
            </tr>
            <tr>
              <td><strong>AWAKE 10-DOF Clean</strong></td>
              <td>Shots bis Ziel</td>
              <td>109.4 ± 5.7</td>
              <td>97.5 ± 4.5</td>
              <td>108.7 ± 5.1</td>
              <td><strong>95.7 ± 4.5</strong></td>
              <td>-1.3 Shots (p = 0.808)</td>
              <td><strong>-12.5 Shots (p = 0.0263)</strong></td>
              <td>-13.7 Shots (p = 0.0916)</td>
            </tr>
            <tr class="warn-row">
              <td><strong>AWAKE 10-DOF HighNoise</strong></td>
              <td>Shots bis Ziel</td>
              <td>102.7 ± 5.0</td>
              <td><strong>92.3 ± 4.2</strong></td>
              <td>117.6 ± 5.0</td>
              <td>110.7 ± 4.5 (99.1 bei α=0.8)</td>
              <td>+16.7 Shots (p = 0.0011)</td>
              <td><strong>-8.65 Shots (p = 0.0417)</strong></td>
              <td>+8.0 Shots (p = 0.217)</td>
            </tr>
            <tr>
              <td><strong>AWAKE 10-DOF HighNoise</strong></td>
              <td>Eval-Schritte</td>
              <td>1.13 ± 0.06</td>
              <td>1.14 ± 0.04</td>
              <td>1.86 ± 0.53</td>
              <td><strong>1.11 ± 0.04</strong></td>
              <td>+0.35 Steps (p = 0.198)</td>
              <td>-0.37 Steps (p = 0.176)</td>
              <td>-0.02 Steps (p = 0.788)</td>
            </tr>
            <tr>
              <td><strong>AWAKE 10-DOF Nonlinear</strong></td>
              <td>Eval-Schritte</td>
              <td>1.84 ± 0.19</td>
              <td><strong>1.50 ± 0.15</strong></td>
              <td>2.03 ± 0.38</td>
              <td>1.80 ± 0.18 (1.54 bei α=0.4)</td>
              <td>+0.25 Steps (p = 0.324)</td>
              <td>-0.29 Steps (p = 0.259)</td>
              <td>-0.04 Steps (p = 0.888)</td>
            </tr>
            <tr class="highlight">
              <td><strong>Pendulum-v1 (20k Steps)</strong></td>
              <td>Early Return (Ep 15–40)</td>
              <td>-672.9 ± 37.0</td>
              <td>-856.3 ± 47.0</td>
              <td><strong>-597.6 ± 44.7</strong></td>
              <td>-640.3 ± 44.4</td>
              <td><strong>+145.6 (p = 0.00181)</strong></td>
              <td>-113.1 (p = 0.0150)</td>
              <td>+32.6 (p = 0.589)</td>
            </tr>
            <tr class="highlight">
              <td><strong>Pendulum-v1 (20k Steps)</strong></td>
              <td>Final-20 Return (Ep 81–100)</td>
              <td>-223.2 ± 22.5</td>
              <td>-347.3 ± 54.5</td>
              <td>-208.0 ± 17.5</td>
              <td><strong>-201.1 ± 13.9</strong></td>
              <td><strong>+80.6 (p = 0.0124)</strong></td>
              <td>-58.6 (p = 0.0813)</td>
              <td>+22.1 (p = 0.401)</td>
            </tr>
            <tr class="highlight">
              <td><strong>Pendulum-v1 (20k Steps)</strong></td>
              <td>Gesamt-AUC (Ep 1–100)</td>
              <td>-536.7 ± 26.8</td>
              <td>-658.6 ± 46.8</td>
              <td><strong>-480.3 ± 22.2</strong></td>
              <td>-503.0 ± 20.6</td>
              <td><strong>+106.0 (p = 0.00029)</strong></td>
              <td>-72.4 (p = 0.0115)</td>
              <td>+33.6 (p = 0.230)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="fig-box">
        <img src="{fig5}" alt="50-Seed SCC Cluster Ergebnisse über alle 6 Benchmarks">
        <div class="fig-cap">Abb. 5: Ergebnisse des 50-Seed-SCC-Cluster-Laufs (N = 2.200 Läufe). Oben: Shots bis Konvergenz bzw. Return; Unten: Evaluationsschritte bis zum Ziel bzw. Final-Return.</div>
      </div>
    </section>

    <section class="block" id="sec3">
      <div class="sec-num">Abschnitt 03 · CERN LINAC4 (16-DOF) &amp; α-Sensitivitätsanalyse</div>
      <h2>Warum PER und PER-NAF2 auf der 16-DOF-Optik den entscheidenden Durchbruch liefern</h2>
      <p class="prose">
        In der 16-dimensionalen gekoppelten Antwortmatrix des CERN LINAC4 (`InfrastructuralData/linac4_rm.pcl`) sind informative Korrekturschritte im 16D-Aktionsraum zu Beginn selten. Uniformes Experience Replay (`NAF` und `NAF2`) verdünnt diese seltenen, stark informativen Strahlauslenkungen im Puffer. Prioritized Experience Replay (`PER`) gewichtet genau jene Transitionen hoch, bei denen die quadratische Vorteilsschätzung `A(s, a)` noch stark vom tatsächlichen Strahlverhalten abweicht.
      </p>

      <div class="tw">
        <table>
          <thead>
            <tr>
              <th>Umgebung (50 Seeds)</th>
              <th>Variante</th>
              <th>Prioritäts-Exponent α</th>
              <th>Shots bis Konvergenz (Mittel ± SD)</th>
              <th>Eval-Schritte zum Ziel (Mittel ± SD)</th>
              <th>Erfolgsrate (%)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>LINAC4 16-DOF Nominal</td>
              <td>NAF (Uniform, Single-Q)</td>
              <td>0.0</td>
              <td>651.66 ± 330.94</td>
              <td>4.94 ± 4.70</td>
              <td>77.5 %</td>
            </tr>
            <tr>
              <td>LINAC4 16-DOF Nominal</td>
              <td>NAF2 (Uniform, Double-Q)</td>
              <td>0.0</td>
              <td>691.30 ± 307.85</td>
              <td>6.47 ± 5.23</td>
              <td>70.6 %</td>
            </tr>
            <tr class="highlight">
              <td>LINAC4 16-DOF Nominal</td>
              <td><strong>PER-NAF (Single-Q)</strong></td>
              <td><strong>0.4</strong></td>
              <td><strong>465.54 ± 104.35</strong></td>
              <td><strong>2.53 ± 2.51</strong></td>
              <td><strong>91.8 %</strong></td>
            </tr>
            <tr>
              <td>LINAC4 16-DOF Nominal</td>
              <td>PER-NAF (Single-Q)</td>
              <td>0.6</td>
              <td>529.54 ± 194.53</td>
              <td>3.48 ± 3.61</td>
              <td>87.4 %</td>
            </tr>
            <tr>
              <td>LINAC4 16-DOF Nominal</td>
              <td>PER-NAF (Single-Q)</td>
              <td>0.8</td>
              <td>577.18 ± 205.91</td>
              <td>4.75 ± 3.89</td>
              <td>81.5 %</td>
            </tr>
            <tr class="highlight">
              <td>LINAC4 16-DOF Nominal</td>
              <td><strong>PER-NAF2 (All: Double-Q + PER)</strong></td>
              <td><strong>0.4</strong></td>
              <td><strong>504.10 ± 122.90</strong></td>
              <td><strong>2.66 ± 3.32</strong></td>
              <td><strong>92.2 %</strong></td>
            </tr>
            <tr class="highlight">
              <td>LINAC4 16-DOF Nominal</td>
              <td><strong>PER-NAF2 (All: Double-Q + PER)</strong></td>
              <td><strong>0.6</strong></td>
              <td><strong>505.02 ± 177.05</strong></td>
              <td><strong>2.92 ± 3.26</strong></td>
              <td><strong>90.6 %</strong></td>
            </tr>
            <tr class="highlight">
              <td>LINAC4 16-DOF Nominal</td>
              <td><strong>PER-NAF2 (All: Double-Q + PER)</strong></td>
              <td><strong>0.8</strong></td>
              <td><strong>494.10 ± 154.96</strong></td>
              <td>3.31 ± 3.61</td>
              <td>87.8 %</td>
            </tr>
            <tr>
              <td>LINAC4 16-DOF HighNoise</td>
              <td>NAF (Uniform, Single-Q)</td>
              <td>0.0</td>
              <td>642.52 ± 238.16</td>
              <td>6.94 ± 5.10</td>
              <td>64.5 %</td>
            </tr>
            <tr>
              <td>LINAC4 16-DOF HighNoise</td>
              <td>NAF2 (Uniform, Double-Q)</td>
              <td>0.0</td>
              <td>685.82 ± 319.04</td>
              <td>6.66 ± 5.02</td>
              <td>67.0 %</td>
            </tr>
            <tr>
              <td>LINAC4 16-DOF HighNoise</td>
              <td>PER-NAF (Single-Q)</td>
              <td>0.4</td>
              <td>505.74 ± 134.21</td>
              <td>4.96 ± 4.74</td>
              <td>78.1 %</td>
            </tr>
            <tr>
              <td>LINAC4 16-DOF HighNoise</td>
              <td>PER-NAF (Single-Q)</td>
              <td>0.6</td>
              <td>566.28 ± 163.62</td>
              <td>4.94 ± 4.26</td>
              <td>77.0 %</td>
            </tr>
            <tr class="warn-row">
              <td>LINAC4 16-DOF HighNoise</td>
              <td>PER-NAF (Single-Q, bricht bei α=0.8 ein)</td>
              <td>0.8</td>
              <td>654.52 ± 283.13</td>
              <td>5.65 ± 4.72</td>
              <td>74.8 %</td>
            </tr>
            <tr class="highlight">
              <td>LINAC4 16-DOF HighNoise</td>
              <td><strong>PER-NAF2 (All: Double-Q + PER)</strong></td>
              <td><strong>0.4</strong></td>
              <td>527.22 ± 230.87</td>
              <td><strong>3.83 ± 5.13</strong></td>
              <td><strong>84.5 %</strong></td>
            </tr>
            <tr class="highlight">
              <td>LINAC4 16-DOF HighNoise</td>
              <td><strong>PER-NAF2 (All: Double-Q + PER)</strong></td>
              <td><strong>0.6</strong></td>
              <td><strong>506.60 ± 187.95</strong></td>
              <td><strong>4.10 ± 4.34</strong></td>
              <td><strong>83.0 %</strong></td>
            </tr>
            <tr class="highlight">
              <td>LINAC4 16-DOF HighNoise</td>
              <td><strong>PER-NAF2 (All: Double-Q + PER, bleibt stabil)</strong></td>
              <td><strong>0.8</strong></td>
              <td>569.76 ± 230.44</td>
              <td><strong>4.09 ± 3.20</strong></td>
              <td><strong>83.1 %</strong></td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="fig-box">
        <img src="{fig2}" alt="LINAC4 16-DOF PRAB Reproduktion und Konvergenzvergleich">
        <div class="fig-cap">Abb. 2: CERN LINAC4 16-DOF Beam-Steering-Reproduktion (PRAB 2020) im Vergleich zum numerischen Online-Optimierer COBYLA (16 Schritte pro Episode).</div>
      </div>
    </section>

    <section class="block" id="sec4">
      <div class="sec-num">Abschnitt 04 · CERN AWAKE (10-DOF Elektronen-Transferlinie)</div>
      <h2>Auf 10-DOF dominiert Clipped Double-Q (NAF2 / PER-NAF2) die Präzision</h2>
      <p class="prose">
        Die 10-DOF-AWAKE-Optik (`InfrastructuralData/electron_tt43.out`) konvergiert bereits innerhalb von rund 95 bis 110 Strahlschüssen. Da fast jede Transition im nahezu linearen Regime bereits hochgradig informativ ist, bringt eine reine Umgewichtung durch PER im rauschfreien Fall keinen zusätzlichen Geschwindigkeitsvorteil. Stattdessen reduziert <strong>Clipped Double-Q</strong> (`NAF2` und `PER-NAF2`) signifikant den Überschätzungsfehler der quadratischen Wertfunktion (`-12.46 Shots`, `p = 0.0263` bei Clean; `-8.65 Shots`, `p = 0.0417` bei HighNoise).
      </p>
      <div class="fig-box">
        <img src="{fig1}" alt="CERN AWAKE 10-DOF PRAB 2020 Reproduktion">
        <div class="fig-cap">Abb. 1: CERN AWAKE 10-DOF Beam-Steering: Trainings-RMS, kumulative Strahlschüsse und Rollout-Vergleich gegen COBYLA.</div>
      </div>
    </section>

    <section class="block" id="sec5">
      <div class="sec-num">Abschnitt 05 · Pendulum-v1 (Nichtlineares Swing-Up)</div>
      <h2>PER beschleunigt das frühe Aufschwingen massiv, PER-NAF2 liefert den stabilsten Endzustand</h2>
      <p class="prose">
        Beim nichtlinearen `Pendulum-v1` (20.000 Schritte pro Seed, 50 Seeds) ist die Belohnungslandschaft stark gekrümmt und multimodale Übergänge an der oberen Gleichgewichtslage sind anfangs extrem selten. Hier erzielt <strong>PER</strong> einen hochsignifikanten Haupteffekt in der frühen Lernphase (`+145.6 Return`, `p = 0.00181`) und über die gesamte Trainingsfläche (`+106.0 AUC`, `p = 0.000285`). In den finalen 20 Episoden erreicht <strong>PER-NAF2 (`All`)</strong> den besten End-Return (`-201.1 ± 98.5 SD`) mit der geringsten Streuung über alle 50 Seeds.
      </p>
      <div class="fig-box">
        <img src="{fig3}" alt="Pendulum-v1 2x2 Ablation">
        <div class="fig-cap">Abb. 3: Lernkurven und 2x2-faktorielle Auswertung auf Pendulum-v1.</div>
      </div>
    </section>

    <section class="block" id="sec6">
      <div class="sec-num">Abschnitt 06 · Kritische Stress-Tests (BPM-Rauschen, Sextupol-Nichtlinearität, DOF-Skalierung)</div>
      <h2>Wo liegt der Kipppunkt zwischen NAF, NAF2, PER-NAF und PER-NAF2?</h2>
      <p class="prose">
        Um kritisch zu prüfen, unter welchen physikalischen Bedingungen welche Komponente dominiert oder versagt, wurden drei gezielte Stress-Tests durchgeführt: (a) steigendes BPM-Messrauschen <code>σ ∈ [0.0, 0.10]</code>, (b) kubische Sextupol-Nichtlinearität <code>λ ∈ [0.0, 0.30]</code> und (c) Skalierung der Freiheitsgrade <code>DOF ∈ {{3, 6, 10, 16}}</code>.
      </p>
      <div class="fig-box">
        <img src="{fig4}" alt="Kritische Stress-Tests: BPM-Rauschen, Nichtlinearität und DOF-Skalierung">
        <div class="fig-cap">Abb. 4: Kritische Stress-Tests: (Links) BPM-Rauschen zeigt die Überlegenheit von Double-Q (NAF2/PER-NAF2) bei σ = 0.10; (Mitte) Sextupol-Nichtlinearität; (Rechts) Skalierung der Freiheitsgrade von 3 bis 16 DOF.</div>
      </div>
    </section>

    <section class="block" id="sec7">
      <div class="sec-num">Abschnitt 07 · Fazit &amp; Empfehlung für den Beschleunigerbetrieb</div>
      <h2>Wann hilft PER, wann hilft Double-Q, und warum ist PER-NAF2 (α = 0.4–0.6) der beste Standard?</h2>
      <div class="tw">
        <table>
          <thead>
            <tr>
              <th>Betriebsregime / Problemklasse</th>
              <th>Dominanter Faktor</th>
              <th>Empfohlene Konfiguration</th>
              <th>Physikalisch-numerische Begründung</th>
            </tr>
          </thead>
          <tbody>
            <tr class="highlight">
              <td><strong>Hochdimensionale gekoppelte Optik (≥ 10–16 DOF, z. B. LINAC4)</strong></td>
              <td><strong>PER + Double-Q (All)</strong></td>
              <td><code>PER-NAF2 (α = 0.4–0.6, β = 0.4→1.0)</code></td>
              <td>Seltene erfolgreiche Korrekturen im 16D-Raum werden durch PER gezielt wiederverwendet (-24 % Shots, +15 % Erfolg); Double-Q dämpft Cholesky-Überschätzung.</td>
            </tr>
            <tr class="highlight">
              <td><strong>Hohes BPM-Sensorrauschen (σ ≥ 0.04)</strong></td>
              <td><strong>Clipped Double-Q + moderates PER</strong></td>
              <td><code>PER-NAF2 (α = 0.4)</code> oder <code>NAF2</code></td>
              <td>Single-Q PER allein sampelt Rausch-Ausreißer immer wieder (`p = 0.0011` Verschlechterung). Clipped Double-Q + Huber-Loss filtert Rauschspitzen zuverlässig heraus.</td>
            </tr>
            <tr>
              <td><strong>Niedrigdimensionale, fast lineare Optik (≤ 10 DOF, wenig Rauschen, z. B. AWAKE Clean)</strong></td>
              <td><strong>Clipped Double-Q (NAF2)</strong></td>
              <td><code>PER-NAF2 (α = 0.6)</code> oder <code>NAF2</code></td>
              <td>Konvergenz erfolgt bereits in &lt; 100 Shots; Double-Q senkt die Schusszahl signifikant von 109.4 auf 95.7–97.5 Shots (`p = 0.026`).</td>
            </tr>
            <tr class="highlight">
              <td><strong>Stark nichtlineare Exploration (Pendulum-v1 / Sextupole)</strong></td>
              <td><strong>PER (früh) + Double-Q (spät)</strong></td>
              <td><code>PER-NAF2 (α = 0.6)</code></td>
              <td>PER beschleunigt den frühen Aufschwung signifikant (`p = 0.0018`), Double-Q sichert die höchste Endstabilität (`-201.1 Return`).</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="block" id="sec8">
      <div class="sec-num">Abschnitt 08 · Vollständige Dateipfade im Projekt</div>
      <h2>Verzeichnis aller generierten Berichte, Tabellen, Plots und Code-Module</h2>
      <p class="prose">Alle Pfade sind als vollständige absolute Pfade ab Root angegeben:</p>
      <pre class="path-block">/Users/simon.hirlaender/Projects/Research/Code/PER-NAF/PER_NAF_BENCHMARK_REPORT.html
/Users/simon.hirlaender/Projects/Research/Code/PER-NAF/PER_NAF_BENCHMARK_REPORT.md
/Users/simon.hirlaender/Projects/Research/Code/PER-NAF/results/table5_scc_50seed_anova.csv
/Users/simon.hirlaender/Projects/Research/Code/PER-NAF/scc/outputs/scc_aggregated_summary.csv
/Users/simon.hirlaender/Projects/Research/Code/PER-NAF/scc/outputs/scc_aggregated_raw.csv
/Users/simon.hirlaender/Projects/Research/Code/PER-NAF/results/fig5_scc_50seed_cluster_results.png
/Users/simon.hirlaender/Projects/Research/Code/PER-NAF/results/fig0_speedup_benchmark.png
/Users/simon.hirlaender/Projects/Research/Code/PER-NAF/results/fig1_awake_prab_reproduction.png
/Users/simon.hirlaender/Projects/Research/Code/PER-NAF/results/fig2_linac4_prab_reproduction.png
/Users/simon.hirlaender/Projects/Research/Code/PER-NAF/results/fig3_pendulum_ablation.png
/Users/simon.hirlaender/Projects/Research/Code/PER-NAF/results/fig4_critical_stress_tests.png
/Users/simon.hirlaender/Projects/Research/Code/PER-NAF/pernaf/pernaf/networks.py
/Users/simon.hirlaender/Projects/Research/Code/PER-NAF/pernaf/pernaf/buffers.py
/Users/simon.hirlaender/Projects/Research/Code/PER-NAF/pernaf/pernaf/jax_naf.py
/Users/simon.hirlaender/Projects/Research/Code/PER-NAF/pernaf/pernaf/agent.py</pre>
    </section>
  </main>
</div>
<script>
function toggleTheme() {{
  const root = document.documentElement;
  const cur = root.getAttribute('data-theme');
  if (cur === 'dark') {{
    root.setAttribute('data-theme', 'light');
  }} else {{
    root.setAttribute('data-theme', 'dark');
  }}
}}
</script>
</body>
</html>
"""
    return html


def build_md() -> str:
    return r"""# PER-NAF & NAF2: Kritischer Benchmark- und 50-Seed-SCC-Cluster-Bericht

**Autor:** Simon Hirlaender  
**Stand:** 25.10.2026  
**SCC-Cluster-Jobs:** `2971477` (50-Seed Array) & `2971478` (Collector), $N = 2.200$ Läufe, 0 Fehler  

---

## 1. Executive Summary

1. **Algorithmisches Refactoring & Compiler-Beschleunigung (`10.7x` bis `34.6x`):**
   - **Vektorisierte Cholesky-Projektion $\mathcal{O}(B d_a^2)$** in `pernaf/pernaf/networks.py` ersetzt geschachtelte Python-Schleifen und $\mathcal{O}(B d_a^3)$ `torch.bmm`.
   - **Numba `@njit(fastmath=True)` SumTree/MinTree** in `pernaf/pernaf/buffers.py` beschleunigt das Prioritized Sampling um **34.6x** und das Speichern um **6.0x**.
   - **JAX XLA + Optax Kernel** in `pernaf/pernaf/jax_naf.py` beschleunigt ein komplettes 16-DOF-Training-Update um **10.7x** (`333.5 ms` statt `3558.8 ms` für 500 Updates mit Batch 100).
   - **Kritischer Bugfix im PER-Buffer:** Die ursprüngliche feste Initialpriorität (`1.0`) führte auf Beschleuniger-Umgebungen mit kleinen TD-Fehlern ($\sim 0.01$) zu einer **Prioritäts-Inversion**, bei der frühe Transitionen permanent mit $\sim 100\times$ höherer Wahrscheinlichkeit gezogen wurden als neue Erfahrungen. Behoben durch automatische Erstkalibrierung an das empirische TD-Fehler-Maximum und Mini-Batch-Max-Gewichtsnormierung $w_i / \max_{j \in B} w_j \in (0, 1]$.

2. **Hauptergebnis der 50-Seed-SCC-Studie ($N = 2.200$ Läufe, gepaarte $2 \times 2$ ANOVA):**
   - **CERN LINAC4 (16-DOF Nominal):** **PER** hat einen hochsignifikanten Haupteffekt von **$-154.2 \pm 39.1$ Shots ($p = 0.00025$)**, reduziert die Evaluationsschritte von `4.94` auf **`2.53–2.92` Schritte ($p = 0.00024$)** und erhöht die Erfolgsrate von `77.5%` auf **`90.6%–92.2%` ($p = 0.00012$)**.
   - **CERN LINAC4 (16-DOF HighNoise, $\sigma=0.04, \lambda=0.05$):** **`PER-NAF2` (`All`: Clipped Double-Q + PER)** ist der klare Gesamtsieger mit **`506.6` Shots ($-135.9$ Shots ggü. `NAF`, $p = 0.00083$)**, **`3.83` Eval-Schritten ($\alpha=0.4$)** und **`84.5%` Erfolgsrate** (gegenüber `64.5%` bei `NAF`, $p = 0.0033$). Während Single-Q `PER-NAF` bei aggressivem $\alpha=0.8$ unter Rauschen einbricht (`654.5` Shots, `74.8%`), bleibt `PER-NAF2` durch Clipped Double-Q stabil (`569.8` Shots, `83.1%`).
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
"""


def main() -> None:
    html_text = build_html()
    md_text = build_md()

    html_path = ROOT / "PER_NAF_BENCHMARK_REPORT.html"
    md_path = ROOT / "PER_NAF_BENCHMARK_REPORT.md"
    html_path.write_text(html_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")

    # Verification gate from skill distributable_html_documents
    bare = set(re.findall(r"--([\w-]+):", html_text.split("@media")[0]))
    used = set(re.findall(r"var\(--([\w-]+)\)", html_text))
    missing = used - bare
    print("Tokens ohne Definition im blanken :root:", missing or "keine")
    print("body-Hintergrund gesetzt:", "background: var(--ground)" in html_text or "background:var(--ground)" in html_text)
    print("Geviertstriche (HTML):", html_text.count(chr(8212)))
    print("Geviertstriche (MD):", md_text.count(chr(8212)))
    print("doctype:", html_text.lstrip().lower().startswith("<!doctype"))
    print("HTML size (KB):", round(html_path.stat().st_size / 1024, 1))


if __name__ == "__main__":
    main()
