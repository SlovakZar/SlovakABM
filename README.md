# SlovakABM — Agent-Based Migration Model of Slovakia

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SlovakZar/SlovakABM/blob/slovak_abm_colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Why do people move?** Is it the salary? Unaffordable housing? A factory that closed in the neighbouring town?

**SlovakABM** is a large-scale agent-based simulation that answers these questions for every one of Slovakia's **79 districts**. Around **70 000 virtual inhabitants** — each with their own job, income, personality, and social network — decide every month whether to stay, commute, or pack up and move. Their choices ripple through the labour and housing markets, gradually reshaping the country's demographic map.

---

## What the model captures

Think of it as a simplified Slovakia in a box:

- **Agents** are calibrated from real census microdata and social surveys (ISSP). They differ in age, education, industry, income, psychological traits — and even how stubborn they are about moving.
- **Labour market** reacts to supply and demand. When too many workers compete for the same jobs in a district, wages drop and some agents lose their positions. When a new employer opens, jobs appear and commuters rush in.
- **Housing market** responds to density. The more agents crowd into a district, the higher the effective price per square metre — making it harder for newcomers to settle.
- **Social signals** spread through networks. When your neighbour moves, you notice. When a colleague loses their job, you worry. These signals lower psychological barriers and accumulate as *migration pressure*.
- **Scenarios** let you test "what if": *What if a big manufacturer closes in Zilina? What if Bratislava gets a new tech hub?*

The model runs in **monthly ticks** (12 ticks = 1 year) and produces a full diagnostic report: heatmaps, regional balance sheets, agent-level parameter dynamics, and top migration routes.

### How agents decide

The decision-making follows a **two-barrier behavioural model** grounded in psychology (TPB — Theory of Planned Behaviour):

```
+--------------------------------------------------+
|  BARRIER 1 — Is the pain strong enough?          |
|  aspirations (accumulated dissatisfaction)       |
|       x capabilities (income, education, ties)   |
|       > dynamic_threshold -> intention formed    |
+--------------------------------------------------+
|  BARRIER 2 — Has the pressure boiled over?       |
|  Each tick while intending:                      |
|    migration_pressure += D_perceived - inertia   |
|  When pressure hits a random threshold: ACT      |
+--------------------------------------------------+
|  ACTION — What to do?                            |
|  Try: commute -> move to new district ->         |
|       move to satellite town -> adapt and stay   |
|  Success -> confidence up  |  Failure -> adapt   |
+--------------------------------------------------+
```

In plain language: an agent first needs to feel dissatisfied *enough* (economic + housing pressure). Then that dissatisfaction must *accumulate* over time to overcome their personal inertia. Only then do they search for a better job or home — and they may succeed or fail.

---

## Project structure

| File / Directory | Purpose |
|---|---|
| `run.py` | CLI entry point and programmatic `run()` function |
| `agents.py` | Quota-based agent initialisation (employed / unemployed / students) with survey-calibrated psychometrics |
| `engine.py` | Main simulation loop: two-barrier activation, heuristic search, domain updates, signal decay |
| `graph.py` | 79-node directed graph from real commuting matrix; market response (wage, housing price, industry pressure) |
| `signals.py` | EventBus and Dispatcher: typed events (AGENT_MOVED, LOST_JOB, NEW_EMPLOYER, ...) with propagation rules |
| `scenario.py` | Scheduled scenario events (factory closure, new employer, infrastructure shock) from JSON |
| `report.py` | Report generator: heatmap, demographic portrait, agent parameter matrix, master district table |
| `seed_runner.py` | Multi-run analysis with different random seeds |
| `data/` | Environment data, agent distributions, commuting matrix, survey parameters |
| `utilities/` | Grid search (`grid_runner.py`), LHS sensitivity analysis (`lhs_runner.py`), batch configs |
| `slovak_abm_colab.ipynb` | Interactive Colab notebook with control panel |

---

## Quick start

### Option 1 — Google Colab (recommended)

Click the badge to open the interactive notebook — no installation needed:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SlovakZar/SlovakABM/blob/main/slovak_abm_colab.ipynb)

The notebook provides a control panel with sliders for:
- Number of agents (5k–70k)
- Number of ticks (6–120)
- Report sections (heatmap, demographic portrait, master table, etc.)
- Custom scenario events (new employer, factory closure, housing shock)

If the repository is private, the Colab notebook will prompt you for a GitHub token. Use a [personal access token](https://github.com/settings/tokens) with `repo` scope.

### Option 2 — Local installation

```bash
# 1. Clone the repository
git clone https://github.com/SlovakZar/SlovakABM.git
cd SlovakABM

# 2. Install dependencies
pip install numpy pandas networkx matplotlib

# 3. Run a small simulation (5k agents, 24 ticks = 2 years)
python run.py --agents 5000 --ticks 24

# 4. Full simulation (70k agents, 60 ticks = 5 years)
python run.py --agents 70000 --ticks 60 --output report.txt

# 5. Run with behavioural audit
python run.py --agents 5000 --ticks 24 --detail
```

### CLI arguments

| Flag | Default | Description |
|---|---|---|
| `--agents` | 70000 | Number of agents |
| `--ticks` | 60 | Number of ticks (1 tick = 1 month) |
| `--seed` | 42 | Random seed |
| `--output` | (stdout) | File path for the report |
| `--detail` | `False` | Include behavioural audit in the report |

---

## Report output

The `report.py` module produces a diagnostic report with toggleable sections:

| Section | Description |
|---|---|
| **Agent Parameters Matrix** | Tick 0 to Tick 6 dynamics: aspirations, D_econ, D_place, capabilities, dynamic inertia, TPB state, migration pressure |
| **Demographic Portrait** | Unemployment rate, mean/median wage, regional table, top-5 industries, dynamic signal variables |
| **Dynamics Summary** | Yearly trends: moves, commutes, unemployment, activations |
| **Interregional Balance** | Population, unemployment, and wage deltas by region |
| **Master District Table** | All 79 districts: agent counts and effective housing price per tick, plus delta econ/place |
| **Top-10 Routes** | Most popular relocation origins/destinations and new commute directions |
| **District Heatmap** | Geographical colour map: green = growth, red = decline; node size = population |
| **Random Regions and Industries** | Occupied vs vacant positions for randomly selected regions and industries |
| **Behavioural Audit** | *(detail only)* Random slice of agent decisions with before/after snapshots |

---

## Data sources

| Source | What |
|---|---|
| **[SASD](https://sasd.sav.sk)** — Slovak Archive of Social Data | Psychological parameters calibrated from social surveys: ISSP Socialne siete Slovensko 2017, ISSP Pracovne orientacie Slovensko 2016, ISSP Digitalne spolocnosti Slovensko 2024 |
| **[DATAcube](https://datacube.statistics.sk/)** — Statistical Office of SR | Main statistical data: wages by industry, unemployment rates, population by age/sex/education, dwelling counts |
| **SODB 2021** (Statistical Office of SR) | Census 2021: detailed district-level demographics, commuting flows |
| **NBS** (National Bank of Slovakia) | Housing prices per square metre by district |
| **OSRM** | Travel time matrix between all 79 districts |

Key data files in the repository:
- `data/environment.json` — node attributes (wages, housing, infrastructure, business counts)
- `data/commuting_filtered_with_travel.csv` — directed flow matrix and travel times between districts
- `data/agent_init_distributions.json` — district-level age/sex/education distributions
- `data/agent_params_from_survey.json` — psychological parameter distributions from ISSP surveys
- `MigrationSaldo2023-2025.csv` — real migration balance 2023–2025 for validation
- `companies_by_district_industry_size.csv` — employer counts by district, industry, and size

---

## Key concepts

### Dynamic signal variables

| Variable | Effect | Decay |
|---|---|---|
| `econ_penalty` | Direct addition to economic dissatisfaction (D_econ) | -0.01 / tick |
| `infra_bonus` | Boosts infrastructure component of place satisfaction | -0.01 / tick |
| `inertia_mobility_penalty` | Added to base inertia — neighbours moving lowers your resistance | +/-0.01 towards 0 / tick |
| `jobloss_econ_gap_bonus` | Temporary spike in economic sensitivity after losing a job | ramp-up +0.05 x3 then down |
| `migration_pressure` | Accumulated gap between perceived dissatisfaction and inertia | Reset on action |
| `signal_reduction` | Social signals erode the psychological barrier to moving | x0.85 / tick |
| `social_boost` | Improves social satisfaction after positive events | x0.80 / tick |

### Event system

Events flow through social networks using the **EventBus** and **Dispatcher** architecture in `signals.py`. Rules define *who* is affected (the agent themselves, same-industry colleagues, neighbours in the district), *what* changes (social_boost, econ_penalty, etc.), and *how strongly*. Scenario events — factory closures, new employers, infrastructure projects — are scheduled declaratively in `scenario.json`.

---

## Citation

If you use SlovakABM in your research, please cite:

```bibtex
@software{SlovakABM,
  author = {},
  title  = {SlovakABM: Agent-Based Migration Model of Slovakia},
  year   = {2025},
  url    = {https://github.com/SlovakZar/SlovakABM}
}
```

---

## License

This project is licensed under the MIT License. See `LICENSE` file for details.
