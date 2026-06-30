"""
agents.py v9 — Квотная инициализация агентов

ИЗМЕНЕНИЯ v9:
  1. КВОТНАЯ ИНИЦИАЛИЗАЦИЯ: вместо семплирования n агентов с последующей
     классификацией (где все не-employed становились безработными),
     теперь явно создаются квоты: занятые (employed_share), безработные
     (unemployed_share) и студенты (enrollment rates). Неактивное население
     15–65 агентами НЕ становится.
  2. ВОЗРАСТНОЙ ДИАПАЗОН расширен с 18–65 до 15–65.
  3. УБРАН inactivity_r — неактивные больше не попадают в статус "unemployed".

ИЗМЕНЕНИЯ v8:
  1. FIX Bernoulli regional: network_location/network_job_search теперь
     получают региональную коррекцию (раньше проверка Bernoulli была до неё).
  4. HURDLE для shock_sensitivity: Bernoulli(был ли шок) × интенсивность,
     вместо normal+clip который давал массу искусственных нулей.
  6. КОРРЕЛЯЦИИ: perceived_control→econ_perceived_control,
     digital_comm→info_quality, digital_comm→digital_trust,
     future_orientation→internal_mig_threshold.
  7. SETTLEMENT в группах: sample_param принимает settlement (metro/city/town/rural),
     влияет на commuter_threshold/internal_mig_threshold/external_mig_threshold/
     job_flexibility/future_orientation.

Ключевые изменения относительно v3.1:

1. РАЗДЕЛЕНИЕ ЛОКАЦИИ:
   Каждый агент имеет два поля:
     residence_district  — где живёт (жильё, социальные связи, inertia)
     workplace_district  — где работает (зарплата, отрасль)
   status: "stay" | "commute" | "unemployed" | "student"

2. ИНИЦИАЛИЗАЦИЯ ИЗ COMMUTING-МАТРИЦЫ:
   Шаг 1 — workplace_district из реальных flow_work пропорций.
   Если origin == destination → status "stay".
   Если origin != destination → status "commute".
   Безработные → workplace_district = residence_district, status "unemployed".
   Студенты (15–26) → workplace_district = место учёбы (flow_school), status "student".

3. ОТРАСЛЬ И ЗАРПЛАТА — из workplace_district:
   industry и wage берутся из распределений района РАБОТЫ, не проживания.
   Это критично: структура Братиславы (ICT, professional) ≠ Сеница.

4. jobs_capacity ИЗ COMMUTING-МАТРИЦЫ:
   Суммируем входящие flow_work по каждому district как destination.
   Это workplace-based занятость, а не residence-based.
   Экспортируется в словарь JOBS_CAPACITY для использования в graph.py и engine.py.

5. FFT-СОВМЕСТИМАЯ СТРУКТУРА:
   intention_state расширен: "none" | "seeking_work" | "seeking_residence" | "commute_pending"
   dst_work — целевой район работы (заполняется в seeking_work фазе engine.py)
   Убран forming_ticks / forming_duration — заменяется FFT-деревом в engine.py.
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

# ── Константы ─────────────────────────────────────────────────────────────────

BERNOULLI_PARAMS = {'network_job_search', 'network_location'}

# Rogers & Castro 1981
RC_A1, RC_MU1, RC_ALPHA1 = 0.09, 22.0, 0.10
RC_A2, RC_MU2, RC_ALPHA2 = 0.01, 65.0, 0.07
RC_C = 0.005

# Возрастные бины SODB в диапазоне 15–65
# (midpoint, width, fraction_of_bin_to_use)
# v9: расширен с 18–65 до 15–65 — бин 15-19 теперь полностью (5/5),
#      бин 65-69 — только возраст 65 (1/5).
AGE_BIN_META = {
    "15 - 19 years": (17.0, 5, 1.0),
    "20 - 24 years": (22.0, 5, 1.0),
    "25 - 29 years": (27.0, 5, 1.0),
    "30 - 34 years": (32.0, 5, 1.0),
    "35 - 39 years": (37.0, 5, 1.0),
    "40 - 44 years": (42.0, 5, 1.0),
    "45 - 49 years": (47.0, 5, 1.0),
    "50 - 54 years": (52.0, 5, 1.0),
    "55 - 59 years": (57.0, 5, 1.0),
    "60 - 64 years": (62.0, 5, 1.0),
    "65 - 69 years": (65.0, 1, 1 / 5),
}

MARITAL_MAP = {
    "Married":       "married",
    "Never married": "single",
    "Divorced":      "divorced",
    "Widowed":       "widowed",
}

DISTRICT_TO_REGION_CODE = {
    "District of Bratislava I": "BA", "District of Bratislava II": "BA",
    "District of Bratislava III": "BA", "District of Bratislava IV": "BA",
    "District of Bratislava V": "BA", "District of Malacky": "BA",
    "District of Pezinok": "BA", "District of Senec": "BA",
    "District of Trnava": "TT", "District of Dunajská Streda": "TT",
    "District of Galanta": "TT", "District of Hlohovec": "TT",
    "District of Piešťany": "TT", "District of Senica": "TT",
    "District of Skalica": "TT",
    "District of Trenčín": "TN", "District of Bánovce nad Bebravou": "TN",
    "District of Ilava": "TN", "District of Myjava": "TN",
    "District of Nové Mesto nad Váhom": "TN", "District of Partizánske": "TN",
    "District of Považská Bystrica": "TN", "District of Púchov": "TN",
    "District of Prievidza": "TN",
    "District of Nitra": "NR", "District of Komárno": "NR",
    "District of Levice": "NR", "District of Nové Zámky": "NR",
    "District of Šaľa": "NR", "District of Topoľčany": "NR",
    "District of Zlaté Moravce": "NR",
    "District of Žilina": "ZA", "District of Bytča": "ZA",
    "District of Čadca": "ZA", "District of Dolný Kubín": "ZA",
    "District of Kysucké Nové Mesto": "ZA", "District of Liptovský Mikuláš": "ZA",
    "District of Martin": "ZA", "District of Námestovo": "ZA",
    "District of Ružomberok": "ZA", "District of Turčianske Teplice": "ZA",
    "District of Tvrdošín": "ZA",
    "District of Banská Bystrica": "BB", "District of Banská Štiavnica": "BB",
    "District of Brezno": "BB", "District of Detva": "BB",
    "District of Krupina": "BB", "District of Lučenec": "BB",
    "District of Poltár": "BB", "District of Revúca": "BB",
    "District of Rimavská Sobota": "BB", "District of Veľký Krtíš": "BB",
    "District of Zvolen": "BB", "District of Žiar nad Hronom": "BB",
    "District of Žarnovica": "BB",
    "District of Prešov": "PO", "District of Bardejov": "PO",
    "District of Humenné": "PO", "District of Kežmarok": "PO",
    "District of Levoča": "PO", "District of Medzilaborce": "PO",
    "District of Poprad": "PO", "District of Sabinov": "PO",
    "District of Snina": "PO", "District of Stará Ľubovňa": "PO",
    "District of Stropkov": "PO", "District of Svidník": "PO",
    "District of Vranov nad Topľou": "PO",
    "District of Košice I": "KE", "District of Košice II": "KE",
    "District of Košice III": "KE", "District of Košice IV": "KE",
    "District of Košice-okolie": "KE", "District of Gelnica": "KE",
    "District of Rožňava": "KE", "District of Sobrance": "KE",
    "District of Spišská Nová Ves": "KE", "District of Trebišov": "KE",
    "District of Michalovce": "KE",
}

# ── Тип поселения (v8) ──────────────────────────────────────────────────────
# metro: Братислава I–V, Кошице I–IV
# city:  региональные центры (Trnava, Nitra, Žilina, Banská Bystrica, Prešov, Trenčín)
# town:  районные центры с населением >30k
# rural: всё остальное
SETTLEMENT_MAP: dict[str, str] = {}
for d in DISTRICT_TO_REGION_CODE:
    name = d.replace("District of ", "")
    if "Bratislava" in name:
        SETTLEMENT_MAP[d] = "metro"
    elif "Košice I" in name or "Košice II" in name or "Košice III" in name or "Košice IV" in name:
        SETTLEMENT_MAP[d] = "metro"
    elif name in ("Trnava", "Nitra", "Žilina", "Banská Bystrica", "Prešov", "Trenčín"):
        SETTLEMENT_MAP[d] = "city"
    elif name in ("Poprad", "Martin", "Zvolen", "Prievidza", "Michalovce",
                  "Spišská Nová Ves", "Humenné", "Bardejov", "Liptovský Mikuláš",
                  "Ružomberok", "Piešťany", "Nové Zámky", "Komárno", "Levice",
                  "Lučenec", "Čadca", "Dunajská Streda", "Trebišov",
                  "Vranov nad Topľou", "Rimavská Sobota", "Senec", "Pezinok",
                  "Považská Bystrica", "Dolný Kubín", "Galanta"):
        SETTLEMENT_MAP[d] = "town"
    else:
        SETTLEMENT_MAP[d] = "rural"

# ── Кэш загруженных данных ────────────────────────────────────────────────────

_INIT_DISTS = None
_SURVEY = None
_COMMUTING = None

# Глобальный словарь jobs_capacity: {district: int}
# Заполняется при первом вызове _get_commuting(), используется graph.py
JOBS_CAPACITY: dict = {}

# v3: Новая структура — отраслевая ёмкость с разделением на occupied/vacant
# {district: {industry: {"occupied": int, "vacant": int}}}
# occupied — занятые рабочие места (агенты с workplace=district, industry=X)
# vacant   — открытые вакансии (изначально от unemployed_share, затем от NEW_EMPLOYER/CLOSED_EMPLOYER)
# job_capacity по отрасли = occupied + vacant
INDUSTRY_JOBS_CAPACITY: dict[str, dict[str, dict[str, int]]] = {}


def _get_init_dists(path="data/agent_init_distributions.json"):
    global _INIT_DISTS
    if _INIT_DISTS is None:
        p = Path(path)
        if not p.exists():
            p = Path(__file__).parent / path
        with open(p, encoding="utf-8") as f:
            _INIT_DISTS = json.load(f)["districts"]
    return _INIT_DISTS


def _get_survey(path="data/agent_params_from_survey.json"):
    global _SURVEY
    if _SURVEY is None:
        p = Path(path)
        if not p.exists():
            p = Path(__file__).parent / path
        with open(p, encoding="utf-8") as f:
            _SURVEY = json.load(f)
    return _SURVEY


def _get_commuting(path="data/commuting_filtered_with_travel.csv"):
    """
    Загружает commuting-матрицу и строит два словаря:
      outflow_probs[origin] = {destination: probability}  — вероятности по flow_work
      jobs_capacity[district] = int  — суммарный входящий flow_work (workplace-based)

    Self-loops (origin == destination) включаются — это агенты "stay".
    """
    global _COMMUTING, JOBS_CAPACITY
    if _COMMUTING is not None:
        return _COMMUTING

    p = Path(path)
    if not p.exists():
        p = Path(__file__).parent / path

    df = pd.read_csv(p)

    # Фильтруем только district→district строки
    mask = (
        df["origin_district"].str.startswith("District of") &
        df["destination_district"].str.startswith("District of")
    )
    df = df[mask].copy()

    # Строим outflow_probs: для каждого origin — распределение по destinations
    outflow_probs = {}
    for origin, grp in df.groupby("origin_district"):
        total = grp["flow_work"].sum()
        if total <= 0:
            continue
        outflow_probs[origin] = {
            row["destination_district"]: row["flow_work"] / total
            for _, row in grp.iterrows()
            if row["flow_work"] > 0
        }

    # Строим jobs_capacity: суммируем входящий flow_work по каждому destination
    jc = df.groupby("destination_district")["flow_work"].sum().to_dict()
    JOBS_CAPACITY.update({d: int(v) for d, v in jc.items()})

    _COMMUTING = outflow_probs
    print(f"  Commuting-матрица: {len(outflow_probs)} районов-источников")
    print(f"  jobs_capacity: min={min(JOBS_CAPACITY.values()):,} "
          f"max={max(JOBS_CAPACITY.values()):,} "
          f"(Bratislava I: {JOBS_CAPACITY.get('District of Bratislava I', 0):,})")
    return _COMMUTING


def _scale_jobs_capacity(n_agents: int):
    """
    Сжимает JOBS_CAPACITY с реальной популяции (~2.5 млн занятых)
    до масштаба симуляции (~70 000 агентов).

    scale = n_agents / сумма всех capacity.
    Минимум 1 — защита от деления на 0 в engine.
    """
    global JOBS_CAPACITY
    total_cap = sum(JOBS_CAPACITY.values())
    if total_cap == 0:
        return
    scale = n_agents / total_cap
    JOBS_CAPACITY = {
        d: max(1, int(v * scale))
        for d, v in JOBS_CAPACITY.items()
    }


def _init_industry_jobs(init_dists: dict, n_agents: int):
    """
    v3: Инициализирует INDUSTRY_JOBS_CAPACITY с разделением на occupied/vacant.

    Для каждого района:
      occupied_by_industry = scaled_population × employed_share × industry_share
      vacant_by_industry   = occupied / (1 − unemployed_share) − occupied

    Итого job_capacity по отрасли = occupied + vacant.

    Также обновляет глобальный JOBS_CAPACITY (сумма по всем отраслям).
    """
    global INDUSTRY_JOBS_CAPACITY, JOBS_CAPACITY

    # Суммарная популяция 18–65 по всем районам (из init_dists)
    total_pop = sum(
        d["population"] for d in init_dists.values()
    )
    if total_pop == 0:
        return

    scale = n_agents / total_pop

    new_industry_jobs: dict[str, dict[str, dict[str, int]]] = {}
    new_jobs_capacity: dict[str, int] = {}

    for district, data in init_dists.items():
        pop = data.get("population", 1)
        scaled_pop = max(1, int(pop * scale))

        employment = data.get("employment", {})
        employed_share = employment.get("employed_share", 0.45)
        unemployed_share = employment.get("unemployed_share", 0.06)

        industry_shares = data.get("industry", {})
        if not industry_shares:
            industry_shares = {"Other": 1.0}

        # Нормализуем доли отраслей
        total_share = sum(industry_shares.values())
        if total_share == 0:
            total_share = 1.0

        district_jobs: dict[str, dict[str, int]] = {}
        district_total_capacity = 0

        for industry, share in industry_shares.items():
            norm_share = share / total_share

            # Занятые места в отрасли
            occupied = max(1, int(scaled_pop * employed_share * norm_share))

            # Вакантные места = occupied / (1 − u) − occupied
            # При unemployed_share → 1 избегаем деления на 0
            safe_unemp = max(unemployed_share, 0.005)
            total_positions = occupied / max(1.0 - safe_unemp, 0.01)
            vacant = max(0, int(round(total_positions - occupied)))

            district_jobs[industry] = {"occupied": occupied, "vacant": vacant}
            district_total_capacity += occupied + vacant

        new_industry_jobs[district] = district_jobs
        new_jobs_capacity[district] = max(1, district_total_capacity)

    INDUSTRY_JOBS_CAPACITY = new_industry_jobs
    JOBS_CAPACITY = new_jobs_capacity

    # Диагностика
    total_occ = sum(
        v["occupied"]
        for d in INDUSTRY_JOBS_CAPACITY.values()
        for v in d.values()
    )
    total_vac = sum(
        v["vacant"]
        for d in INDUSTRY_JOBS_CAPACITY.values()
        for v in d.values()
    )
    print(f"  INDUSTRY_JOBS_CAPACITY: {len(INDUSTRY_JOBS_CAPACITY)} районов")
    print(f"    occupied={total_occ:,}  vacant={total_vac:,}  "
          f"total_capacity={total_occ+total_vac:,}")
    print(f"    vacant/occupied ratio={total_vac/max(total_occ,1):.3f}")
    # Пример для проверки
    sample_d = "District of Bratislava I"
    if sample_d in new_industry_jobs:
        sample = new_industry_jobs[sample_d]
        print(f"    {sample_d}:")
        for ind, v in sorted(sample.items(), key=lambda x: -(x[1]["occupied"]+x[1]["vacant"]))[:5]:
            print(f"      {ind}: occ={v['occupied']:,} vac={v['vacant']:,} "
                  f"total={v['occupied']+v['vacant']:,}")


# ── Школьные/студенческие потоки ──────────────────────────────────────────────

_SCHOOL_OUTFLOW = None
_ENROLLMENT_RATES: dict = {}   # {district: {age_bin: rate}}


def _compute_school_outflow(path="data/commuting_filtered_with_travel.csv"):
    """
    Строит два словаря на основе flow_school:
      school_outflow[origin] = {destination: probability}
      enrollment_rates[district] = {age_bin: rate}

    enrollment_rate = total_flow_school_origin / pop_15_24
    Затем разбивается по возрастным бинам с весами:
      15–19: ×1.20 (средняя школа, почти все)
      20–24: ×0.80 (университет)
      25–29: ×0.15 (PhD, второе высшее)
    """
    global _SCHOOL_OUTFLOW, _ENROLLMENT_RATES
    if _SCHOOL_OUTFLOW is not None:
        return _SCHOOL_OUTFLOW, _ENROLLMENT_RATES

    p = Path(path)
    if not p.exists():
        p = Path(__file__).parent / path

    df = pd.read_csv(p)

    mask = (
        df["origin_district"].str.startswith("District of") &
        df["destination_district"].str.startswith("District of")
    )
    df = df[mask].copy()

    # ── school_outflow ────────────────────────────────────────────────────────
    school_outflow = {}
    total_students_by_district = {}
    for origin, grp in df.groupby("origin_district"):
        total = grp["flow_school"].sum()
        if total <= 0:
            continue
        total_students_by_district[origin] = total
        school_outflow[origin] = {
            row["destination_district"]: row["flow_school"] / total
            for _, row in grp.iterrows()
            if row["flow_school"] > 0
        }

    # ── enrollment_rates ─────────────────────────────────────────────────────
    init_dists = _get_init_dists()

    for district in init_dists:
        age_sex = init_dists[district].get("age_sex", {})
        pop_15_19 = sum(
            meta["count"] * AGE_BIN_META[key.split("|")[0]][2]
            for key, meta in age_sex.items()
            if key.split("|")[0] in ("15 - 19 years",)
        )
        pop_20_24 = sum(
            meta["count"] * AGE_BIN_META[key.split("|")[0]][2]
            for key, meta in age_sex.items()
            if key.split("|")[0] in ("20 - 24 years",)
        )
        pop_25_29 = sum(
            meta["count"] * AGE_BIN_META[key.split("|")[0]][2]
            for key, meta in age_sex.items()
            if key.split("|")[0] in ("25 - 29 years",)
        )
        pop_15_24 = max(pop_15_19 + pop_20_24, 1)

        # Базовая доля студентов из данных flow_school
        total_students = total_students_by_district.get(district, 0)
        base_rate = np.clip(total_students / pop_15_24, 0.02, 0.95)

        # Возрастной профиль
        _ENROLLMENT_RATES[district] = {
            "15-19": float(np.clip(base_rate * 1.20, 0.05, 0.98)),
            "20-24": float(np.clip(base_rate * 0.80, 0.02, 0.90)),
            "25-29": float(np.clip(base_rate * 0.15, 0.01, 0.25)),
        }

    _SCHOOL_OUTFLOW = school_outflow
    n_with = sum(1 for v in _ENROLLMENT_RATES.values() if v["15-19"] > 0.05)
    avg_15_19 = np.mean([v["15-19"] for v in _ENROLLMENT_RATES.values()])
    avg_20_24 = np.mean([v["20-24"] for v in _ENROLLMENT_RATES.values()])
    print(f"  Школьные потоки: {len(school_outflow)} районов-источников")
    print(f"  Enrollment rates: {n_with} районов, "
          f"ср. 15-19={avg_15_19:.2f}, 20-24={avg_20_24:.2f}")
    return school_outflow, _ENROLLMENT_RATES


def _sample_schoolplace(
    residence: str,
    school_outflow: dict,
    rng: np.random.Generator,
) -> str:
    """Выбирает место учёбы из flow_school пропорций (аналог _sample_workplace)."""
    probs = school_outflow.get(residence)
    if not probs:
        return residence
    destinations = list(probs.keys())
    weights = np.array([probs[d] for d in destinations], dtype=float)
    return destinations[rng.choice(len(destinations), p=weights / weights.sum())]


# ── Вспомогательные функции ───────────────────────────────────────────────────

def rogers_castro_mobility(age: float) -> float:
    """Rogers & Castro 1981: базовая вероятность мобильности по возрасту."""
    if age < 5:
        return 0.0
    labour = (RC_A1 * math.exp(-RC_ALPHA1 * (age - RC_MU1)) if age >= RC_MU1
              else RC_A1 * math.exp(-RC_ALPHA1 * (RC_MU1 - age) * 0.3))
    pension = RC_A2 * math.exp(-RC_ALPHA2 * abs(age - RC_MU2))
    return float(np.clip((labour + pension + RC_C) / (RC_A1 + RC_C), 0.0, 1.0))


def sample_param(name: str, age: float, education: str,
                 region: str = None, settlement: str = None,
                 rng: np.random.Generator = None) -> float:
    """Сэмплирует параметр агента из SASD-распределений (по группе возраст×образование×[поселение])."""
    if rng is None:
        rng = np.random.default_rng()
    survey = _get_survey()
    p = survey.get(name)
    if not p:
        return 0.5

    ag = ("18-30" if age < 31 else "31-45" if age < 46
          else "46-60" if age < 61 else "60+")
    eg = education if education in ("low", "medium", "high") else "medium"

    by_group = p.get("by_group", {})

    # Ищем группу: сначала age+edu+settlement, потом age+edu, потом global
    g = None
    if settlement:
        g = by_group.get(str((ag, eg, settlement)))
    if g is None:
        g = by_group.get(str((ag, eg)))
    mu = float(g["mean"]) if g else float(p.get("global_mean", 0.5))
    sd = float(g["std"])  if g else float(p.get("global_std", 0.1))
    n  = int(g["n"])      if g else 100

    # Если группа не найдена — заполняем из регионального среднего
    if not g and region:
        regional = survey.get("_regional", {}).get(name, {})
        r = regional.get(region)
        if r and r.get("n", 0) > 5:
            mu = float(r["mean"])
            sd = float(r["std"])

    # Взвешивание с региональным профилем
    regional = survey.get("_regional", {}).get(name, {})
    if region and regional:
        r = regional.get(region)
        if r and r.get("n", 0) > 5:
            w_g = 1.0 / max(sd / max(n ** 0.5, 1), 1e-6)
            w_r = 1.0 / max(r["std"] / max(r["n"] ** 0.5, 1), 1e-6)
            mu  = (w_g * mu + w_r * r["mean"]) / (w_g + w_r)
            sd  = max(sd, r["std"]) * 0.8

    # ═══ ИСПРАВЛЕНИЕ v8: Bernoulli ПОСЛЕ regional ═══
    # Раньше Bernoulli-проверка была до regional-коррекции,
    # из-за чего network_location и network_job_search игнорировали
    # региональные профили (BA=0.18 vs NR=0.43 для network_location).
    if name in BERNOULLI_PARAMS:
        return float(rng.random() < float(np.clip(mu, 0, 1)))
    return float(np.clip(rng.normal(mu, max(sd, 0.01)), 0.0, 1.0))


def _weighted_choice(d: dict, rng: np.random.Generator) -> str:
    keys    = list(d.keys())
    weights = np.array([d[k] for k in keys], dtype=float)
    total   = weights.sum()
    if total == 0:
        return keys[0]
    return keys[rng.choice(len(keys), p=weights / total)]


def _sample_workplace(
    residence: str,
    outflow_probs: dict,
    rng: np.random.Generator,
) -> str:
    """
    Выбирает workplace_district из commuting-матрицы.
    Если для района нет данных — возвращает сам район (stay).
    """
    probs = outflow_probs.get(residence)
    if not probs:
        return residence
    destinations = list(probs.keys())
    weights = np.array([probs[d] for d in destinations], dtype=float)
    return destinations[rng.choice(len(destinations), p=weights / weights.sum())]


# ── Главная функция создания агентов ─────────────────────────────────────────

def create_agents(
    dist_path: str = "agent_init_distributions.json",
    survey_path: str = "agent_params_from_survey.json",
    commuting_path: str = "data/commuting_filtered_with_travel.csv",
    n_agents: int = 70000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    init_dists   = _get_init_dists(dist_path)
    _get_survey(survey_path)
    outflow_probs = _get_commuting(commuting_path)
    school_outflow, _ = _compute_school_outflow(commuting_path)

    districts = list(init_dists.keys())

    # ── v9: целевые квоты по категориям (занятые / безработные / студенты) ─
    # Вместо семплирования n случайных людей с последующей классификацией,
    # явно выделяем три НЕПЕРЕСЕКАЮЩИХСЯ категории. Неактивное население 15–65
    # (не работающее, не безработное, не учащееся) агентами НЕ становится.
    pop_1564 = {}
    pop_1526 = {}
    expected_students = {}

    for d in districts:
        age_sex = init_dists[d].get("age_sex", {})
        total_1564 = 0.0
        total_1526 = 0.0
        exp_stu = 0.0

        for key, meta in age_sex.items():
            bin_label = key.split("|")[0]
            if bin_label not in AGE_BIN_META:
                continue
            eff = meta["count"] * AGE_BIN_META[bin_label][2]

            # 15–64: все бины КРОМЕ 65-69
            if bin_label != "65 - 69 years":
                total_1564 += eff

            # 15–26: бины для студентов
            if bin_label in ("15 - 19 years", "20 - 24 years", "25 - 29 years"):
                total_1526 += eff
                age_key = {"15 - 19 years": "15-19", "20 - 24 years": "20-24",
                           "25 - 29 years": "25-29"}[bin_label]
                enroll_r = _ENROLLMENT_RATES.get(d, {}).get(age_key, 0.05)
                exp_stu += eff * enroll_r

        pop_1564[d] = max(1.0, total_1564)
        pop_1526[d] = max(1.0, total_1526)
        expected_students[d] = max(0.0, exp_stu)

    # Целевые количества по категориям (в масштабе реальной популяции)
    targets = {}
    total_active_real = 0.0
    for d in districts:
        dd = init_dists[d]
        emp_share = dd.get("employment", {}).get("employed_share", 0.45)
        unemp_share = dd.get("employment", {}).get("unemployed_share", 0.06)

        n_emp = pop_1564[d] * emp_share
        n_unemp = pop_1564[d] * unemp_share
        n_stu = expected_students[d]

        targets[d] = {"employed": n_emp, "unemployed": n_unemp, "student": n_stu}
        total_active_real += n_emp + n_unemp + n_stu

    # Масштабируем до n_agents
    scale = n_agents / max(total_active_real, 1.0)
    for d in districts:
        for cat in ("employed", "unemployed", "student"):
            targets[d][cat] = max(1, round(targets[d][cat] * scale))

    # Корректировка округления
    current_total = sum(targets[d][cat] for d in districts
                        for cat in ("employed", "unemployed", "student"))
    diff = n_agents - current_total
    # Распределяем разницу по крупнейшим категориям
    dlist = list(districts)
    while diff != 0:
        for d in dlist:
            if diff == 0:
                break
            # Приоритет: employed (самая большая категория)
            if diff > 0:
                targets[d]["employed"] += 1
                diff -= 1
            else:
                if targets[d]["employed"] > 1:
                    targets[d]["employed"] -= 1
                    diff += 1

    records  = []
    agent_id = 0

    for residence in districts:
        dd = init_dists[residence]
        tg = targets[residence]
        region_code = DISTRICT_TO_REGION_CODE.get(residence, "XX")

        age_sex = dd.get("age_sex", {})
        # Ключи для 15–64 (занятые/безработные) — исключаем 65-69
        work_keys = [k for k in age_sex
                     if k.split("|")[0] in AGE_BIN_META
                     and k.split("|")[0] != "65 - 69 years"]
        work_weights = np.array([age_sex[k]["count"] * AGE_BIN_META[k.split("|")[0]][2]
                                 for k in work_keys], dtype=float)
        if work_weights.sum() > 0:
            work_weights /= work_weights.sum()
        else:
            work_weights = np.ones(len(work_keys)) / len(work_keys)

        # Ключи для студентов (15–29)
        stu_keys = [k for k in age_sex
                    if k.split("|")[0] in ("15 - 19 years", "20 - 24 years", "25 - 29 years")]
        stu_weights = np.array([age_sex[k]["count"] * AGE_BIN_META[k.split("|")[0]][2]
                                for k in stu_keys], dtype=float)
        if stu_weights.sum() > 0:
            stu_weights /= stu_weights.sum()
        else:
            stu_weights = np.ones(len(stu_keys)) / len(stu_keys)

        edu_dist       = dd.get("education", {"low": 0.3, "medium": 0.5, "high": 0.2})
        owner_share    = dd.get("owner_share", 0.65)
        housing_m2     = dd.get("housing_price_m2", 1500.0)
        nat_dist_d     = dd.get("nationality", {"Slovak": 1.0})

        # ── Вспомогательная: создание записи агента ──────────────────────────
        def make_agent(age, sex, status, is_employed, workplace, industry, wage,
                       graduation_cohort=-1, education_override=None):
            nonlocal agent_id
            education = (education_override if education_override is not None
                         else _weighted_choice(edu_dist, rng) if edu_dist else "medium")
            marital_sex = dd.get("marital", {}).get(sex, {})
            marital     = MARITAL_MAP.get(
                _weighted_choice(marital_sex, rng) if marital_sex else "Never married",
                "single"
            )
            nationality = _weighted_choice(nat_dist_d, rng)

            # ── SASD параметры ───────────────────────────────────────────────
            settlement = SETTLEMENT_MAP.get(residence, "town")
            def sp(name): return sample_param(name, age, education, region_code,
                                              settlement, rng)

            perceived_control      = sp("perceived_control")
            econ_perceived_control = sp("econ_perceived_control")
            econ_perceived_control = float(np.clip(
                econ_perceived_control * 0.60 + perceived_control * 0.40, 0.0, 1.0
            ))

            inertia_social         = sp("inertia_social_component")
            info_quality           = sp("info_quality_modifier")
            d_econ_weight          = sp("domain_economic_weight")
            d_econ_gap             = sp("domain_economic_gap")
            d_econ_threshold       = sp("domain_economic_threshold")
            d_social_weight        = sp("domain_social_weight")
            d_future_value         = sp("domain_future_value")
            d_future_place         = sp("domain_future_place")
            family_modifier        = sp("family_weight_modifier")
            commuter_threshold     = sp("commuter_mode_threshold")
            internal_mig_thr       = sp("internal_mig_threshold")
            external_mig_thr       = sp("external_mig_threshold")
            internal_mig_thr = float(np.clip(
                internal_mig_thr + (1.0 - d_future_value) * 0.20, 0.0, 1.0
            ))
            job_flexibility        = sp("job_flexibility_threshold")
            shock_sensitivity      = sp("inertia_shock_sensitivity")
            satisfaction_base      = sp("satisfaction_init")

            p_shock_raw = _get_survey().get("inertia_shock_sensitivity", {}).get("p_shock", 0.30)
            if rng.random() < p_shock_raw:
                shock_intensity = max(0.02, rng.normal(0.20, 0.08))
                shock_sensitivity = float(np.clip(shock_intensity, 0.02, 0.50))
            else:
                shock_sensitivity = 0.0

            network_loc            = sp("network_location")
            network_job            = sp("network_job_search")
            weak_ties              = sp("weak_ties_utility")
            digital_comm           = sp("digital_comm_intensity")
            net_signal_susc        = sp("network_signal_susceptibility")
            digital_trust_v        = sp("digital_trust")
            info_quality = float(np.clip(info_quality * 0.55 + digital_comm * 0.45, 0.0, 1.0))
            digital_trust_v = float(np.clip(digital_trust_v * 0.75 + digital_comm * 0.25, 0.0, 1.0))

            # ── Inertia ───────────────────────────────────────────────────────
            age_base         = rogers_castro_mobility(age)
            inertia_from_age = float(np.clip(1.0 - age_base, 0.1, 0.95))
            tenure_mean      = min(12 + age * 1.0, 180)
            tenure           = int(np.clip(rng.exponential(tenure_mean), 0, 420))
            tenure_bonus     = (1.0 - perceived_control) * math.log1p(tenure / 12) * 0.1
            owns_property    = bool(rng.random() < owner_share) and age >= 25

            inertia = float(np.clip(
                inertia_from_age * 0.70 +
                inertia_social   * 0.15 +
                tenure_bonus     * 0.15 +
                (0.07 if owns_property else 0.0),
                0.05, 0.95
            ))
            if marital == "married":
                inertia = float(np.clip(inertia + 0.08, 0.05, 0.95))

            w_econ   = d_econ_weight
            w_social = d_social_weight
            w_family = family_modifier
            w_future = d_future_value
            w_total  = w_econ + w_social + w_family + w_future + 1e-9
            w_econ  /= w_total; w_social /= w_total
            w_family /= w_total; w_future /= w_total

            sat_init   = float(np.clip(satisfaction_base + rng.normal(0, 0.06), 0.05, 0.99))
            econ_value = float(np.clip(1.0 - d_econ_gap + rng.normal(0, 0.05), 0.0, 1.0))

            thr_place_val = float(np.clip(
                0.28 + 0.25 * (d_future_place - 0.316),
                0.05, 0.85
            ))
            thr_social_val = float(np.clip(
                0.35 + rng.normal(0, 0.04), 0.15, 0.85
            ))
            thr_family_val = float(np.clip(
                0.35 + rng.normal(0, 0.04), 0.15, 0.85
            ))

            records.append({
                "id":                  agent_id,
                "residence_district":  residence,
                "workplace_district":  workplace,
                "region":              region_code,
                "district":            residence,
                "status":              status,
                "graduation_cohort":   graduation_cohort,
                "age":                 round(age, 2),
                "sex":                 sex,
                "education":           education,
                "nationality":         nationality,
                "marital":             marital,
                "is_employed":         is_employed,
                "industry":            industry,
                "wage":                round(wage, 2),
                "owns_property":       owns_property,
                "inertia":             round(inertia, 4),
                "tenure":              tenure,
                "moved_ticks":         999,
                "intention_state":     "none",
                "dst_work":            "",
                "activation_timer":    0,
                "activation_domain":   "",
                "social_boost":        0.0,
                "sb_pending":          "",
                "sat_economic":        round(econ_value, 4),
                "sat_social":          round(sat_init, 4),
                "sat_family":          round(sat_init, 4),
                "sat_place":           round(d_future_place, 4),
                "w_economic":          round(w_econ, 4),
                "w_social":            round(w_social, 4),
                "w_family":            round(w_family, 4),
                "w_future":            round(w_future, 4),
                "thr_economic":        round(d_econ_threshold, 4),
                "thr_social":          round(thr_social_val, 4),
                "thr_family":          round(thr_family_val, 4),
                "thr_place":           round(thr_place_val, 4),
                "perceived_control":       round(perceived_control, 4),
                "econ_perceived_control":  round(econ_perceived_control, 4),
                "inertia_social":          round(inertia_social, 4),
                "info_quality":            round(info_quality, 4),
                "commuter_threshold":  round(commuter_threshold, 4),
                "internal_mig_thr":    round(internal_mig_thr, 4),
                "external_mig_thr":    round(external_mig_thr, 4),
                "job_flexibility":     round(job_flexibility, 4),
                "family_weight_mod":   round(family_modifier, 4),
                "shock_sensitivity":   round(shock_sensitivity, 4),
                "network_location":    float(network_loc),
                "network_job_search":  float(network_job),
                "weak_ties_utility":   round(weak_ties, 4),
                "network_signal":      "neutral",
                "net_signal_susc":     round(net_signal_susc, 4),
                "digital_comm":        round(digital_comm, 4),
                "digital_trust":       round(digital_trust_v, 4),
                "housing_price_m2":    round(housing_m2, 0),
                "aspirations":         0.0,
                "signal_reduction":    0.0,
                "place_deficit_penalty": 0.0,
                "tpb_active":          False,
                "intention_delay":     0,
                "econ_gap":            round(float(d_econ_gap), 4),
                "domain_future_place": round(float(d_future_place), 4),
                "econ_penalty":            0.0,
                "infra_bonus":             0.0,
                "inertia_mobility_penalty": 0.0,
                "jobloss_econ_gap_bonus":  0.0,
                "soc_calibration_signal":  0.0,
                "migration_pressure":   0.0,
            })
            agent_id += 1

        # ══════════════════════════════════════════════════════════════════════
        # 1. СТУДЕНТЫ
        # ══════════════════════════════════════════════════════════════════════
        for _ in range(tg["student"]):
            idx = rng.choice(len(stu_keys), p=stu_weights)
            key = stu_keys[idx]
            bin_label, sex = key.split("|")
            mid, width, _ = AGE_BIN_META[bin_label]
            age = float(np.clip(mid + rng.uniform(-width / 2, width / 2), 15, 29))

            workplace = _sample_schoolplace(residence, school_outflow, rng)

            if age < 18:
                graduation_cohort = 4   # тик 48
            elif age < 20:
                graduation_cohort = 3   # тик 36
            elif age < 22:
                graduation_cohort = 2   # тик 24
            else:
                graduation_cohort = 1   # тик 12

            make_agent(age, sex, "student", False, workplace, "Education", 0.0,
                       graduation_cohort)

        # ══════════════════════════════════════════════════════════════════════
        # 2. ЗАНЯТЫЕ
        # ══════════════════════════════════════════════════════════════════════
        for _ in range(tg["employed"]):
            idx = rng.choice(len(work_keys), p=work_weights)
            key = work_keys[idx]
            bin_label, sex = key.split("|")
            mid, width, _ = AGE_BIN_META[bin_label]
            age = float(np.clip(mid + rng.uniform(-width / 2, width / 2), 15, 65))

            workplace = _sample_workplace(residence, outflow_probs, rng)
            status = "stay" if workplace == residence else "commute"

            wp_data       = init_dists.get(workplace, dd)
            industry_dist = wp_data.get("industry", {})
            salary_by_ind = wp_data.get("salary_by_industry", {})
            avg_wage_wp   = wp_data.get("avg_wage", 1400.0)

            industry = _weighted_choice(industry_dist, rng) if industry_dist else "Other"
            base_wage = salary_by_ind.get(industry, avg_wage_wp)
            # Семплируем education заранее для корректного edu_mult
            edu_pre = _weighted_choice(edu_dist, rng) if edu_dist else "medium"
            edu_mult = {"low": 0.82, "medium": 1.0, "high": 1.35}.get(edu_pre, 1.0)
            wage = float(max(0, rng.normal(base_wage * edu_mult, base_wage * 0.22)))

            make_agent(age, sex, status, True, workplace, industry, wage,
                       education_override=edu_pre)

        # ══════════════════════════════════════════════════════════════════════
        # 3. БЕЗРАБОТНЫЕ
        # ══════════════════════════════════════════════════════════════════════
        for _ in range(tg["unemployed"]):
            idx = rng.choice(len(work_keys), p=work_weights)
            key = work_keys[idx]
            bin_label, sex = key.split("|")
            mid, width, _ = AGE_BIN_META[bin_label]
            age = float(np.clip(mid + rng.uniform(-width / 2, width / 2), 15, 65))

            res_industry_dist = dd.get("industry", {})
            industry = _weighted_choice(res_industry_dist, rng) if res_industry_dist else "Other"

            make_agent(age, sex, "unemployed", False, residence, industry, 0.0)

    df = pd.DataFrame(records)
    print(f"\n  Создано агентов: {len(df):,}  |  Районов: {df['residence_district'].nunique()}")
    _print_summary(df)

    # v3: Инициализируем INDUSTRY_JOBS_CAPACITY с occupied/vacant по отраслям
    # из данных init_dists (population, employed_share, unemployed_share, industry shares).
    # Это заменяет старый _scale_jobs_capacity — отраслевая структура точнее,
    # а commuting-матрица используется только для распределения агентов по workplace.
    _init_industry_jobs(init_dists, n_agents)

    return df


# ── Диагностика ───────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame):
    print(f"  Возраст: {df['age'].min():.0f}–{df['age'].max():.0f}  "
          f"mean={df['age'].mean():.1f}  median={df['age'].median():.1f}")
    print(f"  Ср. инерция:      {df['inertia'].mean():.3f}")
    print(f"  Ср. percontrol:   {df['perceived_control'].mean():.3f}")

    # Статусы занятости
    status_counts = df["status"].value_counts()
    print(f"\n  СТАТУСЫ ЗАНЯТОСТИ:")
    for s, n in status_counts.items():
        print(f"    {s:<12}: {n:>7,}  ({n / len(df) * 100:.1f}%)")

    # Commute диагностика
    commuters = df[df["status"] == "commute"]
    if len(commuters) > 0:
        print(f"\n  МАЯТНИКОВЫЕ (commute): {len(commuters):,}")
        top_flows = (commuters
                     .groupby(["residence_district", "workplace_district"])
                     .size()
                     .sort_values(ascending=False)
                     .head(5))
        print("  Топ-5 потоков:")
        for (res, wp), cnt in top_flows.items():
            r = res.replace("District of ", "")
            w = wp.replace("District of ", "")
            print(f"    {r} → {w}: {cnt:,}")

    # Student диагностика
    students = df[df["status"] == "student"]
    if len(students) > 0:
        print(f"\n  СТУДЕНТЫ: {len(students):,}  "
              f"ср.возраст={students['age'].mean():.1f}  "
              f"когорты: " + ", ".join(
                  f"к{int(k)}={int(v)}" for k, v in
                  students['graduation_cohort'].value_counts().sort_index().items()))
        top_school_flows = (students
                            .groupby(["residence_district", "workplace_district"])
                            .size()
                            .sort_values(ascending=False)
                            .head(5))
        print("  Топ-5 студенческих потоков:")
        for (res, wp), cnt in top_school_flows.items():
            r = res.replace("District of ", "")
            w = wp.replace("District of ", "")
            print(f"    {r} → {w}: {cnt:,}")

    # Зарплата
    emp = df[df["wage"] > 0]
    print(f"\n  Занятых: {df['is_employed'].mean():.1%}  "
          f"ср. зарплата {emp['wage'].mean():,.0f}€  "
          f"p25={emp['wage'].quantile(.25):,.0f}€  "
          f"p75={emp['wage'].quantile(.75):,.0f}€")

    print(f"  owns_property:    {df['owns_property'].mean():.1%}")
    print(f"  network_location: {df['network_location'].mean():.1%}")

    print(f"\n  Топ-5 отраслей (workplace):")
    emp_ind = df[df["is_employed"]]["industry"].value_counts().head(5)
    for ind, n in emp_ind.items():
        print(f"    {str(ind)[:50]:<50}: {n:>5,}")
