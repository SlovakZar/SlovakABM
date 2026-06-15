"""
agents.py v8 — FFT Architecture

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

# Возрастные бины SODB в диапазоне 18–65
# (midpoint, width, fraction_of_bin_to_use)
AGE_BIN_META = {
    "15 - 19 years": (18.5, 2, 2 / 5),
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


def _get_init_dists(path="agent_init_distributions.json"):
    global _INIT_DISTS
    if _INIT_DISTS is None:
        p = Path(path)
        if not p.exists():
            p = Path(__file__).parent / path
        with open(p, encoding="utf-8") as f:
            _INIT_DISTS = json.load(f)["districts"]
    return _INIT_DISTS


def _get_survey(path="agent_params_from_survey.json"):
    global _SURVEY
    if _SURVEY is None:
        p = Path(path)
        if not p.exists():
            p = Path(__file__).parent / path
        with open(p, encoding="utf-8") as f:
            _SURVEY = json.load(f)
    return _SURVEY


def _get_commuting(path="commuting_filtered_with_travel.csv"):
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


# ── Школьные/студенческие потоки ──────────────────────────────────────────────

_SCHOOL_OUTFLOW = None
_ENROLLMENT_RATES: dict = {}   # {district: {age_bin: rate}}


def _compute_school_outflow(path="commuting_filtered_with_travel.csv"):
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


def _infer_agent_type(age: float, education: str, marital: str,
                      perceived_control: float, inertia: float) -> str:
    if inertia > 0.65 and age > 40:
        return "anchored"
    if marital == "married" and age > 30:
        return "family_first"
    if perceived_control > 0.6 and education == "high":
        return "seeker"
    return "waiting"


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
    commuting_path: str = "commuting_filtered_with_travel.csv",
    n_agents: int = 70000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    init_dists   = _get_init_dists(dist_path)
    _get_survey(survey_path)
    outflow_probs = _get_commuting(commuting_path)
    school_outflow, _ = _compute_school_outflow(commuting_path)

    districts = list(init_dists.keys())

    # Эффективная популяция 18–65 по каждому району
    pop_1865 = {}
    for d in districts:
        age_sex = init_dists[d].get("age_sex", {})
        total = sum(
            meta["count"] * AGE_BIN_META[key.split("|")[0]][2]
            for key, meta in age_sex.items()
            if key.split("|")[0] in AGE_BIN_META
        )
        pop_1865[d] = max(1, total)

    total_pop = sum(pop_1865.values())
    agents_per_d = {d: max(1, round(n_agents * pop_1865[d] / total_pop))
                    for d in districts}
    diff = n_agents - sum(agents_per_d.values())
    agents_per_d[districts[0]] += diff

    records  = []
    agent_id = 0

    for residence in districts:
        n  = agents_per_d[residence]
        dd = init_dists[residence]
        region_code = DISTRICT_TO_REGION_CODE.get(residence, "XX")

        # Возрастные бины
        age_sex     = dd.get("age_sex", {})
        valid_keys  = [k for k in age_sex if k.split("|")[0] in AGE_BIN_META]
        if not valid_keys:
            continue

        weights = np.array([
            age_sex[k]["count"] * AGE_BIN_META[k.split("|")[0]][2]
            for k in valid_keys
        ], dtype=float)
        if weights.sum() == 0:
            continue
        weights /= weights.sum()

        sampled_idx    = rng.choice(len(valid_keys), size=n, p=weights)
        edu_dist       = dd.get("education", {"low": 0.3, "medium": 0.5, "high": 0.2})
        unemployment_r = dd.get("employment", {}).get("unemployed_share", 0.06)
        owner_share    = dd.get("owner_share", 0.65)
        housing_m2     = dd.get("housing_price_m2", 1500.0)
        nat_dist_d     = dd.get("nationality", {"Slovak": 1.0})

        for idx in sampled_idx:
            key = valid_keys[idx]
            bin_label, sex = key.split("|")
            mid, width, _ = AGE_BIN_META[bin_label]
            age = float(np.clip(mid + rng.uniform(-width / 2, width / 2), 18, 65))

            education   = _weighted_choice(edu_dist, rng) if edu_dist else "medium"
            marital_sex = dd.get("marital", {}).get(sex, {})
            marital     = MARITAL_MAP.get(
                _weighted_choice(marital_sex, rng) if marital_sex else "Never married",
                "single"
            )
            nationality = _weighted_choice(nat_dist_d, rng)

            # ── Шаг 0a: ОТРАСЛЕВАЯ СПЕЦИАЛИЗАЦИЯ (для всех, кроме студентов) ─
            # Безработные получают отрасль из residence_district — они потеряли
            # работу в своей отрасли, но сохраняют специализацию.
            # Для занятых отрасль позже переопределяется из workplace_district.
            res_industry_dist = dd.get("industry", {})
            agent_industry = _weighted_choice(res_industry_dist, rng) if res_industry_dist else "Other"

            # ── Шаг 0b: СТУДЕНТ? ─────────────────────────────────────────────
            is_student   = False
            graduation_tick_val = -1  # -1 = не студент

            if age <= 26:
                if age < 20:
                    age_bin_key = "15-19"
                elif age < 25:
                    age_bin_key = "20-24"
                else:
                    age_bin_key = "25-29"

                enroll_r = _ENROLLMENT_RATES.get(residence, {}).get(age_bin_key, 0.05)
                is_student = rng.random() < enroll_r

            if is_student:
                # ── Студент ──────────────────────────────────────────────────
                status    = "student"
                is_employed = False
                workplace = _sample_schoolplace(residence, school_outflow, rng)
                industry  = "Education"
                wage      = 0.0

                # Дата выпуска: тики от текущего возраста до graduation_age
                if age < 19:
                    grad_age = 19.0   # средняя школа
                elif age < 22:
                    grad_age = 22.0   # бакалавриат
                else:
                    grad_age = 24.0   # магистратура

                graduation_tick_val = max(1, int(round((grad_age - age) * 12)))
            else:
                # ── Не студент: обычная занятость ────────────────────────────
                if age < 25:
                    inactivity_r = 0.35
                elif age < 55:
                    inactivity_r = 0.10
                else:
                    inactivity_r = 0.22

                p_employed = float(np.clip(
                    (1.0 - inactivity_r) * (1.0 - unemployment_r), 0.0, 1.0
                ))
                is_employed = rng.random() < p_employed

                if is_employed:
                    workplace = _sample_workplace(residence, outflow_probs, rng)
                    status    = "stay" if workplace == residence else "commute"
                else:
                    workplace = residence
                    status    = "unemployed"

                # ── Отрасль и зарплата ───────────────────────────────────────
                if is_employed:
                    # Занятый: отрасль из workplace_district (реальная работа)
                    wp_data        = init_dists.get(workplace, dd)
                    industry_dist  = wp_data.get("industry", {})
                    salary_by_ind  = wp_data.get("salary_by_industry", {})
                    avg_wage_wp    = wp_data.get("avg_wage", 1400.0)

                    industry = _weighted_choice(industry_dist, rng) if industry_dist else agent_industry
                    base_wage = salary_by_ind.get(industry, avg_wage_wp)
                    edu_mult  = {"low": 0.82, "medium": 1.0, "high": 1.35}.get(education, 1.0)
                    wage = float(max(0, rng.normal(base_wage * edu_mult, base_wage * 0.22)))
                else:
                    # Безработный: сохраняет отраслевую специализацию из residence_district
                    industry = agent_industry
                    wage     = 0.0

            # ── SASD параметры ────────────────────────────────────────────────
            settlement = SETTLEMENT_MAP.get(residence, "town")
            def sp(name): return sample_param(name, age, education, region_code,
                                              settlement, rng)

            perceived_control      = sp("perceived_control")
            econ_perceived_control = sp("econ_perceived_control")

            # ═══ Пункт 6: условная связь perceived_control → econ_perceived_control ═══
            # В реальности они коррелируют ~0.4: общий локус контроля влияет на
            # восприятие контроля в рабочей сфере. Смешиваем 60% независимого
            # семпла + 40% от perceived_control чтобы избежать нереалистичных
            # комбинаций (pc=0.9, epc=0.1).
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

            # ═══ v8: future_orientation → internal_mig_threshold (r≈0.3) ═══
            # Оптимистичные люди имеют более низкий порог внутренней миграции.
            internal_mig_thr = float(np.clip(
                internal_mig_thr + (1.0 - d_future_value) * 0.20, 0.0, 1.0
            ))
            job_flexibility        = sp("job_flexibility_threshold")
            shock_sensitivity      = sp("inertia_shock_sensitivity")
            satisfaction_base      = sp("satisfaction_init")

            # ═══ Пункт 4 (v8): Hurdle модель для shock_sensitivity ═══
            # В опросе большинство не переживало шок → mean≈0.05.
            # Normal + clip создаёт массу искусственных нулей.
            # Правильно: Bernoulli(был ли шок) × intensity(насколько сильный).
            # p_shock — из данных: доля респондентов с shock > 0 после norm01.
            # Используем survey-параметр если доступен, иначе консервативно 0.30.
            p_shock_raw = _get_survey().get("inertia_shock_sensitivity", {}).get("p_shock", 0.30)
            if rng.random() < p_shock_raw:
                # Интенсивность шока: усечённый normal со средним ~0.20
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

            # ═══ v8: условные связи цифровых параметров ═══
            # digital_comm → info_quality: активные пользователи лучше информированы (r≈0.5)
            info_quality = float(np.clip(info_quality * 0.55 + digital_comm * 0.45, 0.0, 1.0))
            # digital_comm → digital_trust: больше пользуешься → больше доверяешь (r≈0.3)
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

            agent_type = _infer_agent_type(age, education, marital,
                                           perceived_control, inertia)

            # Веса доменов с модификаторами типа агента
            type_modifiers = {
                "seeker":       {"econ": 1.2, "social": 1.0, "family": 0.8, "future": 1.1, "place": 0.8},
                "waiting":      {"econ": 1.0, "social": 0.9, "family": 1.0, "future": 0.9, "place": 1.0},
                "anchored":     {"econ": 0.7, "social": 1.1, "family": 1.2, "future": 0.7, "place": 1.5},
                "family_first": {"econ": 0.9, "social": 0.9, "family": 1.4, "future": 0.9, "place": 1.2},
            }
            mod = type_modifiers.get(agent_type,
                                     {"econ": 1.0, "social": 1.0, "family": 1.0, "future": 1.0, "place": 1.0})

            w_econ   = d_econ_weight * mod["econ"]
            w_social = d_social_weight * mod["social"]
            w_family = family_modifier * mod["family"]
            w_future = d_future_value * mod["future"]
            w_total  = w_econ + w_social + w_family + w_future + 1e-9
            w_econ  /= w_total; w_social /= w_total
            w_family /= w_total; w_future /= w_total

            sat_init   = float(np.clip(satisfaction_base + rng.normal(0, 0.06), 0.05, 0.99))
            econ_value = float(np.clip(1.0 - d_econ_gap + rng.normal(0, 0.05), 0.0, 1.0))

            # ── Порог place: база × тип агента + шум от place_aspiration ─────
            # domain_future_place μ≈0.316, σ≈0.185 — широкий разброс
            # множители типа: seeker=0.8, waiting=1.0, anchored=1.5, family_first=1.2
            thr_place_val = float(np.clip(
                0.28 * mod["place"] + 0.25 * (d_future_place - 0.316),
                0.05, 0.85
            ))

            # ═══ Блок E: индивидуальные пороги social/family по типу агента ═══
            thr_social_map = {
                "seeker": 0.30, "waiting": 0.35, "anchored": 0.55, "family_first": 0.40
            }
            thr_family_map = {
                "seeker": 0.25, "waiting": 0.35, "anchored": 0.50, "family_first": 0.60
            }
            thr_social_val = float(np.clip(
                thr_social_map.get(agent_type, 0.35) + rng.normal(0, 0.04), 0.15, 0.85
            ))
            thr_family_val = float(np.clip(
                thr_family_map.get(agent_type, 0.35) + rng.normal(0, 0.04), 0.15, 0.85
            ))

            records.append({
                # ── Идентификация ────────────────────────────────────────────
                "id":                  agent_id,

                # ── Локация (НОВОЕ: два поля) ─────────────────────────────────
                "residence_district":  residence,
                "workplace_district":  workplace,
                "region":              region_code,
                # Обратная совместимость с engine/report — текущий "home"
                "district":            residence,

                # ── Статус занятости (НОВОЕ) ──────────────────────────────────
                "status":              status,   # stay | commute | unemployed | student
                "graduation_tick":     graduation_tick_val,  # -1 = не студент

                # ── Демография ───────────────────────────────────────────────
                "age":                 round(age, 2),
                "sex":                 sex,
                "education":           education,
                "nationality":         nationality,
                "marital":             marital,

                # ── Занятость ────────────────────────────────────────────────
                "is_employed":         is_employed,
                "industry":            industry,
                "wage":                round(wage, 2),
                "owns_property":       owns_property,

                # ── Тип агента ───────────────────────────────────────────────
                "agent_type":          agent_type,

                # ── Инерция и стаж ───────────────────────────────────────────
                "inertia":             round(inertia, 4),
                "tenure":              tenure,
                "moved_ticks":         999,  # давно на месте

                # ── TPB состояние (ОБНОВЛЕНО для FFT) ────────────────────────
                # "none"              — агент не активен
                # "seeking_work"      — Фильтр 1 открылся, ищет dst_work
                # "seeking_residence" — dst_work найден, ищет жильё
                # "commute_pending"   — решил на commute, проверяет feasibility
                "intention_state":     "none",
                "dst_work":            "",    # целевой район работы (пуст до активации)
                "activation_timer":    0,     # Блок D: счётчик тиков ожидания (inertia-задержка)
                "activation_domain":   "",    # доминантный домен при активации (economic/place/social/family)
                "social_boost":        0.0,   # Блок B: временный буст social target от событий
                "sb_pending":          "",    # v2: очередь активных decay-потоков social_boost ("M5,M3,C2" формат)

                # ── Домены satisfaction ───────────────────────────────────────
                "sat_economic":        round(econ_value, 4),
                "sat_social":          round(sat_init, 4),
                "sat_family":          round(sat_init, 4),
                "sat_place":           round(d_future_place, 4),

                # ── Веса доменов ─────────────────────────────────────────────
                "w_economic":          round(w_econ, 4),
                "w_social":            round(w_social, 4),
                "w_family":            round(w_family, 4),
                "w_future":            round(w_future, 4),

                # ── Пороги доменов ───────────────────────────────────────────
                "thr_economic":        round(d_econ_threshold, 4),
                "thr_social":          round(thr_social_val, 4),
                "thr_family":          round(thr_family_val, 4),
                "thr_place":           round(thr_place_val, 4),

                # ── Психологические параметры (SASD) ──────────────────────────
                "perceived_control":       round(perceived_control, 4),
                "econ_perceived_control":  round(econ_perceived_control, 4),
                "inertia_social":          round(inertia_social, 4),
                "info_quality":            round(info_quality, 4),

                # ── Пороги мобильности ───────────────────────────────────────
                "commuter_threshold":  round(commuter_threshold, 4),
                "internal_mig_thr":    round(internal_mig_thr, 4),
                "external_mig_thr":    round(external_mig_thr, 4),
                "job_flexibility":     round(job_flexibility, 4),
                "family_weight_mod":   round(family_modifier, 4),

                # ── Инерционные параметры ────────────────────────────────────
                "shock_sensitivity":   round(shock_sensitivity, 4),

                # ── Сеть ─────────────────────────────────────────────────────
                "network_location":    float(network_loc),
                "network_job_search":  float(network_job),
                "weak_ties_utility":   round(weak_ties, 4),
                "network_signal":      "neutral",
                "net_signal_susc":     round(net_signal_susc, 4),

                # ── Цифровые параметры ───────────────────────────────────────
                "digital_comm":        round(digital_comm, 4),
                "digital_trust":       round(digital_trust_v, 4),

                # ── Жильё ────────────────────────────────────────────────────
                "housing_price_m2":    round(housing_m2, 0),

                # ── Двухбарьерная модель (Aspirations×Capabilities → TPB) ────
                "aspirations":         0.0,          # EWMA-накопление D_instant (обновляется в engine.tick)
                "signal_reduction":    0.0,          # накопленный эффект соц. сигналов, снижающий инерцию
                "place_deficit_penalty": 0.0,        # накопленный штраф за неудовлетворённость местом
                "tpb_active":          False,        # флаг фазы TPB
                "intention_delay":     0,            # счётчик тиков задержки после превышения порога намерения
                "econ_gap":            round(float(d_econ_gap), 4),  # адаптивное восприятие econ-разрыва
                "domain_future_place": round(float(d_future_place), 4),  # адаптивные ожидания места

                # ── Динамические переменные сигнальной системы v2 ────────────
                "econ_penalty":            0.0,   # динамический штраф к D_econ
                "infra_bonus":             0.0,   # динамический бонус к инфраструктуре
                "inertia_mobility_penalty": 0.0,  # динамический штраф к инерции от переездов соседей
                "jobloss_econ_gap_bonus":  0.0,   # временный бонус к econ_gap от LOST_JOB
            })
            agent_id += 1

    df = pd.DataFrame(records)
    print(f"\n  Создано агентов: {len(df):,}  |  Районов: {df['residence_district'].nunique()}")
    _print_summary(df)

    # Масштабируем JOBS_CAPACITY от реальной популяции к числу агентов,
    # чтобы jobs_pressure колебался вокруг 1.0, а не 0.03
    _scale_jobs_capacity(n_agents)

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
              f"ср.выпуск через {students['graduation_tick'].mean():.0f} тиков")
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

    print(f"\n  Типы агентов:")
    for t, n in df["agent_type"].value_counts().items():
        print(f"    {t:<14}: {n:>7,}  ({n / len(df) * 100:.1f}%)")

    print(f"\n  Топ-5 отраслей (workplace):")
    emp_ind = df[df["is_employed"]]["industry"].value_counts().head(5)
    for ind, n in emp_ind.items():
        print(f"    {str(ind)[:50]:<50}: {n:>5,}")
