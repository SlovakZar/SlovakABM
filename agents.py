"""
agents.py v4 — FFT Architecture

Ключевые изменения относительно v3.1:

1. РАЗДЕЛЕНИЕ ЛОКАЦИИ:
   Каждый агент имеет два поля:
     residence_district  — где живёт (жильё, социальные связи, inertia)
     workplace_district  — где работает (зарплата, отрасль)
   status: "stay" | "commute" | "unemployed"

2. ИНИЦИАЛИЗАЦИЯ ИЗ COMMUTING-МАТРИЦЫ:
   Шаг 1 — workplace_district из реальных flow_work пропорций.
   Если origin == destination → status "stay".
   Если origin != destination → status "commute".
   Безработные → workplace_district = residence_district, status "unemployed".

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
                 region: str = None, rng: np.random.Generator = None) -> float:
    """Сэмплирует параметр агента из SASD-распределений (по группе возраст×образование)."""
    if rng is None:
        rng = np.random.default_rng()
    survey = _get_survey()
    p = survey.get(name)
    if not p:
        return 0.5

    ag = ("18-30" if age < 31 else "31-45" if age < 46
          else "46-60" if age < 61 else "60+")
    eg = education if education in ("low", "medium", "high") else "medium"

    g  = p.get("by_group", {}).get(str((ag, eg)))
    mu = float(g["mean"]) if g else float(p.get("global_mean", 0.5))
    sd = float(g["std"])  if g else float(p.get("global_std", 0.1))
    n  = int(g["n"])      if g else 100

    # Взвешивание с региональным профилем
    regional = survey.get("_regional", {}).get(name, {})
    if region and regional:
        r = regional.get(region)
        if r and r.get("n", 0) > 5:
            w_g = 1.0 / max(sd / max(n ** 0.5, 1), 1e-6)
            w_r = 1.0 / max(r["std"] / max(r["n"] ** 0.5, 1), 1e-6)
            mu  = (w_g * mu + w_r * r["mean"]) / (w_g + w_r)
            sd  = max(sd, r["std"]) * 0.8

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

            # ── Шаг 1: workplace из commuting-матрицы ─────────────────────────
            # Занятость сначала определяем по вероятности
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
                # Workplace из commuting-матрицы
                workplace = _sample_workplace(residence, outflow_probs, rng)
                status    = "stay" if workplace == residence else "commute"
            else:
                workplace = residence
                status    = "unemployed"

            # ── Шаг 2: отрасль и зарплата из workplace_district ───────────────
            if is_employed:
                wp_data        = init_dists.get(workplace, dd)
                industry_dist  = wp_data.get("industry", {})
                salary_by_ind  = wp_data.get("salary_by_industry", {})
                avg_wage_wp    = wp_data.get("avg_wage", 1400.0)

                industry = _weighted_choice(industry_dist, rng) if industry_dist else "Other"
                base_wage = salary_by_ind.get(industry, avg_wage_wp)
                edu_mult  = {"low": 0.82, "medium": 1.0, "high": 1.35}.get(education, 1.0)
                wage = float(max(0, rng.normal(base_wage * edu_mult, base_wage * 0.22)))
            else:
                industry = "Unemployed"
                wage     = 0.0

            # ── SASD параметры ────────────────────────────────────────────────
            def sp(name): return sample_param(name, age, education, region_code, rng)

            perceived_control      = sp("perceived_control")
            econ_perceived_control = sp("econ_perceived_control")
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
            job_flexibility        = sp("job_flexibility_threshold")
            tenure_loyalty         = sp("tenure_loyalty_bonus")
            shock_sensitivity      = sp("inertia_shock_sensitivity")
            satisfaction_base      = sp("satisfaction_init")
            network_loc            = sp("network_location")
            network_job            = sp("network_job_search")
            weak_ties              = sp("weak_ties_utility")
            digital_comm           = sp("digital_comm_intensity")
            net_signal_susc        = sp("network_signal_susceptibility")
            digital_trust_v        = sp("digital_trust")

            # ── Inertia ───────────────────────────────────────────────────────
            age_base         = rogers_castro_mobility(age)
            inertia_from_age = float(np.clip(1.0 - age_base, 0.1, 0.95))
            tenure_mean      = min(12 + age * 1.0, 180)
            tenure           = int(np.clip(rng.exponential(tenure_mean), 0, 420))
            tenure_bonus     = tenure_loyalty * math.log1p(tenure / 12) * 0.1
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
                "seeker":       {"econ": 1.2, "social": 1.0, "family": 0.8, "future": 1.1},
                "waiting":      {"econ": 1.0, "social": 0.9, "family": 1.0, "future": 0.9},
                "anchored":     {"econ": 0.7, "social": 1.1, "family": 1.2, "future": 0.7},
                "family_first": {"econ": 0.9, "social": 0.9, "family": 1.4, "future": 0.9},
            }
            mod = type_modifiers.get(agent_type,
                                     {"econ": 1.0, "social": 1.0, "family": 1.0, "future": 1.0})

            w_econ   = d_econ_weight * mod["econ"]
            w_social = d_social_weight * mod["social"]
            w_family = family_modifier * mod["family"]
            w_future = d_future_value * mod["future"]
            w_total  = w_econ + w_social + w_family + w_future + 1e-9
            w_econ  /= w_total; w_social /= w_total
            w_family /= w_total; w_future /= w_total

            sat_init   = float(np.clip(satisfaction_base + rng.normal(0, 0.06), 0.05, 0.99))
            econ_value = float(np.clip(1.0 - d_econ_gap + rng.normal(0, 0.05), 0.0, 1.0))

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
                "status":              status,   # stay | commute | unemployed

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
                "thr_social":          0.35,
                "thr_family":          0.35,
                "thr_place":           0.40,

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

                # ── Инерционные параметры ────────────────────────────────────
                "tenure_loyalty":      round(tenure_loyalty, 4),
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
            })
            agent_id += 1

    df = pd.DataFrame(records)
    print(f"\n  Создано агентов: {len(df):,}  |  Районов: {df['residence_district'].nunique()}")
    _print_summary(df)
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
