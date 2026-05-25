"""
agents.py v3.1

Изменения относительно v3:
  - Возрастной диапазон агентов: 18–65 лет (не генерируем детей и пенсионеров)
  - Бин "15 - 19 years" обрезается до 18–19 (2/5 от счётчика)
  - Бин "65 - 69 years" обрезается до 65 (1/5 от счётчика)
  - Занятость: derived из unemployment_rate (residence-based)
    Working = True с p = 1 - unemployment_rate
    Not working = False (безработный / неактивный)
  - industry_counts из WorkersCategories — workplace-based,
    используется ТОЛЬКО для распределения отрасли занятого агента,
    не для подсчёта числа занятых
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
INERTIA_WEIGHT_AGE = 0.5
INERTIA_WEIGHT_SOCIAL = 0.4
INERTIA_WEIGHT_TENURE = 0.1
INERTIA_PROPERTY_BONUS = 0.07
THRESHOLD_OFFSET = 0.0
CONTROL_BOOST = 0.0

BERNOULLI_PARAMS = {'network_job_search', 'network_location'}

# Rogers & Castro 1981
RC_A1, RC_MU1, RC_ALPHA1 = 0.09, 22.0, 0.10
RC_A2, RC_MU2, RC_ALPHA2 = 0.01, 65.0, 0.07
RC_C = 0.005

# Бины SODB в диапазоне 18-65 и их доля которую берём
# (midpoint, width, fraction_of_bin_to_use)
AGE_BIN_META = {
    "15 - 19 years":    (18.5, 2, 2/5),   # только 18-19
    "20 - 24 years":    (22.0, 5, 1.0),
    "25 - 29 years":    (27.0, 5, 1.0),
    "30 - 34 years":    (32.0, 5, 1.0),
    "35 - 39 years":    (37.0, 5, 1.0),
    "40 - 44 years":    (42.0, 5, 1.0),
    "45 - 49 years":    (47.0, 5, 1.0),
    "50 - 54 years":    (52.0, 5, 1.0),
    "55 - 59 years":    (57.0, 5, 1.0),
    "60 - 64 years":    (62.0, 5, 1.0),
    "65 - 69 years":    (65.0, 1, 1/5),   # только 65
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

_INIT_DISTS = None
_SURVEY = None


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


def rogers_castro_mobility(age):
    if age < 5:
        return 0.0
    labour = (RC_A1 * math.exp(-RC_ALPHA1 * (age - RC_MU1)) if age >= RC_MU1
              else RC_A1 * math.exp(-RC_ALPHA1 * (RC_MU1 - age) * 0.3))
    pension = RC_A2 * math.exp(-RC_ALPHA2 * abs(age - RC_MU2))
    return float(np.clip((labour + pension + RC_C) / (RC_A1 + RC_C), 0.0, 1.0))


def sample_param(name, age, education, region=None, rng=None):
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

    regional = survey.get("_regional", {}).get(name, {})
    if region and regional:
        r = regional.get(region)
        if r and r.get("n", 0) > 5:
            w_g = 1.0 / max(sd / max(n**0.5, 1), 1e-6)
            w_r = 1.0 / max(r["std"] / max(r["n"]**0.5, 1), 1e-6)
            mu  = (w_g * mu + w_r * r["mean"]) / (w_g + w_r)
            sd  = max(sd, r["std"]) * 0.8

    if name in BERNOULLI_PARAMS:
        return float(rng.random() < float(np.clip(mu, 0, 1)))
    return float(np.clip(rng.normal(mu, max(sd, 0.01)), 0.0, 1.0))


def _weighted_choice(d, rng):
    keys    = list(d.keys())
    weights = np.array([d[k] for k in keys], dtype=float)
    total   = weights.sum()
    if total == 0:
        return keys[0]
    return keys[rng.choice(len(keys), p=weights / total)]


def _infer_agent_type(age, education, marital, perceived_control, inertia):
    if inertia > 0.65 and age > 40:
        return "anchored"
    if marital == "married" and age > 30:
        return "family_first"
    if perceived_control > 0.6 and education == "high":
        return "seeker"
    return "waiting"


def create_agents(
    dist_path="agent_init_distributions.json",
    survey_path="agent_params_from_survey.json",
    n_agents=70000,
    seed=42,
):
    rng = np.random.default_rng(seed)
    init_dists = _get_init_dists(dist_path)
    _get_survey(survey_path)

    districts = list(init_dists.keys())

    # Считаем эффективную популяцию 18-65 по каждому району
    # с учётом частичных бинов
    pop_1865 = {}
    for d in districts:
        age_sex = init_dists[d].get("age_sex", {})
        total = 0
        for key, meta in age_sex.items():
            bin_label = key.split("|")[0]
            if bin_label in AGE_BIN_META:
                fraction = AGE_BIN_META[bin_label][2]
                total += meta["count"] * fraction
        pop_1865[d] = max(1, total)

    total_pop = sum(pop_1865.values())
    agents_per_d = {d: max(1, round(n_agents * pop_1865[d] / total_pop))
                    for d in districts}
    diff = n_agents - sum(agents_per_d.values())
    agents_per_d[districts[0]] += diff

    records  = []
    agent_id = 0

    for district in districts:
        n  = agents_per_d[district]
        dd = init_dists[district]
        region_code = DISTRICT_TO_REGION_CODE.get(district, "XX")

        age_sex = dd.get("age_sex", {})
        # Фильтруем бины 18-65
        valid_keys = [k for k in age_sex if k.split("|")[0] in AGE_BIN_META]
        if not valid_keys:
            continue

        # Веса с учётом частичных бинов
        weights = np.array([
            age_sex[k]["count"] * AGE_BIN_META[k.split("|")[0]][2]
            for k in valid_keys
        ], dtype=float)
        if weights.sum() == 0:
            continue
        weights /= weights.sum()

        sampled_idx = rng.choice(len(valid_keys), size=n, p=weights)

        industry_dist  = dd.get("industry", {})
        salary_by_ind  = dd.get("salary_by_industry", {})
        avg_wage       = dd.get("avg_wage", 1400.0)
        housing_m2     = dd.get("housing_price_m2", 1500.0)
        owner_share    = dd.get("owner_share", 0.65)
        edu_dist       = dd.get("education", {"low": 0.3, "medium": 0.5, "high": 0.2})
        # unemployment_rate — доля незанятых среди трудоспособного населения
        unemployment_r = dd.get("employment", {}).get("unemployed_share", 0.06)

        for idx in sampled_idx:
            key = valid_keys[idx]
            bin_label, sex = key.split("|")
            mid, width, _ = AGE_BIN_META[bin_label]
            age = float(np.clip(mid + rng.uniform(-width / 2, width / 2), 18, 65))

            education = _weighted_choice(edu_dist, rng) if edu_dist else "medium"

            marital_sex = dd.get("marital", {}).get(sex, {})
            marital = MARITAL_MAP.get(
                _weighted_choice(marital_sex, rng) if marital_sex else "Never married",
                "single"
            )

            nat_dist = dd.get("nationality", {"Slovak": 1.0})
            nationality = _weighted_choice(nat_dist, rng)

            # Занятость: p(employed) = (1 - inactivity) * (1 - unemployment_rate)
            # inactivity_rate варьируется по возрасту:
            #   18-24: ~35% (студенты, неактивные)
            #   25-54: ~10% (декрет, инвалидность, другие)
            #   55-65: ~20% (досрочная пенсия, уход за родственниками)
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

            if is_employed and industry_dist:
                industry = _weighted_choice(industry_dist, rng)
                base = salary_by_ind.get(industry, avg_wage)
                edu_mult = {"low": 0.82, "medium": 1.0, "high": 1.35}.get(education, 1.0)
                wage = float(max(0, rng.normal(base * edu_mult, base * 0.22)))
            elif is_employed:
                industry = "Other"
                wage = float(max(0, rng.normal(avg_wage, avg_wage * 0.25)))
            else:
                industry = "Unemployed"
                wage = 0.0

            # SASD параметры
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

            # Inertia
            age_base         = rogers_castro_mobility(age)
            inertia_from_age = float(np.clip(1.0 - age_base, 0.1, 0.95))
            tenure_mean      = min(12 + age * 1.0, 180)
            tenure           = int(np.clip(rng.exponential(tenure_mean), 0, 420))
            tenure_bonus     = tenure_loyalty * math.log1p(tenure / 12) * 0.1
            owns_property    = bool(rng.random() < owner_share) and age >= 25

            inertia = float(np.clip(
                inertia_from_age * 0.5 +
                inertia_social   * 0.4 +
                tenure_bonus     * 0.1 +
                (0.07 if owns_property else 0.0),
                0.05, 0.95
            ))
            if marital == "married":
                inertia = float(np.clip(inertia + 0.08, 0.05, 0.95))

            agent_type = _infer_agent_type(age, education, marital,
                                           perceived_control, inertia)

            type_modifiers = {
                "seeker":      {"econ": 1.2, "social": 1.0, "family": 0.8, "future": 1.1},
                "waiting":     {"econ": 1.0, "social": 0.9, "family": 1.0, "future": 0.9},
                "anchored":    {"econ": 0.7, "social": 1.1, "family": 1.2, "future": 0.7},
                "family_first":{"econ": 0.9, "social": 0.9, "family": 1.4, "future": 0.9},
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
                "id": agent_id, "district": district, "region": region_code,
                "age": round(age, 2), "sex": sex, "education": education,
                "industry": industry, "nationality": nationality, "marital": marital,
                "is_employed": is_employed,
                "wage": round(wage, 2), "owns_property": owns_property,
                "agent_type": agent_type,
                "inertia": round(inertia, 4), "tenure": tenure, "moved_ticks": 999,
                "intention_state": "none", "forming_ticks": 0,
                "forming_duration": int(rng.integers(3, 9)),
                "sat_economic": round(econ_value, 4), "sat_social": round(sat_init, 4),
                "sat_family": round(sat_init, 4), "sat_place": round(d_future_place, 4),
                "w_economic": round(w_econ, 4), "w_social": round(w_social, 4),
                "w_family": round(w_family, 4), "w_future": round(w_future, 4),
                "thr_economic": round(d_econ_threshold, 4),
                "thr_social": 0.35, "thr_family": 0.35, "thr_place": 0.40,
                "perceived_control": round(perceived_control, 4),
                "econ_perceived_control": round(econ_perceived_control, 4),
                "inertia_social": round(inertia_social, 4),
                "info_quality": round(info_quality, 4),
                "commuter_threshold": round(commuter_threshold, 4),
                "internal_mig_thr": round(internal_mig_thr, 4),
                "external_mig_thr": round(external_mig_thr, 4),
                "job_flexibility": round(job_flexibility, 4),
                "tenure_loyalty": round(tenure_loyalty, 4),
                "shock_sensitivity": round(shock_sensitivity, 4),
                "network_location": float(network_loc),
                "network_job_search": float(network_job),
                "weak_ties_utility": round(weak_ties, 4),
                "network_signal": "neutral",
                "digital_comm": round(digital_comm, 4),
                "net_signal_susc": round(net_signal_susc, 4),
                "digital_trust": round(digital_trust_v, 4),
            })
            agent_id += 1

    df = pd.DataFrame(records)
    print(f"  Создано агентов: {len(df):,}  |  Районов: {df['district'].nunique()}")
    _print_summary(df)
    return df


def _print_summary(df):
    print(f"  Возраст: {df['age'].min():.0f}–{df['age'].max():.0f}  "
          f"mean={df['age'].mean():.1f}  median={df['age'].median():.1f}")
    print(f"  Ср. инерция:      {df['inertia'].mean():.3f}")
    print(f"  Ср. percontrol:   {df['perceived_control'].mean():.3f}")
    emp = df[df["wage"] > 0]
    print(f"  Занятых:          {df['is_employed'].mean():.1%}  "
          f"(ср. зарплата {emp['wage'].mean():,.0f}€  "
          f"p25={emp['wage'].quantile(.25):,.0f}€  p75={emp['wage'].quantile(.75):,.0f}€)")
    print(f"  owns_property:    {df['owns_property'].mean():.1%}")
    print(f"  network_location: {df['network_location'].mean():.1%}")
    print(f"  Типы агентов:")
    for t, n in df["agent_type"].value_counts().items():
        print(f"    {t:<14}: {n:>7,}  ({n/len(df)*100:.1f}%)")
    print(f"  Топ-5 отраслей (занятые):")
    emp_ind = df[df['is_employed']]['industry'].value_counts().head(5)
    for ind, n in emp_ind.items():
        print(f"    {str(ind)[:50]:<50}: {n:>5,}")
