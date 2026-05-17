"""
agents.py v2 — агент с полной TPB-структурой по документу модели v7.
Адаптирован для загрузки демографических распределений из agent_init_distributions.json
и расчёта зарплаты через salary_by_industry.
"""

import json
import random
import math
import numpy as np
import pandas as pd
from pathlib import Path

# ── Константы ─────────────────────────────────────────────────────────────────

BERNOULLI_PARAMS = {'network_job_search', 'network_location'}

# Rogers & Castro 1981 — параметры возрастной кривой мобильности
RC_A1    = 0.09
RC_MU1   = 22.0
RC_ALPHA1= 0.10
RC_A2    = 0.01
RC_MU2   = 65.0
RC_ALPHA2= 0.07
RC_C     = 0.005

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

# ── Загрузка параметров из JSON (SASD) ───────────────────────────────────────

def load_survey_params(json_path: str = "agent_params_from_survey.json") -> dict:
    path = Path(json_path)
    if not path.exists():
        path = Path(__file__).parent / json_path
    with open(path, encoding='utf-8') as f:
        return json.load(f)

_SURVEY = None

def _get_survey() -> dict:
    global _SURVEY
    if _SURVEY is None:
        _SURVEY = load_survey_params()
    return _SURVEY

def sample_param(
    name: str,
    age: float,
    education: str,
    region: str = None,
    rng: np.random.Generator = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng()
    survey = _get_survey()
    p = survey.get(name)
    if not p:
        return 0.5
    ag = ("18-30" if age < 31 else
          "31-45" if age < 46 else
          "46-60" if age < 61 else "60+")
    eg = ("low"    if education in ["Without education", "Basic"]
          else "medium" if "Secondary" in education
          else "high")
    g = p.get("by_group", {}).get(str((ag, eg)))
    mu = float(g["mean"]) if g else float(p.get("global_mean", 0.5))
    sd = float(g["std"])  if g else float(p.get("global_std", 0.1))
    n  = int(g["n"])      if g else 100
    regional = survey.get("_regional", {}).get(name, {})
    if region and regional:
        r = regional.get(region)
        if r and r.get("n", 0) > 5:
            se_g = sd / max(n ** 0.5, 1)
            se_r = r["std"] / max(r["n"] ** 0.5, 1)
            w_g  = 1.0 / max(se_g, 1e-6)
            w_r  = 1.0 / max(se_r, 1e-6)
            mu   = (w_g * mu + w_r * r["mean"]) / (w_g + w_r)
            sd   = max(sd, r["std"]) * 0.8
    if name in BERNOULLI_PARAMS:
        return float(rng.random() < np.clip(mu, 0, 1))
    else:
        val = rng.normal(mu, max(sd, 0.01))
        return float(np.clip(val, 0.0, 1.0))

# ── Rogers & Castro возрастная кривая ─────────────────────────────────────────

def rogers_castro_mobility(age: float) -> float:
    if age < 5:
        return 0.0
    labour = RC_A1 * math.exp(-RC_ALPHA1 * (age - RC_MU1)) if age >= RC_MU1 else \
             RC_A1 * math.exp(-RC_ALPHA1 * (RC_MU1 - age) * 0.3)
    pension = RC_A2 * math.exp(-RC_ALPHA2 * abs(age - RC_MU2))
    raw = labour + pension + RC_C
    return float(np.clip(raw / (RC_A1 + RC_C), 0.0, 1.0))

# ── Тип агента ────────────────────────────────────────────────────────────────

def _infer_agent_type(
    age: float,
    education: str,
    marital: str,
    perceived_control: float,
    inertia: float,
) -> str:
    if inertia > 0.65 and age > 40:
        return "anchored"
    if marital in ("married", "partner", "Married person", "Married") and age > 30:
        return "family_first"
    if perceived_control > 0.6 and education in ("University", "high"):
        return "seeker"
    return "waiting"

# ── Построение распределений из agent_init_distributions.json ─────────────────

def _build_district_distributions(data_path: str = "agent_init_distributions.json") -> dict:
    """Читает agent_init_distributions.json и строит внутреннюю структуру для create_agents."""
    path = Path(data_path)
    if not path.exists():
        path = Path(__file__).parent / data_path
    with open(path, encoding='utf-8') as f:
        root = json.load(f)
    districts_data = root.get("districts", {})

    dists = {}
    for district, d in districts_data.items():
        region_code = DISTRICT_TO_REGION_CODE.get(district, d.get("region", "XX"))
        # age_sex: ключ (bin_name, sex) -> {'count': int, 'midpoint': float, 'width': float}
        age_sex = {}
        for key, info in d.get("age_sex", {}).items():
            # key формат "15 - 19 years|Female"
            if "|" not in key:
                continue
            bin_name, sex = key.split("|", 1)
            count = info.get("count", 0)
            midpoint = info.get("midpoint", 0.0)
            width = info.get("width", 5.0)
            if count > 0:
                age_sex[(bin_name, sex)] = {
                    "count": count,
                    "midpoint": midpoint,
                    "width": width
                }
        # education: доли low/medium/high
        edu = d.get("education", {})
        edu_dist = {
            "low": edu.get("low", 0.0),
            "medium": edu.get("medium", 0.0),
            "high": edu.get("high", 0.0)
        }
        # industry (заменяет occupations)
        industry = d.get("industry", {})
        # salary_by_industry
        salary_by_industry = d.get("salary_by_industry", {})
        avg_wage = d.get("avg_wage", 1200.0)
        # marital
        marital = d.get("marital", {})
        # nationalities
        nat = d.get("nationality", {})
        # employment (оставим для возможного использования)
        employment = d.get("employment", {})
        # owner_share и housing_price_m2 для будущего
        owner_share = d.get("owner_share", 0.7)
        housing_price_m2 = d.get("housing_price_m2", 2000.0)

        # суммарное население района (для пропорций)
        population = sum(info["count"] for info in age_sex.values())

        dists[district] = {
            "age_sex": age_sex,
            "education": edu_dist,
            "industry": industry,                # вместо occupations
            "salary_by_industry": salary_by_industry,
            "avg_wage": avg_wage,
            "marital": marital,
            "nationalities": nat,
            "employment": employment,
            "owner_share": owner_share,
            "housing_price_m2": housing_price_m2,
            "population": population,
            "region": region_code,
        }
    return dists

# ── Создание агентов ──────────────────────────────────────────────────────────

def create_agents(
    agents_dist_path: str = "agent_init_distributions.json",
    n_agents: int = 70000,
    year: int = 2024,   # year не используется, но оставлен для совместимости
    seed: int = 42,
) -> pd.DataFrame:
    """
    Создаёт популяцию агентов с полной TPB-структурой,
    используя agent_init_distributions.json для демографии.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    dists = _build_district_distributions(agents_dist_path)
    districts = [d for d in dists.keys() if dists[d]["population"] > 0]

    populations = {d: dists[d]["population"] for d in districts}
    total_pop = sum(populations.values())
    agents_per_d = {d: max(1, round(n_agents * populations[d] / total_pop)) for d in districts}
    diff = n_agents - sum(agents_per_d.values())
    agents_per_d[districts[0]] += diff

    records = []
    agent_id = 0

    for district in districts:
        n = agents_per_d[district]
        d = dists[district]
        region_code = d["region"]

        # Подготовка списка возраст-пол ключей и весов
        age_sex_keys = list(d["age_sex"].keys())
        age_sex_weights = [d["age_sex"][k]["count"] for k in age_sex_keys]
        if sum(age_sex_weights) == 0:
            age_sex_weights = [1] * len(age_sex_keys)
        age_sex_w = np.array(age_sex_weights, dtype=float)
        age_sex_w /= age_sex_w.sum()
        sampled_idx = rng.choice(len(age_sex_keys), size=n, p=age_sex_w)

        for idx in sampled_idx:
            bin_name, sex = age_sex_keys[idx]
            age_info = d["age_sex"][(bin_name, sex)]
            mid = age_info["midpoint"]
            width = age_info["width"]
            age = float(np.clip(mid + rng.uniform(-width/2, width/2), 0, 105))

            # Образование
            edu_keys = list(d["education"].keys())
            edu_w = np.array([d["education"][k] for k in edu_keys], dtype=float)
            if edu_w.sum() > 0:
                education = rng.choice(edu_keys, p=edu_w/edu_w.sum())
            else:
                education = "Unknown"

            # Отрасль (заменяет occupation)
            ind_keys = list(d["industry"].keys())
            ind_w = np.array([d["industry"][k] for k in ind_keys], dtype=float)
            if ind_w.sum() > 0:
                occupation = rng.choice(ind_keys, p=ind_w/ind_w.sum())
            else:
                occupation = "Unknown"

            # Национальность
            nat_keys = list(d["nationalities"].keys())
            nat_w = np.array([d["nationalities"][k] for k in nat_keys], dtype=float)
            if nat_w.sum() > 0:
                nationality = rng.choice(nat_keys, p=nat_w/nat_w.sum())
            else:
                nationality = "Slovak"

            # Семейное положение
            marital_by_sex = d["marital"].get(sex, {})
            mar_keys = list(marital_by_sex.keys())
            mar_w = np.array([marital_by_sex[k] for k in mar_keys], dtype=float)
            if mar_w.sum() > 0:
                marital = rng.choice(mar_keys, p=mar_w/mar_w.sum())
            else:
                marital = "Single person"

            # Зарплата на основе отрасли
            salary_by_ind = d["salary_by_industry"]
            base_salary = salary_by_ind.get(occupation, d["avg_wage"])
            if age < 18:
                wage = 0.0
            elif education == "high":   # University
                wage = float(max(0, rng.normal(base_salary * 1.35, base_salary * 0.25)))
            else:
                wage = float(max(0, rng.normal(base_salary, base_salary * 0.28)))

            # ── Параметры из SASD JSON ────────────────────────────────────────
            def sp(name): return sample_param(name, age, education, region_code, rng)

            perceived_control       = sp("perceived_control")
            econ_perceived_control  = sp("econ_perceived_control")
            inertia_social          = sp("inertia_social_component")
            info_quality            = sp("info_quality_modifier")

            d_econ_weight     = sp("domain_economic_weight")
            d_econ_gap        = sp("domain_economic_gap")
            d_econ_threshold  = sp("domain_economic_threshold")
            d_social_weight   = sp("domain_social_weight")
            d_future_value    = sp("domain_future_value")
            d_future_place    = sp("domain_future_place")
            family_modifier   = sp("family_weight_modifier")

            commuter_threshold   = sp("commuter_mode_threshold")
            internal_mig_thr     = sp("internal_mig_threshold")
            external_mig_thr     = sp("external_mig_threshold")
            job_flexibility      = sp("job_flexibility_threshold")
            tenure_loyalty       = sp("tenure_loyalty_bonus")
            shock_sensitivity    = sp("inertia_shock_sensitivity")
            satisfaction_base    = sp("satisfaction_init")
            network_loc          = sp("network_location")
            network_job          = sp("network_job_search")
            weak_ties            = sp("weak_ties_utility")
            digital_comm         = sp("digital_comm_intensity")
            net_signal_susc      = sp("network_signal_susceptibility")
            digital_trust_val    = sp("digital_trust")

            # ── Inertia ───────────────────────────────────────────────────────
            age_base = rogers_castro_mobility(age)
            inertia_from_age = float(np.clip(1.0 - age_base, 0.1, 0.95))
            tenure_mean = min(12 + age * 1.0, 180)
            tenure = int(np.clip(rng.exponential(tenure_mean), 0, 420))
            tenure_bonus = tenure_loyalty * math.log1p(tenure / 12) * 0.1
            inertia = float(np.clip(
                inertia_from_age * 0.5 + inertia_social * 0.4 + tenure_bonus * 0.1,
                0.05, 0.95
            ))
            if marital in ("Married person", "married", "Married"):
                inertia = float(np.clip(inertia + 0.08, 0.05, 0.95))

            # ── Тип агента ────────────────────────────────────────────────────
            agent_type = _infer_agent_type(age, education, marital, perceived_control, inertia)

            # ── Веса доменов (нормировка) ─────────────────────────────────────
            type_modifiers = {
                "seeker":      {"econ": 1.2, "social": 1.0, "family": 0.8, "future": 1.1},
                "waiting":     {"econ": 1.0, "social": 0.9, "family": 1.0, "future": 0.9},
                "anchored":    {"econ": 0.7, "social": 1.1, "family": 1.2, "future": 0.7},
                "family_first":{"econ": 0.9, "social": 0.9, "family": 1.4, "future": 0.9},
            }
            mod = type_modifiers.get(agent_type, {k: 1.0 for k in ["econ","social","family","future"]})
            w_econ   = d_econ_weight * mod["econ"]
            w_social = d_social_weight * mod["social"]
            w_family = family_modifier * mod["family"]
            w_future = d_future_value * mod["future"]
            w_total  = w_econ + w_social + w_family + w_future + 1e-9
            w_econ   /= w_total
            w_social /= w_total
            w_family /= w_total
            w_future /= w_total

            sat_noise = rng.normal(0, 0.06)
            sat_init  = float(np.clip(satisfaction_base + sat_noise, 0.05, 0.99))
            econ_value = float(np.clip(1.0 - d_econ_gap + rng.normal(0, 0.05), 0.0, 1.0))

            intention_state  = "none"
            forming_ticks    = 0
            forming_duration = int(rng.integers(3, 9))

            records.append({
                "id":            agent_id,
                "district":      district,
                "region":        region_code,
                "age":           round(age, 2),
                "sex":           sex,
                "education":     education,
                "occupation":    occupation,
                "nationality":   nationality,
                "marital":       marital,
                "wage":          round(wage, 2),
                "agent_type":    agent_type,
                "inertia":       round(inertia, 4),
                "tenure":        tenure,
                "moved_ticks":   999,
                "intention_state":  intention_state,
                "forming_ticks":    forming_ticks,
                "forming_duration": forming_duration,
                "sat_economic":   round(econ_value, 4),
                "sat_social":     round(sat_init, 4),
                "sat_family":     round(sat_init, 4),
                "sat_place":      round(d_future_place, 4),
                "w_economic":     round(w_econ, 4),
                "w_social":       round(w_social, 4),
                "w_family":       round(w_family, 4),
                "w_future":       round(w_future, 4),
                "thr_economic":   round(d_econ_threshold, 4),
                "thr_social":     round(0.35, 4),
                "thr_family":     round(0.35, 4),
                "thr_place":      round(0.40, 4),
                "perceived_control":      round(perceived_control, 4),
                "econ_perceived_control": round(econ_perceived_control, 4),
                "inertia_social":         round(inertia_social, 4),
                "info_quality":           round(info_quality, 4),
                "commuter_threshold":   round(commuter_threshold, 4),
                "internal_mig_thr":     round(internal_mig_thr, 4),
                "external_mig_thr":     round(external_mig_thr, 4),
                "job_flexibility":      round(job_flexibility, 4),
                "tenure_loyalty":       round(tenure_loyalty, 4),
                "shock_sensitivity":    round(shock_sensitivity, 4),
                "network_location":     float(network_loc),
                "network_job_search":   float(network_job),
                "weak_ties_utility":    round(weak_ties, 4),
                "network_signal":       "neutral",
                "digital_comm":         round(digital_comm, 4),
                "net_signal_susc":      round(net_signal_susc, 4),
                "digital_trust":        round(digital_trust_val, 4),
            })
            agent_id += 1

    df = pd.DataFrame(records)
    print(f"  Создано агентов: {len(df):,}  |  Районов: {df['district'].nunique()}")
    _print_agent_summary(df)
    return df

def _print_agent_summary(df: pd.DataFrame):
    print(f"  Ср. возраст:    {df['age'].mean():.1f}")
    print(f"  Ср. инерция:    {df['inertia'].mean():.3f}")
    print(f"  Ср. percontrol: {df['perceived_control'].mean():.3f}")
    print(f"  Ср. зарплата:   {df[df['wage']>0]['wage'].mean():,.0f}€")
    print(f"  Типы агентов:")
    for t, n in df['agent_type'].value_counts().items():
        print(f"    {t:<14}: {n:>7,}  ({n/len(df)*100:.1f}%)")
    print(f"  network_location (Bernoulli): {df['network_location'].mean():.1%}")
    print(f"  network_job_search:           {df['network_job_search'].mean():.1%}")

if __name__ == "__main__":
    df = create_agents("agent_init_distributions.json", n_agents=70000)
    print("\nРаспределение по регионам:")
    print(df.groupby("region")["id"].count().sort_values(ascending=False).to_string())
