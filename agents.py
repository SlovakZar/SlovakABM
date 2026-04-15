"""
agents.py v2 — агент с полной TPB-структурой по документу модели v7.

Структура агента:
  Статичные: age, sex, education, occupation, nationality, region_origin
  Тип: seeker / waiting / anchored / family_first (латентный)
  
  Домены satisfaction (каждый: value, weight, threshold):
    economic    — зарплата, занятость, жильё
    social      — социальная сеть, отъезд знакомых
    family      — партнёр, дети, синхронизация домохозяйства
    place_future — aspirational gap, видимое будущее места

  Инерция: age_base (Rogers & Castro) + inertia_social_component + tenure_bonus
  
  TPB pipeline: intention_state ∈ {none, forming, strong, move, commute, adapt}
  
  Параметры из SASD JSON:
    perceived_control, econ_perceived_control
    inertia_social_component (составной)
    domain_economic_weight, domain_economic_gap, domain_economic_threshold
    domain_social_weight, domain_future_value, domain_future_place
    family_weight_modifier, work_family_conflict
    commuter_mode_threshold, internal_mig_threshold, external_mig_threshold
    job_flexibility_threshold, tenure_loyalty_bonus
    satisfaction_init, inertia_shock_sensitivity
    info_quality_modifier (составной)
    network_location, network_job_search, weak_ties_utility
    digital_comm_intensity, network_signal_susceptibility, digital_trust
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
# Упрощённая двухкомпонентная версия: трудовая + пенсионная волна
RC_A1    = 0.09    # амплитуда трудовой волны
RC_MU1   = 22.0   # пик мобильности (лет)
RC_ALPHA1= 0.10   # ширина трудовой волны
RC_A2    = 0.01   # амплитуда пенсионной волны
RC_MU2   = 65.0   # пик пенсионной мобильности
RC_ALPHA2= 0.07   # ширина пенсионной волны
RC_C     = 0.005  # базовый уровень

AGE_BIN_MIDPOINTS = {
    "Zero years": 0.5, "From 1 to 4 years": 2.5, "From 5 to 9 years": 7.0,
    "From 10 to 14 years": 12.0, "From 15 to 19 years": 17.0,
    "From 20 to 24 years": 22.0, "From 25 to 29 years": 27.0,
    "From 30 to 34 years": 32.0, "From 35 to 39 years": 37.0,
    "From 40 to 44 years": 42.0, "From 45 to 49 years": 47.0,
    "From 50 to 54 years": 52.0, "From 55 to 59 years": 57.0,
    "From 60 to 64 years": 62.0, "From 65 to 69 years": 67.0,
    "From 70 to 74 years": 72.0, "From 75 to 79 years": 77.0,
    "From 80 to 84 years": 82.0, "From 85 to 89 years": 87.0,
    "From 90 to 94 years": 92.0, "From 95 to 99 years": 97.0,
    "100 years or over": 102.0,
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


# ── Загрузка параметров из JSON ───────────────────────────────────────────────

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
    """
    Сэмплирует параметр агента из SASD JSON.

    Алгоритм:
      1. Определяем (age_group, edu_group) для данного агента
      2. Ищем групповые статистики в by_group
      3. Если есть региональный профиль — взвешиваем через inverse-SE
      4. Сэмплируем из N(mu, sigma) или Bernoulli(p)
    """
    if rng is None:
        rng = np.random.default_rng()

    survey = _get_survey()
    p = survey.get(name)
    if not p:
        return 0.5

    # Демографическая группа
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

    # Региональный профиль (inverse-SE взвешивание)
    regional = survey.get("_regional", {}).get(name, {})
    if region and regional:
        r = regional.get(region)
        if r and r.get("n", 0) > 5:
            se_g = sd / max(n ** 0.5, 1)
            se_r = r["std"] / max(r["n"] ** 0.5, 1)
            w_g  = 1.0 / max(se_g, 1e-6)
            w_r  = 1.0 / max(se_r, 1e-6)
            mu   = (w_g * mu + w_r * r["mean"]) / (w_g + w_r)
            sd   = max(sd, r["std"]) * 0.8  # консервативное SD после взвешивания

    # Сэмплирование
    if name in BERNOULLI_PARAMS:
        return float(rng.random() < np.clip(mu, 0, 1))
    else:
        val = rng.normal(mu, max(sd, 0.01))
        return float(np.clip(val, 0.0, 1.0))


# ── Rogers & Castro возрастная кривая ─────────────────────────────────────────

def rogers_castro_mobility(age: float) -> float:
    """
    Возрастная кривая мобильности Rogers & Castro 1981.
    Возвращает базовую вероятность смены места жительства [0, 1].
    Нормализована так что максимум ≈ 1.0.
    """
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
    """
    Выводит тип агента из комбинации атрибутов.
    Тип — центр распределений, а не жёсткая категория.
    """
    if inertia > 0.65 and age > 40:
        return "anchored"
    if marital in ("married", "partner") and age > 30:
        return "family_first"
    if perceived_control > 0.6 and education in ("University",):
        return "seeker"
    return "waiting"


# ── Вспомогательные функции environment.json ─────────────────────────────────

def _get_latest(data: dict, year: int = 2024):
    if not data: return None
    if year in data: return data[year]
    for y in sorted(data.keys(), reverse=True):
        if data[y] is not None: return data[y]
    return None


def _sample_from_dist(dist: dict, rng) -> str:
    keys    = [k for k, v in dist.items() if v and v > 0]
    weights = [dist[k] for k in keys]
    if not keys:
        return "Unknown"
    idx = rng.choice(len(keys), p=np.array(weights, dtype=float) / sum(weights))
    return keys[idx]


def _build_district_distributions(env: dict, year: int = 2024) -> dict:
    """Строит распределения демографических атрибутов по районам."""
    locations = env.get("locations", {})
    regions   = env.get("regions", {})

    # Семейный статус — уровень региона
    marital_by_region = {}
    for region_name, reg_data in regions.items():
        marital_regional = {}
        for status, sex_data in reg_data.get("marital_status", {}).items():
            if status == "Unknown":
                continue
            for sex in ("male", "female"):
                v = _get_latest(sex_data.get(sex, {}), year) or 0
                marital_regional.setdefault(sex, {})[status] = (
                    marital_regional.get(sex, {}).get(status, 0) + v
                )
        marital_by_region[region_name] = marital_regional

    dists = {}
    for district, data in locations.items():
        if data.get("type") != "district":
            continue

        region_full = data.get("region", "")

        age_sex = {}
        for bin_name in AGE_BIN_MIDPOINTS:
            ag = data.get("age_groups", {}).get(bin_name, {})
            for sex in ("male", "female"):
                age_sex[(bin_name, sex)] = _get_latest(ag.get(sex, {}), year) or 0

        edu_dist = {k: _get_latest(v, year) or 0
                    for k, v in data.get("education", {}).items()
                    if k != "Total" and _get_latest(v, year)}

        occ_dist = {k: _get_latest(v, year) or 0
                    for k, v in data.get("occupations", {}).items()
                    if k != "Total" and _get_latest(v, year)}

        total_wage = _get_latest(data.get("wages", {}).get("Total", {}), year) or 1200

        nat_dist = {}
        for nat, sex_data in data.get("nationalities", {}).items():
            total = sum(_get_latest(sex_data.get(s, {}), year) or 0
                        for s in ("male", "female"))
            if total > 0:
                nat_dist[nat] = total

        marital = marital_by_region.get(region_full, {})

        dists[district] = {
            "age_sex":      age_sex,
            "education":    edu_dist,
            "occupations":  occ_dist,
            "total_wage":   total_wage,
            "marital":      marital,
            "nationalities": nat_dist,
            "population":   sum(age_sex.values()),
            "region":       region_full,
        }

    return dists


# ── Создание агентов ──────────────────────────────────────────────────────────

def create_agents(
    env_path: str = "environment.json",
    n_agents: int = 70000,
    year: int = 2024,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Создаёт популяцию агентов с полной TPB-структурой.

    Каждый агент содержит:
      - Статичные демографические атрибуты из переписи
      - Параметры из SASD JSON (сэмплированные с учётом демографии и региона)
      - Четыре домена satisfaction (value, weight, threshold)
      - Inertia (Rogers & Castro + social component + tenure)
      - TPB pipeline state
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    env_path_obj = Path(env_path)
    if not env_path_obj.exists():
        env_path_obj = Path(__file__).parent / env_path

    with open(env_path_obj, encoding='utf-8') as f:
        env = json.load(f)

    dists = _build_district_distributions(env, year)
    districts = [d for d in dists.keys() if dists[d]["population"] > 0]

    # Пропорциональное распределение агентов по районам
    populations  = {d: dists[d]["population"] for d in districts}
    total_pop    = sum(populations.values())
    agents_per_d = {d: max(1, round(n_agents * populations[d] / total_pop))
                    for d in districts}
    # Корректировка до точного числа
    diff = n_agents - sum(agents_per_d.values())
    agents_per_d[districts[0]] += diff

    records  = []
    agent_id = 0

    for district in districts:
        n = agents_per_d[district]
        d = dists[district]
        region_code = DISTRICT_TO_REGION_CODE.get(district, "XX")

        # Сэмплируем пары (возрастной бин, пол)
        age_sex_keys    = list(d["age_sex"].keys())
        age_sex_weights = [d["age_sex"][k] for k in age_sex_keys]
        if sum(age_sex_weights) == 0:
            age_sex_weights = [1] * len(age_sex_keys)
        age_sex_w = np.array(age_sex_weights, dtype=float)
        age_sex_w /= age_sex_w.sum()
        sampled_idx = rng.choice(len(age_sex_keys), size=n, p=age_sex_w)

        for idx in sampled_idx:
            bin_name, sex = age_sex_keys[idx]
            mid   = AGE_BIN_MIDPOINTS[bin_name]
            width = 1.0 if bin_name == "Zero years" else 5.0
            age   = float(np.clip(mid + rng.uniform(-width/2, width/2), 0, 105))

            # Демография
            edu_keys = list(d["education"].keys())
            edu_w    = np.array([d["education"][k] for k in edu_keys], dtype=float)
            education = edu_keys[rng.choice(len(edu_keys), p=edu_w/edu_w.sum())] if edu_w.sum() > 0 else "Unknown"

            occ_keys = list(d["occupations"].keys())
            occ_w    = np.array([d["occupations"][k] for k in occ_keys], dtype=float)
            occupation = occ_keys[rng.choice(len(occ_keys), p=occ_w/occ_w.sum())] if occ_w.sum() > 0 else "Unknown"

            nat_keys = list(d["nationalities"].keys())
            nat_w    = np.array([d["nationalities"][k] for k in nat_keys], dtype=float)
            nationality = nat_keys[rng.choice(len(nat_keys), p=nat_w/nat_w.sum())] if nat_w.sum() > 0 else "Slovak"

            marital_dist = d["marital"].get(sex, {})
            mar_keys = list(marital_dist.keys())
            mar_w    = np.array([marital_dist[k] for k in mar_keys], dtype=float)
            marital = mar_keys[rng.choice(len(mar_keys), p=mar_w/mar_w.sum())] if mar_w.sum() > 0 else "Single person"

            # Зарплата
            base_wage = d["total_wage"]
            if age < 18:
                wage = 0.0
            elif occupation in ("Elementary occupations",):
                wage = float(max(0, rng.normal(base_wage * 0.75, base_wage * 0.15)))
            elif education == "University":
                wage = float(max(0, rng.normal(base_wage * 1.35, base_wage * 0.25)))
            else:
                wage = float(max(0, rng.normal(base_wage, base_wage * 0.28)))

            # ── Параметры из SASD JSON ────────────────────────────────────────
            def sp(name): return sample_param(name, age, education, region_code, rng)

            perceived_control       = sp("perceived_control")
            econ_perceived_control  = sp("econ_perceived_control")
            inertia_social          = sp("inertia_social_component")
            info_quality            = sp("info_quality_modifier")

            # Домен: экономика
            d_econ_weight     = sp("domain_economic_weight")
            d_econ_gap        = sp("domain_economic_gap")
            d_econ_threshold  = sp("domain_economic_threshold")

            # Домен: социальный
            d_social_weight   = sp("domain_social_weight")

            # Домен: место-будущее
            d_future_value    = sp("domain_future_value")
            d_future_place    = sp("domain_future_place")

            # Домен: семья
            family_modifier   = sp("family_weight_modifier")

            # Мобильность
            commuter_threshold   = sp("commuter_mode_threshold")
            internal_mig_thr     = sp("internal_mig_threshold")
            external_mig_thr     = sp("external_mig_threshold")
            job_flexibility      = sp("job_flexibility_threshold")

            # Инерция (компоненты)
            tenure_loyalty       = sp("tenure_loyalty_bonus")
            shock_sensitivity    = sp("inertia_shock_sensitivity")

            # Инициализация
            satisfaction_base    = sp("satisfaction_init")

            # Сеть
            network_loc          = sp("network_location")   # Bernoulli
            network_job          = sp("network_job_search") # Bernoulli
            weak_ties            = sp("weak_ties_utility")

            # Цифровые
            digital_comm         = sp("digital_comm_intensity")
            net_signal_susc      = sp("network_signal_susceptibility")
            digital_trust_val    = sp("digital_trust")

            # ── Inertia ───────────────────────────────────────────────────────
            age_base = rogers_castro_mobility(age)
            # Инерция = обратная мобильности + социальный компонент
            # Высокий age_base мобильности → низкая инерция
            inertia_from_age = float(np.clip(1.0 - age_base, 0.1, 0.95))

            # Стаж (tenure) — инициализация
            tenure_mean = min(12 + age * 1.0, 180)
            tenure = int(np.clip(rng.exponential(tenure_mean), 0, 420))

            tenure_bonus = tenure_loyalty * math.log1p(tenure / 12) * 0.1

            # Итоговая инерция
            inertia = float(np.clip(
                inertia_from_age * 0.5 + inertia_social * 0.4 + tenure_bonus * 0.1,
                0.05, 0.95
            ))

            # Женатый — дополнительная инерция
            if marital in ("Married person", "married"):
                inertia = float(np.clip(inertia + 0.08, 0.05, 0.95))

            # ── Тип агента ────────────────────────────────────────────────────
            agent_type = _infer_agent_type(
                age, education, marital, perceived_control, inertia
            )

            # ── Веса доменов (нормировка) ─────────────────────────────────────
            # Базовые веса из SASD + модификаторы по типу агента
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

            # ── Начальное состояние удовлетворённости доменов ─────────────────
            # Используем satisfaction_baseline из SASD как стартовый value
            sat_noise = rng.normal(0, 0.06)
            sat_init  = float(np.clip(satisfaction_base + sat_noise, 0.05, 0.99))

            # Домен economic: начальный value — инвертированный gap
            econ_value = float(np.clip(1.0 - d_econ_gap + rng.normal(0, 0.05), 0.0, 1.0))

            # ── TPB pipeline ──────────────────────────────────────────────────
            # Состояние: none / forming / strong / move / commute / adapt
            intention_state  = "none"
            forming_ticks    = 0   # сколько тиков в фазе forming
            forming_duration = int(rng.integers(3, 9))  # 3-8 тиков до strong

            records.append({
                "id":            agent_id,
                "district":      district,
                "region":        region_code,
                # Демография
                "age":           round(age, 2),
                "sex":           sex,
                "education":     education,
                "occupation":    occupation,
                "nationality":   nationality,
                "marital":       marital,
                "wage":          round(wage, 2),
                "agent_type":    agent_type,
                # Инерция
                "inertia":       round(inertia, 4),
                "tenure":        tenure,
                "moved_ticks":   999,
                # TPB pipeline
                "intention_state":  intention_state,
                "forming_ticks":    forming_ticks,
                "forming_duration": forming_duration,
                # Домены — текущее состояние (value) и параметры
                "sat_economic":   round(econ_value, 4),
                "sat_social":     round(sat_init, 4),
                "sat_family":     round(sat_init, 4),
                "sat_place":      round(d_future_place, 4),
                "w_economic":     round(w_econ, 4),
                "w_social":       round(w_social, 4),
                "w_family":       round(w_family, 4),
                "w_future":       round(w_future, 4),
                "thr_economic":   round(d_econ_threshold, 4),
                "thr_social":     round(0.35, 4),  # фиксированный порог для социального
                "thr_family":     round(0.35, 4),
                "thr_place":      round(0.40, 4),
                # Психологические параметры
                "perceived_control":      round(perceived_control, 4),
                "econ_perceived_control": round(econ_perceived_control, 4),
                "inertia_social":         round(inertia_social, 4),
                "info_quality":           round(info_quality, 4),
                # Мобильность
                "commuter_threshold":   round(commuter_threshold, 4),
                "internal_mig_thr":     round(internal_mig_thr, 4),
                "external_mig_thr":     round(external_mig_thr, 4),
                "job_flexibility":      round(job_flexibility, 4),
                "tenure_loyalty":       round(tenure_loyalty, 4),
                "shock_sensitivity":    round(shock_sensitivity, 4),
                # Сеть
                "network_location":     float(network_loc),
                "network_job_search":   float(network_job),
                "weak_ties_utility":    round(weak_ties, 4),
                "network_signal":       "neutral",  # neutral / positive / shock
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
    """Краткая диагностика популяции агентов."""
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
    df = create_agents("environment.json", n_agents=70000)
    print("\nРаспределение по регионам:")
    print(df.groupby("region")["id"].count().sort_values(ascending=False).to_string())
