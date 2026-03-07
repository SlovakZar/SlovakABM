"""
agents.py — создание популяции агентов путём сэмплирования из environment.json.

Каждый агент — строка pandas DataFrame со следующими атрибутами:
  id            int       уникальный идентификатор
  district      str       текущий район (один из 7 в Трнавском крае)
  age           float     возраст в годах (точный, не бин)
  sex           str       'male' / 'female'
  education     str       уровень образования
  occupation    str       тип занятости
  wage          float     индивидуальная зарплата (EUR/мес, с шумом)
  marital       str       семейный статус
  nationality   str       национальность
  moved_ticks   int       сколько тиков назад переехал (инерция)
"""

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path


TRNAVA_DISTRICTS = [
    "District of Trnava",
    "District of Dunajská\xa0Streda",
    "District of Galanta",
    "District of Hlohovec",
    "District of Piešťany",
    "District of Senica",
    "District of Skalica",
]

# Полные имена возрастных бинов и их середины
AGE_BIN_MIDPOINTS = {
    "Zero years": 0.5,
    "From 1 to 4 years": 2.5,
    "From 5 to 9 years": 7.0,
    "From 10 to 14 years": 12.0,
    "From 15 to 19 years": 17.0,
    "From 20 to 24 years": 22.0,
    "From 25 to 29 years": 27.0,
    "From 30 to 34 years": 32.0,
    "From 35 to 39 years": 37.0,
    "From 40 to 44 years": 42.0,
    "From 45 to 49 years": 47.0,
    "From 50 to 54 years": 52.0,
    "From 55 to 59 years": 57.0,
    "From 60 to 64 years": 62.0,
    "From 65 to 69 years": 67.0,
    "From 70 to 74 years": 72.0,
    "From 75 to 79 years": 77.0,
    "From 80 to 84 years": 82.0,
    "From 85 to 89 years": 87.0,
    "From 90 to 94 years": 92.0,
    "From 95 to 99 years": 97.0,
    "100 years or over": 102.0,
}


def _get_latest(data: dict, year: int = 2024):
    if not data:
        return None
    if year in data:
        return data[year]
    for y in sorted(data.keys(), reverse=True):
        if data[y] is not None:
            return data[y]
    return None


def _sample_from_dist(dist: dict) -> str:
    """Сэмплируем ключ из словаря {категория: вес}. Нули и None игнорируем."""
    keys = [k for k, v in dist.items() if v and v > 0]
    weights = [dist[k] for k in keys]
    if not keys:
        return "Unknown"
    return random.choices(keys, weights=weights, k=1)[0]


def _build_district_distributions(env: dict, year: int = 2024) -> dict:
    """
    Для каждого района Трнавского края извлекаем распределения
    для сэмплирования атрибутов агента.
    """
    locations = env["locations"]
    regions = env.get("regions", {})
    trnava_region = regions.get("Region of Trnava", {})

    # Семейный статус — только на уровне региона
    marital_regional = {}
    ms = trnava_region.get("marital_status", {})
    for status, sex_data in ms.items():
        if status == "Unknown":
            continue
        for sex in ("male", "female"):
            v = _get_latest(sex_data.get(sex, {}), year) or 0
            marital_regional.setdefault(sex, {})[status] = (
                marital_regional.get(sex, {}).get(status, 0) + v
            )

    dists = {}
    for district in TRNAVA_DISTRICTS:
        data = locations.get(district, {})

        # ── возраст × пол ────────────────────────────────────────────────────
        age_sex = {}  # { (bin_name, sex): count }
        for bin_name, midpoint in AGE_BIN_MIDPOINTS.items():
            ag = data.get("age_groups", {}).get(bin_name, {})
            for sex in ("male", "female"):
                v = _get_latest(ag.get(sex, {}), year) or 0
                age_sex[(bin_name, sex)] = v

        # ── образование ──────────────────────────────────────────────────────
        edu_dist = {}
        for k, v in data.get("education", {}).items():
            val = _get_latest(v, year) or 0
            if val > 0:
                edu_dist[k] = val

        # ── занятость (профессии) ─────────────────────────────────────────────
        occ_dist = {}
        for k, v in data.get("occupations", {}).items():
            if k == "Total":
                continue
            val = _get_latest(v, year) or 0
            if val > 0:
                occ_dist[k] = val

        # ── зарплата: берём средние по секторам, строим распределение ────────
        wages_by_sector = {}
        for sector, v in data.get("wages", {}).items():
            if sector == "Total":
                continue
            val = _get_latest(v, year)
            if val:
                wages_by_sector[sector] = val
        # Total wage как базовое значение для шума
        total_wage = _get_latest(data.get("wages", {}).get("Total", {}), year) or 1000

        # ── национальности ────────────────────────────────────────────────────
        nat_dist = {}
        for nat, sex_data in data.get("nationalities", {}).items():
            total = 0
            for sex in ("male", "female"):
                total += _get_latest(sex_data.get(sex, {}), year) or 0
            if total > 0:
                nat_dist[nat] = total

        dists[district] = {
            "age_sex": age_sex,
            "education": edu_dist,
            "occupations": occ_dist,
            "total_wage": total_wage,
            "wages_by_sector": wages_by_sector,
            "marital": marital_regional,
            "nationalities": nat_dist,
            # население для пропорции агентов между районами
            "population": sum(age_sex.values()),
        }

    return dists


def create_agents(
    env_path: str = "environment.json",
    n_agents: int = 5000,
    year: int = 2024,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Создаёт DataFrame с n_agents агентами,
    сэмплированными пропорционально населению районов.
    """
    random.seed(seed)
    np.random.seed(seed)

    with open(env_path) as f:
        env = json.load(f)

    dists = _build_district_distributions(env, year)

    # Распределяем агентов по районам пропорционально населению
    populations = {d: dists[d]["population"] for d in TRNAVA_DISTRICTS}
    total_pop = sum(populations.values())
    agents_per_district = {
        d: max(1, round(n_agents * populations[d] / total_pop))
        for d in TRNAVA_DISTRICTS
    }
    # Корректируем до точного n_agents
    diff = n_agents - sum(agents_per_district.values())
    first_district = TRNAVA_DISTRICTS[0]
    agents_per_district[first_district] += diff

    records = []
    agent_id = 0

    for district in TRNAVA_DISTRICTS:
        n = agents_per_district[district]
        d = dists[district]

        # --- строим плоский список (bin_name, sex) с весами для age/sex ---
        age_sex_keys = list(d["age_sex"].keys())
        age_sex_weights = [d["age_sex"][k] for k in age_sex_keys]

        if sum(age_sex_weights) == 0:
            age_sex_weights = [1] * len(age_sex_keys)

        # Сэмплируем n агентов для этого района
        sampled_age_sex = random.choices(age_sex_keys, weights=age_sex_weights, k=n)

        for bin_name, sex in sampled_age_sex:
            # Точный возраст: середина бина ± равномерный шум внутри бина
            mid = AGE_BIN_MIDPOINTS[bin_name]
            # Размер бина
            if bin_name == "Zero years":
                width = 1.0
            elif bin_name == "100 years or over":
                width = 5.0
            else:
                width = 5.0
            age = mid + random.uniform(-width / 2, width / 2)
            age = max(0.0, age)

            education = _sample_from_dist(d["education"])
            occupation = _sample_from_dist(d["occupations"])
            nationality = _sample_from_dist(d["nationalities"])

            # Зарплата: base ± 30% нормальный шум; у безработных/детей — 0
            base_wage = d["total_wage"]
            if age < 18 or occupation in ("Elementary occupations",):
                wage = max(0.0, np.random.normal(base_wage * 0.6, base_wage * 0.15))
            else:
                wage = max(0.0, np.random.normal(base_wage, base_wage * 0.3))

            # Семейный статус с учётом пола
            marital_dist = d["marital"].get(sex, {})
            marital = _sample_from_dist(marital_dist) if marital_dist else "Single person"

            records.append({
                "id": agent_id,
                "district": district,
                "age": round(age, 2),
                "sex": sex,
                "education": education,
                "occupation": occupation,
                "wage": round(wage, 2),
                "marital": marital,
                "nationality": nationality,
                "moved_ticks": 999,  # давно не переезжал (инерция снята)
            })
            agent_id += 1

    df = pd.DataFrame(records)
    print(f"  Создано агентов: {len(df):,}  |  Районов: {df['district'].nunique()}")
    return df


if __name__ == "__main__":
    df = create_agents("environment.json", n_agents=5000)
    print("\nРаспределение по районам:")
    print(df.groupby("district")["id"].count().to_string())
    print("\nВозрастные бины:")
    df["age_bin"] = pd.cut(df["age"], bins=[0,18,30,45,60,75,150],
                           labels=["0-17","18-29","30-44","45-59","60-74","75+"])
    print(df["age_bin"].value_counts().sort_index().to_string())
