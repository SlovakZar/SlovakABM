"""
agents.py v1 — агенты с удовлетворённостью, персональным порогом и якорями.

Новые атрибуты:
  satisfaction    float [0,1]   текущая удовлетворённость (старт ~0.5, разброс)
  move_threshold  float         персональный порог недовольства для рассмотрения переезда
  tenure          int           месяцев в текущем районе (якорь места)
"""

import json, random
import numpy as np
import pandas as pd

TRNAVA_DISTRICTS = [
    "District of Trnava",
    "District of Dunajská\xa0Streda",
    "District of Galanta",
    "District of Hlohovec",
    "District of Piešťany",
    "District of Senica",
    "District of Skalica",
]

AGE_BIN_MIDPOINTS = {
    "Zero years": 0.5, "From 1 to 4 years": 2.5, "From 5 to 9 years": 7.0,
    "From 10 to 14 years": 12.0, "From 15 to 19 years": 17.0, "From 20 to 24 years": 22.0,
    "From 25 to 29 years": 27.0, "From 30 to 34 years": 32.0, "From 35 to 39 years": 37.0,
    "From 40 to 44 years": 42.0, "From 45 to 49 years": 47.0, "From 50 to 54 years": 52.0,
    "From 55 to 59 years": 57.0, "From 60 to 64 years": 62.0, "From 65 to 69 years": 67.0,
    "From 70 to 74 years": 72.0, "From 75 to 79 years": 77.0, "From 80 to 84 years": 82.0,
    "From 85 to 89 years": 87.0, "From 90 to 94 years": 92.0, "From 95 to 99 years": 97.0,
    "100 years or over": 102.0,
}


def _get_latest(data: dict, year: int = 2024):
    if not data: return None
    if year in data: return data[year]
    for y in sorted(data.keys(), reverse=True):
        if data[y] is not None: return data[y]
    return None


def _sample_from_dist(dist: dict) -> str:
    keys = [k for k, v in dist.items() if v and v > 0]
    weights = [dist[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0] if keys else "Unknown"


def _build_district_distributions(env: dict, year: int = 2024) -> dict:
    locations = env["locations"]
    regions   = env.get("regions", {})
    trnava_region = regions.get("Region of Trnava", {})

    # Семейный статус — уровень региона
    marital_regional = {}
    for status, sex_data in trnava_region.get("marital_status", {}).items():
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

        age_sex = {}
        for bin_name in AGE_BIN_MIDPOINTS:
            ag = data.get("age_groups", {}).get(bin_name, {})
            for sex in ("male", "female"):
                age_sex[(bin_name, sex)] = _get_latest(ag.get(sex, {}), year) or 0

        edu_dist = {k: _get_latest(v, year) or 0
                    for k, v in data.get("education", {}).items()
                    if _get_latest(v, year)}

        occ_dist = {k: _get_latest(v, year) or 0
                    for k, v in data.get("occupations", {}).items()
                    if k != "Total" and _get_latest(v, year)}

        total_wage = _get_latest(data.get("wages", {}).get("Total", {}), year) or 1400

        nat_dist = {}
        for nat, sex_data in data.get("nationalities", {}).items():
            total = sum(_get_latest(sex_data.get(s, {}), year) or 0
                        for s in ("male", "female"))
            if total > 0:
                nat_dist[nat] = total

        dists[district] = {
            "age_sex":    age_sex,
            "education":  edu_dist,
            "occupations": occ_dist,
            "total_wage": total_wage,
            "marital":    marital_regional,
            "nationalities": nat_dist,
            "population": sum(age_sex.values()),
        }

    return dists


def create_agents(
    env_path: str = "environment.json",
    n_agents: int = 5000,
    year: int = 2024,
    seed: int = 42,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    with open(env_path) as f:
        env = json.load(f)

    dists = _build_district_distributions(env, year)

    # Пропорциональное распределение по районам
    populations = {d: dists[d]["population"] for d in TRNAVA_DISTRICTS}
    total_pop = sum(populations.values())
    agents_per_district = {d: max(1, round(n_agents * populations[d] / total_pop))
                           for d in TRNAVA_DISTRICTS}
    diff = n_agents - sum(agents_per_district.values())
    agents_per_district[TRNAVA_DISTRICTS[0]] += diff

    records = []
    agent_id = 0

    for district in TRNAVA_DISTRICTS:
        n  = agents_per_district[district]
        d  = dists[district]

        age_sex_keys    = list(d["age_sex"].keys())
        age_sex_weights = [d["age_sex"][k] for k in age_sex_keys]
        if sum(age_sex_weights) == 0:
            age_sex_weights = [1] * len(age_sex_keys)

        sampled_age_sex = random.choices(age_sex_keys, weights=age_sex_weights, k=n)

        for bin_name, sex in sampled_age_sex:
            mid   = AGE_BIN_MIDPOINTS[bin_name]
            width = 1.0 if bin_name == "Zero years" else 5.0
            age   = max(0.0, mid + random.uniform(-width / 2, width / 2))

            education   = _sample_from_dist(d["education"])
            occupation  = _sample_from_dist(d["occupations"])
            nationality = _sample_from_dist(d["nationalities"])

            base_wage = d["total_wage"]
            if age < 18:
                wage = 0.0
            elif occupation in ("Elementary occupations",):
                wage = max(0.0, np.random.normal(base_wage * 0.7, base_wage * 0.15))
            else:
                wage = max(0.0, np.random.normal(base_wage, base_wage * 0.3))

            marital = _sample_from_dist(d["marital"].get(sex, {})) if d["marital"] else "Single person"

            # ── v1: новые атрибуты ────────────────────────────────────────────

            # Персональный порог мобильности:
            # ~15% агентов очень мобильны (threshold=0.05), ~15% очень инертны (0.35)
            # Образование повышает мобильность
            edu_mobility_bonus = {"University": -0.04, "Basic": 0.05,
                                  "Without education": 0.08}.get(education, 0.0)
            move_threshold = float(np.clip(
                np.random.normal(0.20, 0.07) + edu_mobility_bonus, 0.04, 0.45
            ))

            # Начальная удовлетворённость: случайная ~0.55 (большинство "терпимо довольны")
            # Немного выше в богатых районах
            wage_quartile = (d["total_wage"] - 1424) / (1761 - 1424)  # нормализация 0..1
            sat_init = float(np.clip(
                np.random.normal(0.52 + 0.08 * wage_quartile, 0.12), 0.1, 0.95
            ))

            # Стаж в районе (tenure): случайный 0-120 месяцев (10 лет)
            # Старшие агенты в среднем живут дольше на месте
            tenure_mean = min(12 + age * 0.8, 120)
            tenure = int(np.clip(np.random.exponential(tenure_mean), 0, 360))

            records.append({
                "id":            agent_id,
                "district":      district,
                "age":           round(age, 2),
                "sex":           sex,
                "education":     education,
                "occupation":    occupation,
                "wage":          round(wage, 2),
                "marital":       marital,
                "nationality":   nationality,
                # v1
                "satisfaction":  round(sat_init, 4),
                "move_threshold": round(move_threshold, 4),
                "tenure":        tenure,
                "moved_ticks":   999,
            })
            agent_id += 1

    df = pd.DataFrame(records)
    print(f"  Создано агентов: {len(df):,}  |  Районов: {df['district'].nunique()}")
    print(f"  Ср. move_threshold: {df['move_threshold'].mean():.3f}  |"
          f"  Ср. satisfaction: {df['satisfaction'].mean():.3f}")
    return df


if __name__ == "__main__":
    df = create_agents("environment.json", n_agents=5000)
    print("\nРаспределение move_threshold:")
    print(pd.cut(df["move_threshold"],
                 bins=[0, 0.1, 0.2, 0.3, 0.45],
                 labels=["очень моб.", "умеренно", "инертный", "очень инертный"]
                 ).value_counts().sort_index().to_string())
