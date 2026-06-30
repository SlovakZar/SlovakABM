# build_environment_colab_fixed.py
import json
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

# ---- Определение среды Colab ----
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def mount_drive_if_colab():
    if is_colab():
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive смонтирован в /content/drive")
    else:
        print("Запуск не в Colab, монтирование Drive пропущено.")

# ---- Вспомогательные функции (без изменений) ----
def _to_float(val) -> float | None:
    if pd.isna(val):
        return None
    s = str(val).replace("\xa0", "").replace("\u202f", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

EDU_MAP = {
    "With no school education (aged 15+)": "low",
    "Elementary - 1st level of primary school": "low",
    "Elementary - 2nd level of primary school": "low",
    "Primary education (not specified)": "low",
    "Secondary technical (vocational) education with no graduation - with no certificate of apprenticeship (on-the-job training, pre-employment training)": "low",
    "Secondary technical (vocational) education with no graduation - with a certificate of apprenticeship": "low",
    "Secondary technical (vocational) education with no graduation - with a final examination certificate": "low",
    "Secondary technical (vocational) education with no graduation (not specified)": "low",
    "Complete secondary education with graduation (not specified)": "medium",
    "Complete secondary education with graduation - general": "medium",
    "Complete secondary education with graduation - technical": "medium",
    "Complete secondary education with graduation - technical (vocational) with a certificate of apprenticeship": "medium",
    "Higher technical education (not specified)": "medium",
    "Higher technical education - higher technical (school-leaving examination, graduation diploma)": "medium",
    "Higher technical education - post-secondary (post-graduation qualifying)": "medium",
    "Higher technical education - post-secondary (school-leaving examination in the fields of study for secondary technical school graduates)": "medium",
    "Higher education (not specified)": "high",
    "Higher education - 1st level (Bc.)": "high",
    "Higher education - 2nd level (Ing.; Mgr.; MUDr.; etc.)": "high",
    "Higher education - 3rd level (PhD.; etc.)": "high",
    "Confidential": None,
    "Not found out": None,
}

AGE_BIN_META = {
    "15 - 19 years": (17.0, 5),
    "20 - 24 years": (22.0, 5),
    "25 - 29 years": (27.0, 5),
    "30 - 34 years": (32.0, 5),
    "35 - 39 years": (37.0, 5),
    "40 - 44 years": (42.0, 5),
    "45 - 49 years": (47.0, 5),
    "50 - 54 years": (52.0, 5),
    "55 - 59 years": (57.0, 5),
    "60 - 64 years": (62.0, 5),
    "65 - 69 years": (67.0, 5),
    "70 - 74 years": (72.0, 5),
    "75 - 79 years": (77.0, 5),
    "80 - 84 years": (82.0, 5),
    "85 - 89 years": (87.0, 5),
    "90 and more years": (93.0, 10),
}

INDUSTRY_TO_SALARY_COL = {
    "Agriculture, forestry and fishing": "salary_agriculture",
    "Manufacturing total": "salary_manufacturing",
    "Water supply; sewerage, waste management and remediation activities": "salary_water",
    "Construction": "salary_construction",
    "Wholesale and retail trade; repair of motor vehicles and motorcycles": "salary_trade",
    "Transportation and storage": "salary_transport",
    "Accommodation and food service activities": "salary_accommodation",
    "Information and communication": "salary_ict",
    "Professional, scientific and technical activities": "salary_professional",
    "Administrative and support service activities": "salary_admin",
    "Public administration and defence": "salary_public",
    "Human health and social work activities": "salary_health",
    "Other": "avg_wage",
}

def _pivot_distribution(df_var: pd.DataFrame, sex_col_needed: bool = True) -> dict:
    result = {}
    for (district, sex), grp in df_var.groupby(["district", "sex"]):
        cats = {row["category"]: int(row["count"])
                for _, row in grp.iterrows()
                if pd.notna(row["count"]) and row["count"] > 0}
        if not cats:
            continue
        if sex_col_needed:
            result.setdefault(district, {})[sex] = cats
        else:
            result[district] = cats
    return result

def _normalize(d: dict) -> dict:
    total = sum(d.values())
    if total == 0:
        return {}
    return {k: round(v / total, 6) for k, v in d.items() if v > 0}

def _collapse_education(edu_counts: dict) -> dict:
    out = {"low": 0, "medium": 0, "high": 0}
    for cat, cnt in edu_counts.items():
        grp = EDU_MAP.get(cat)
        if grp:
            out[grp] += cnt
    return {k: v for k, v in out.items() if v > 0}

def build(
    master_path: str = "districts_master.csv",
    dist_path: str = "agent_distributions_with_industry.csv",
    env_out: str = "data/environment.json",
    dist_out: str = "data/agent_init_distributions.json",
):
    print("Загрузка данных...")
    dm = pd.read_csv(master_path, sep=";")
    dd = pd.read_csv(dist_path)

    dd.columns = [c.lstrip("\ufeff") for c in dd.columns]

    print(f"  districts_master: {len(dm)} районов, {len(dm.columns)} столбцов")
    print(f"  agent_distributions: {len(dd)} строк, {dd['variable'].nunique()} переменных")

    var_dfs = {v: dd[dd["variable"] == v].copy() for v in dd["variable"].unique()}

    age_dist    = _pivot_distribution(var_dfs["age_group"], sex_col_needed=True)
    marital_dist = _pivot_distribution(var_dfs["marital_status"], sex_col_needed=True)
    nat_dist    = _pivot_distribution(var_dfs["nationality"], sex_col_needed=True)
    edu_dist    = _pivot_distribution(var_dfs["education"], sex_col_needed=False)
    occ_dist    = _pivot_distribution(var_dfs["occupation"], sex_col_needed=False)
    econ_dist   = _pivot_distribution(var_dfs["economic_activity"], sex_col_needed=False)
    ind_dist    = _pivot_distribution(var_dfs["industry"], sex_col_needed=False)

    # environment.json
    print("\nСтроим environment.json...")
    locations = {}
    for _, row in dm.iterrows():
        district = row["district"]
        region   = row["region"]

        salary_by_industry = {}
        for ind_label, col in INDUSTRY_TO_SALARY_COL.items():
            if col in dm.columns:
                val = row.get(col)
                if pd.notna(val) and val > 0:
                    salary_by_industry[ind_label] = float(val)

        housing = {
            "price_m2":             _to_float(row["housing_price_m2"]),
            "apartment_price_eur":  _to_float(row["apartment_price_eur"]),
            "house_price_eur":      _to_float(row["house_price_eur"]),
            "owner_share":          _to_float(row["owner_share"]),
            "total_dwellings":      int(row["total_dwellings"]) if pd.notna(row["total_dwellings"]) else None,
            "vacant_dwellings":     int(row["vacant_dwellings"]) if pd.notna(row["vacant_dwellings"]) else None,
        }

        infrastructure = {
            "polyclinics":     (int(_to_float(row["polyclinics_count"]) or 0)),
            "hospitals":       (int(_to_float(row["hospitals_count"]) or 0)),
            "cinemas":         (int(_to_float(row["cinemas_count"]) or 0)),
            "museums":         (int(_to_float(row["museums_count"]) or 0)),
            "galleries":       (int(_to_float(row["galleries_count"]) or 0)),
        }

        business = {
            "total_companies":   (int(_to_float(row["total_companies"]) or 0)),
            "foreign_companies": int(str(row["foreign_companies"]).replace("\xa0", "").replace(" ", "")) if pd.notna(row["foreign_companies"]) else 0,
            "small_companies":   int(str(row["small_companies"]).replace("\xa0", "").replace(" ", "")) if pd.notna(row["small_companies"]) else 0,
            "medium_companies":  int(str(row["medium_companies"]).replace("\xa0", "").replace(" ", "")) if pd.notna(row["medium_companies"]) else 0,
            "large_companies":   int(str(row["large_companies"]).replace("\xa0", "").replace(" ", "")) if pd.notna(row["large_companies"]) else 0,
        }

        locations[district] = {
            "type":              "district",
            "region":            region,
            "population":        int(row["population"]),
            "unemployment_rate": float(row["unemployment_rate"]) if pd.notna(row["unemployment_rate"]) else None,
            "avg_wage":          float(row["avg_wage"]) if pd.notna(row["avg_wage"]) else None,
            "salary_by_industry": salary_by_industry,
            "housing":           housing,
            "infrastructure":    infrastructure,
            "business":          business,
        }

    env_data = {
        "meta": {
            "source": "SODB 2021 / NBS / ŠÚ SR",
            "districts_count": len(locations),
            "fields": ["population", "unemployment_rate", "avg_wage",
                       "salary_by_industry", "housing", "infrastructure", "business"],
        },
        "locations": locations,
    }

    with open(env_out, "w", encoding="utf-8") as f:
        json.dump(env_data, f, ensure_ascii=False, indent=2)
    size_kb = Path(env_out).stat().st_size / 1024
    print(f"  ✓ {env_out} — {len(locations)} районов, {size_kb:.0f} KB")

    # agent_init_distributions.json
    print("\nСтроим agent_init_distributions.json...")
    agent_dists = {}

    for district in dm["district"]:
        row = dm[dm["district"] == district].iloc[0]
        region = row["region"]

        # Возраст × пол
        age_sex = {}
        for sex, age_cats in age_dist.get(district, {}).items():
            for cat, cnt in age_cats.items():
                if cat in AGE_BIN_META and cnt > 0:
                    age_sex[f"{cat}|{sex}"] = {
                        "count": cnt,
                        "midpoint": AGE_BIN_META[cat][0],
                        "width": AGE_BIN_META[cat][1],
                        "sex": sex,
                    }

        # Образование
        edu_raw = edu_dist.get(district, {})
        edu_grouped = _collapse_education(edu_raw)
        edu_shares = _normalize(edu_grouped)

        # Семейный статус
        marital = {}
        for sex, cats in marital_dist.get(district, {}).items():
            clean = {k: v for k, v in cats.items()
                     if k not in ("Not found out", "Confidential")}
            if clean:
                marital[sex] = _normalize(clean)

        # Национальность
        nat_agg = {"Slovak": 0, "Hungarian": 0, "Roma": 0, "Other": 0}
        for sex, cats in nat_dist.get(district, {}).items():
            for cat, cnt in cats.items():
                if "Slovak" in cat or cat == "Slovak Republic":
                    nat_agg["Slovak"] += cnt
                elif "Hungary" in cat or "Magyar" in cat:
                    nat_agg["Hungarian"] += cnt
                elif "Roma" in cat:
                    nat_agg["Roma"] += cnt
                elif cat not in ("Confidential", "Not found out"):
                    nat_agg["Other"] += cnt
        nat_shares = _normalize({k: v for k, v in nat_agg.items() if v > 0})
        if not nat_shares:
            nat_shares = {"Slovak": 1.0}

        # Занятость
        econ_raw = econ_dist.get(district, {})
        employed_cnt = sum(v for k, v in econ_raw.items()
                           if "Working" in k and "Confidential" not in k)
        unemployed_cnt = int(row["population"] * row["unemployment_rate"]) \
            if pd.notna(row["unemployment_rate"]) else 0
        inactive_cnt = max(0, int(row["population"]) - employed_cnt - unemployed_cnt)

        employment = {
            "employed_share":   round(employed_cnt / max(row["population"], 1), 4),
            "unemployed_share": float(row["unemployment_rate"]) if pd.notna(row["unemployment_rate"]) else 0.0,
            "employment_detail": {k: v for k, v in econ_raw.items()
                                  if k not in ("Confidential",)},
        }

        # Отрасль занятости
        ind_raw = ind_dist.get(district, {})
        workers_counts = {k: v for k, v in ind_raw.items() if k not in ("Confidential",)}
        ind_shares = _normalize(workers_counts)

        # Зарплата по отраслям
        salary_lookup = {}
        for ind_label, col in INDUSTRY_TO_SALARY_COL.items():
            if col in dm.columns:
                val = row.get(col)
                if pd.notna(val) and val > 0:
                    salary_lookup[ind_label] = float(val)

        owner_share = float(row["owner_share"]) if pd.notna(row["owner_share"]) else 0.65

        agent_dists[district] = {
            "region":          region,
            "population":      int(row["population"]),
            "age_sex":         age_sex,
            "education":       edu_shares,
            "marital":         marital,
            "nationality":     nat_shares,
            "employment":      employment,
            "industry":        ind_shares,
            "salary_by_industry": salary_lookup,
            "avg_wage":        float(row["avg_wage"]) if pd.notna(row["avg_wage"]) else 1400.0,
            "owner_share":     owner_share,
            "housing_price_m2": (_to_float(row["housing_price_m2"]) or 1500.0),
        }

    dist_data = {
        "meta": {
            "source": "SODB 2021",
            "districts_count": len(agent_dists),
            "note": "Используется agents.py для инициализации агентов. "
                    "Психологические параметры в agent_params_from_survey.json.",
        },
        "districts": agent_dists,
    }

    with open(dist_out, "w", encoding="utf-8") as f:
        json.dump(dist_data, f, ensure_ascii=False, indent=2)
    size_kb = Path(dist_out).stat().st_size / 1024
    print(f"  ✓ {dist_out} — {len(agent_dists)} районов, {size_kb:.0f} KB")

    # Диагностика
    print("\n── Диагностика ──")
    sample_d = "District of Bratislava I"
    if sample_d in agent_dists:
        s = agent_dists[sample_d]
        print(f"  {sample_d}:")
        print(f"    population={s['population']:,}  unemployment={s['employment']['unemployed_share']:.1%}")
        print(f"    avg_wage={s['avg_wage']:.0f}€  housing={s['housing_price_m2']:.0f}€/m²")
        print(f"    owner_share={s['owner_share']:.1%}")
        print(f"    industries: {list(s['industry'].keys())[:4]}")
        print(f"    edu: {s['education']}")

    sample_d2 = "District of Rimavská Sobota"
    if sample_d2 in agent_dists:
        s2 = agent_dists[sample_d2]
        print(f"  {sample_d2}:")
        print(f"    unemployment={s2['employment']['unemployed_share']:.1%}  avg_wage={s2['avg_wage']:.0f}€")

    print("\n✓ Готово.")
    return env_data, dist_data

# ---- CLI с поддержкой Colab и parse_known_args ----
if __name__ == "__main__":
    mount_drive_if_colab()

    parser = argparse.ArgumentParser(description="Build environment + agent init distributions (Colab-ready)")
    parser.add_argument("--data_dir", default="/content/drive/MyDrive/",
                        help="Корневая папка с CSV-файлами (например, /content/drive/MyDrive/agent_data)")
    parser.add_argument("--master", default="districts_master.csv",
                        help="Имя файла districts_master.csv (относительно data_dir)")
    parser.add_argument("--distributions", default="agent_distributions_with_industry.csv",
                        help="Имя файла agent_distributions_with_industry.csv")
    parser.add_argument("--env_out", default="data/environment.json",
                        help="Выходной файл environment.json")
    parser.add_argument("--dist_out", default="data/agent_init_distributions.json",
                        help="Выходной файл agent_init_distributions.json")

    # Разрешаем неизвестные аргументы (например, -f из Colab)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"⚠️ Предупреждение: игнорирую неизвестные аргументы: {unknown}")

    master_path = os.path.join(args.data_dir, args.master)
    dist_path = os.path.join(args.data_dir, args.distributions)

    build(
        master_path=master_path,
        dist_path=dist_path,
        env_out=args.env_out,
        dist_out=args.dist_out,
    )
    from google.colab import files
    files.download('environment.json')
    files.download('agent_init_distributions.json')
