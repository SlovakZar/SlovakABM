"""
initializer.py — собирает все источники в единый environment.json.

Структура на выходе:
{
  "locations": {
    "District of Bratislava I": {
      "type": "district",
      "region": "Region of Bratislava",
      "population": {...},
      "wages": {...},
      "occupations": {...},
      "age_groups": {...},
      "education": {...},
      "marital_status": {...},
      "nationalities": {...},
      "housing": {...},    # downscale с региона
      "labour": {...}      # downscale с региона
    },
    ...
  },
  "regions": { ... },   # агрегаты уровня региона
  "cities": { ... },    # Urban Audit данные по городам
  "meta": { "year": 2024, "source": "Datacube SK" }
}
"""

import json
from pathlib import Path
from loaders import (
    load_marital_by_region, load_marital_by_age,
    load_age_groups, load_wages, load_occupations,
    load_nationalities, load_education,
    load_students, load_city_population, load_kraje_population,
    load_labour, load_housing,
)

# ── маппинг: как называются регионы в разных файлах ──────────────────────────

# Какой district принадлежит какому региону
DISTRICT_TO_REGION = {
    "District of Bratislava I":   "Region of Bratislava",
    "District of Bratislava II":  "Region of Bratislava",
    "District of Bratislava III": "Region of Bratislava",
    "District of Bratislava IV":  "Region of Bratislava",
    "District of Bratislava V":   "Region of Bratislava",
    "District of Malacky":        "Region of Bratislava",
    "District of Pezinok":        "Region of Bratislava",
    "District of Senec":          "Region of Bratislava",

    "District of Trnava":         "Region of Trnava",
    "District of Dunajská Streda":"Region of Trnava",
    "District of Galanta":        "Region of Trnava",
    "District of Hlohovec":       "Region of Trnava",
    "District of Piešťany":       "Region of Trnava",
    "District of Senica":         "Region of Trnava",
    "District of Skalica":        "Region of Trnava",

    "District of Trenčín":        "Region of Trenčín",
    "District of Bánovce nad Bebravou": "Region of Trenčín",
    "District of Ilava":          "Region of Trenčín",
    "District of Myjava":         "Region of Trenčín",
    "District of Nové Mesto nad Váhom": "Region of Trenčín",
    "District of Partizánske":    "Region of Trenčín",
    "District of Považská Bystrica": "Region of Trenčín",
    "District of Púchov":         "Region of Trenčín",

    "District of Nitra":          "Region of Nitra",
    "District of Komárno":        "Region of Nitra",
    "District of Levice":         "Region of Nitra",
    "District of Nové Zámky":     "Region of Nitra",
    "District of Šaľa":           "Region of Nitra",
    "District of Topoľčany":      "Region of Nitra",
    "District of Zlaté Moravce":  "Region of Nitra",

    "District of Žilina":         "Region of Žilina",
    "District of Bytča":          "Region of Žilina",
    "District of Čadca":          "Region of Žilina",
    "District of Dolný Kubín":    "Region of Žilina",
    "District of Kysucké Nové Mesto": "Region of Žilina",
    "District of Liptovský Mikuláš": "Region of Žilina",
    "District of Martin":         "Region of Žilina",
    "District of Námestovo":      "Region of Žilina",
    "District of Ružomberok":     "Region of Žilina",
    "District of Turčianske Teplice": "Region of Žilina",
    "District of Tvrdošín":       "Region of Žilina",

    "District of Banská Bystrica":"Region of Banská Bystrica",
    "District of Banská Štiavnica": "Region of Banská Bystrica",
    "District of Brezno":         "Region of Banská Bystrica",
    "District of Detva":          "Region of Banská Bystrica",
    "District of Krupina":        "Region of Banská Bystrica",
    "District of Lučenec":        "Region of Banská Bystrica",
    "District of Poltár":         "Region of Banská Bystrica",
    "District of Revúca":         "Region of Banská Bystrica",
    "District of Rimavská Sobota":"Region of Banská Bystrica",
    "District of Veľký Krtíš":    "Region of Banská Bystrica",
    "District of Zvolen":         "Region of Banská Bystrica",
    "District of Žiar nad Hronom":"Region of Banská Bystrica",
    "District of Žarnovica":      "Region of Banská Bystrica",

    "District of Prešov":         "Region of Prešov",
    "District of Bardejov":       "Region of Prešov",
    "District of Humenné":        "Region of Prešov",
    "District of Kežmarok":       "Region of Prešov",
    "District of Levoča":         "Region of Prešov",
    "District of Medzilaborce":   "Region of Prešov",
    "District of Michalovce":     "Region of Prešov",
    "District of Poprad":         "Region of Prešov",
    "District of Sabinov":        "Region of Prešov",
    "District of Snina":          "Region of Prešov",
    "District of Stará Ľubovňa":  "Region of Prešov",
    "District of Stropkov":       "Region of Prešov",
    "District of Vranov nad Topľou": "Region of Prešov",

    "District of Košice I":       "Region of Košice",
    "District of Košice II":      "Region of Košice",
    "District of Košice III":     "Region of Košice",
    "District of Košice IV":      "Region of Košice",
    "District of Košice-okolie":  "Region of Košice",
    "District of Gelnica":        "Region of Košice",
    "District of Rožňava":        "Region of Košice",
    "District of Sobrance":       "Region of Košice",
    "District of Spišská Nová Ves": "Region of Košice",
    "District of Trebišov":       "Region of Košice",
    # Districts которые были Unknown из-за отсутствия в маппинге
    "District of Prievidza":       "Region of Trenčín",
    "District of Dunajská Streda":  "Region of Trnava",
    "District of Spišská Nová Ves": "Region of Prešov",
    "District of Śaľa":           "Region of Nitra",
    "District of Svidník":        "Region of Prešov",
    "District of Košice - okolie":"Region of Košice",
}

# Нормализация названий регионов между файлами
REGION_ALIASES = {
    "Region of Bratislava (NUTS2)": "Region of Bratislava",
    "Region of Bratislava (NUTS 2)": "Region of Bratislava",
    "West Slovakia": None,       # агрегат — пропускаем
    "Central Slovakia": None,
    "East Slovakia": None,
    "Slovak Republic": None,
    "Country": None,
    "City": None,
}

# Города-агрегаты: Bratislava и Košice состоят из districts, отдельный тип
CITY_AGGREGATES = {"Bratislava", "Košice"}

# Маппинг нестандартных названий в housing/labour файлах → Region
HOUSING_REGION_MAP = {
    "Region of Bratislava":                    "Region of Bratislava",
    "Košice City + District of Košice-okolie": "Region of Košice",
    "District of Banská Bystrica":             "Region of Banská Bystrica",
    "District of Nitra":                       "Region of Nitra",
    "District of Prešov":                      "Region of Prešov",
    "District of Žilina":                      "Region of Žilina",
    "District of Trnava":                      "Region of Trnava",
    "District of Trenčín":                     "Region of Trenčín",
}

# Labour файл использует те же нестандартные имена что и housing
LABOUR_REGION_MAP = {
    "Region of Bratislava":                    "Region of Bratislava",
    "Košice City + District of Košice-okolie": "Region of Košice",
    "District of Banská Bystrica":             "Region of Banská Bystrica",
    "District of Nitra":                       "Region of Nitra",
    "District of Prešov":                      "Region of Prešov",
    "District of Žilina":                      "Region of Žilina",
    "District of Trnava":                      "Region of Trnava",
    "District of Trenčín":                     "Region of Trenčín",
}

# Города как отдельные сущности (Urban Audit)
KNOWN_CITIES = {"Bratislava", "Košice"}

# Маппинг: город → его регион
CITY_TO_REGION = {
    "Bratislava": "Region of Bratislava",
    "Košice":     "Region of Košice",
}


def _is_district(name: str) -> bool:
    return name.startswith("District of")

def _is_region(name: str) -> bool:
    return name.startswith("Region of")

def _normalize_region(name: str) -> str | None:
    return REGION_ALIASES.get(name, name)

def _get_latest(data: dict, year: int = 2024):
    """Берём значение за конкретный год, fallback на ближайший."""
    if not data:
        return None
    if year in data:
        return data[year]
    for y in sorted(data.keys(), reverse=True):
        if data[y] is not None:
            return data[y]
    return None


# ── сборка ────────────────────────────────────────────────────────────────────

# Атомарные возрастные группы без агрегатов
ATOMIC_AGE_GROUPS = {
    "Zero years", "From 1 to 4 years", "From 5 to 9 years",
    "From 10 to 14 years", "From 15 to 19 years", "From 20 to 24 years",
    "From 25 to 29 years", "From 30 to 34 years", "From 35 to 39 years",
    "From 40 to 44 years", "From 45 to 49 years", "From 50 to 54 years",
    "From 55 to 59 years", "From 60 to 64 years", "From 65 to 69 years",
    "From 70 to 74 years", "From 75 to 79 years", "From 80 to 84 years",
    "From 85 to 89 years", "From 90 to 94 years", "From 95 to 99 years",
    "100 years or over",
}

def _filter_age_groups(ag: dict) -> dict:
    return {k: v for k, v in ag.items() if k in ATOMIC_AGE_GROUPS}

def _filter_education(edu: dict) -> dict:
    # убираем Total — это дубликат population
    return {k: v for k, v in edu.items() if k != "Total"}

def build_environment(year: int = 2024) -> dict:
    print("Loading data sources...")
    age_groups    = load_age_groups()
    wages         = load_wages()
    occupations   = load_occupations()
    nationalities = load_nationalities()
    education     = load_education()
    students      = load_students()
    city_pop      = load_city_population()
    kraje_pop     = load_kraje_population()
    labour        = load_labour()
    housing       = load_housing()
    marital_reg   = load_marital_by_region()
    marital_age   = load_marital_by_age()
    print("All sources loaded.")

    locations = {}
    regions   = {}
    cities    = {}

    # Нормализация написания между файлами (Śaľa в age_groups = Šaľa в wages)
    NAME_ALIASES = {
        "District of Śaľa":                        "District of Šaľa",
        "District of Dunajská Streda":           "District of Dunajská Streda",
        "District of Spišská Nová Ves":          "District of Spišská Nová Ves",
    }
    # Обратный маппинг: canonical → original (для age_groups ключей)
    NAME_ALIASES_REVERSE = {v: k for k, v in NAME_ALIASES.items()}
    def _normalize_name(name):
        return NAME_ALIASES.get(name, name)

    # Нормализуем housing и labour по региональным маппингам
    housing_normalized = {}
    for raw_name, data in housing.items():
        mapped = HOUSING_REGION_MAP.get(raw_name)
        if mapped:
            housing_normalized[mapped] = data

    labour_normalized = {}
    for raw_name, data in labour.items():
        mapped = LABOUR_REGION_MAP.get(raw_name)
        if mapped:
            labour_normalized[mapped] = data

    # Нормализуем marital_reg — ключи уже Region of X
    marital_reg_normalized = {}
    for raw_name, data in marital_reg.items():
        norm = _normalize_region(raw_name)
        if norm:
            marital_reg_normalized[norm] = data

    # Используем age_groups как источник имён локаций (самый полный)
    all_location_names = set(age_groups.keys())

    for name in all_location_names:
        norm = _normalize_region(name)
        if norm is None:
            continue

        canonical = _normalize_name(name)

        if name in CITY_AGGREGATES:
            region = CITY_TO_REGION.get(name, "Unknown")
            # Только districts которые реально входят в город (I/II/III/IV/V)
            # Košice-okolie исключается — это пригородный район вне города
            city_districts = [
                d for d, r in DISTRICT_TO_REGION.items()
                if r == region
                and name in d                        # "Bratislava" in "District of Bratislava I"
                and "okolie" not in d.lower()        # исключаем пригород
            ]
            cities[name] = {
                "type": "city",
                "region": region,
                "districts": sorted(city_districts),
                "age_groups": _filter_age_groups(age_groups.get(name, {})),
                "wages": wages.get(name, {}),
                "housing": housing_normalized.get(region, {}),
            }

        elif _is_district(name):
            region = DISTRICT_TO_REGION.get(name, "Unknown")
            locations[name] = {
                "type": "district",
                "region": region,
                "age_groups": _filter_age_groups(age_groups.get(name, {})),
                "wages": wages.get(canonical, {}),
                "occupations": occupations.get(canonical, {}),
                "education": _filter_education(education.get(name, {})),
                "nationalities": nationalities.get(name, {}),
                # региональный downscale — заполним ниже
                "housing": {},
                "labour": {},
            }

        elif _is_region(name):
            regions[name] = {
                "type": "region",
                "age_groups": _filter_age_groups(age_groups.get(name, {})),
                "wages": wages.get(name, {}),
                "occupations": occupations.get(name, {}),
                "education": _filter_education(education.get(name, {})),
                "nationalities": nationalities.get(name, {}),
                "housing": housing_normalized.get(name, {}),
                "labour": labour_normalized.get(name, {}),
                "marital_status": marital_reg_normalized.get(name, {}),
                "kraje_population": kraje_pop.get(name, {}),
            }

    # Downscale региональных данных на districts
    for dist_name, dist_data in locations.items():
        region_name = dist_data["region"]
        if region_name in regions:
            dist_data["housing"] = regions[region_name].get("housing", {})
            dist_data["labour"]  = regions[region_name].get("labour", {})

    # Добавляем Urban Audit данные (students, детальная популяция) к городам
    for city_name in CITY_AGGREGATES:
        if city_name in cities:
            cities[city_name]["city_population_detail"] = city_pop.get(city_name, {})
            cities[city_name]["students"] = students.get(city_name, {})

    return {
        "meta": {
            "year": year,
            "source": "Datacube SK (datacube.statistics.sk)",
            "districts_count": len(locations),
            "regions_count": len(regions),
            "cities_count": len(cities),
            "marital_by_age": marital_age,   # национальный уровень
        },
        "locations": locations,
        "regions": regions,
        "cities": cities,
    }


if __name__ == "__main__":
    import sys
    out_path = Path(__file__).parent / "environment.json"
    env = build_environment(year=2024)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)

    meta = env["meta"]
    print(f"\n✓ environment.json saved to {out_path}")
    print(f"  Districts : {meta['districts_count']}")
    print(f"  Regions   : {meta['regions_count']}")
    print(f"  Cities    : {meta['cities_count']}")
    print(f"  File size : {out_path.stat().st_size / 1024:.1f} KB")
