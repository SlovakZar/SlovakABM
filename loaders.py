"""
loaders.py — каждая функция читает один xlsx и возвращает чистый dict.
Ключ везде — название локации (District/Region/City) как есть в данных.
Нормализация названий происходит в initializer.py через mappings.
"""

import pandas as pd
from pathlib import Path

RAW = Path(__file__).parent / "raw"

# ── утилиты ──────────────────────────────────────────────────────────────────

def _read(filename: str, sheet=0, skiprows=0) -> pd.DataFrame:
    return pd.read_excel(RAW / filename, sheet_name=sheet, header=None, skiprows=skiprows)

def _parse_number(val):
    """'1 749' → 1749, '100,7' → 100.7, NaN → None"""
    if pd.isna(val):
        return None
    s = str(val).replace(" ", "").replace(",", ".")
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return None

def _extract_location_series(df: pd.DataFrame, year_col: int = 1) -> dict:
    """
    Стандартная структура большинства файлов:
      row 0 — год (в year_col начиная с col 1)
      row 1+ — (location_name, value_2024, value_2023, ...)
    Возвращает {location: {2024: v, 2023: v, ...}}
    """
    years_row = df.iloc[0]
    years = []
    for v in years_row[year_col:]:
        n = _parse_number(v)
        if n is not None:
            years.append(int(n))

    result = {}
    for _, row in df.iloc[1:].iterrows():
        loc = str(row.iloc[0]).strip()
        if not loc or loc == "nan":
            continue
        values = {}
        for i, year in enumerate(years):
            values[year] = _parse_number(row.iloc[year_col + i])
        result[loc] = values
    return result


# ── население по муниципалитетам ──────────────────────────────────────────────

def load_population() -> dict:
    """
    Файл: Население_по_муниципалитетам.xlsx
    Один лист. Строки: локации (Country/City/Region/District).
    Возвращает {location: {year: population}}
    """
    df = _read("Население_по_муниципалитетам.xlsx", skiprows=5)
    df.columns = range(len(df.columns))
    return _extract_location_series(df)


# ── возрастные группы ─────────────────────────────────────────────────────────

def load_age_groups() -> dict:
    """
    Файл: Группы_возрастов_по_муниципалитетам.xlsx
    Листы — возрастные группы, внутри каждого: мужчины (_1 в имени = женщины).
    Структура: 2 колонки — location | value (только 2024).
    Возвращает {location: {age_group: {sex: {year: count}}}}
    """
    xl = pd.ExcelFile(RAW / "Группы_возрастов_по_муниципалитетам.xlsx")
    result = {}

    for sheet_name in xl.sheet_names:
        is_female = sheet_name.endswith("_1")
        age_group = sheet_name.removesuffix("_1")
        sex = "female" if is_female else "male"

        # skiprows=7: первая строка = NaN | 2024, данные с строки 2
        df = pd.read_excel(RAW / "Группы_возрастов_по_муниципалитетам.xlsx",
                           sheet_name=sheet_name, header=None, skiprows=7)
        df.columns = range(len(df.columns))

        # Первая строка содержит год
        year_val = _parse_number(df.iloc[0, 1])
        year = int(year_val) if year_val else 2024

        for _, row in df.iloc[1:].iterrows():
            loc = str(row.iloc[0]).strip()
            if not loc or loc == "nan":
                continue
            val = _parse_number(row.iloc[1])
            result.setdefault(loc, {})
            result[loc].setdefault(age_group, {})
            result[loc][age_group].setdefault(sex, {})
            result[loc][age_group][sex][year] = val

    return result


# ── зарплаты по секторам ──────────────────────────────────────────────────────

def load_wages() -> dict:
    """
    Файл: Зарлпаты_по_муниципалитетам_и_типам.xlsx
    Листы — секторы экономики.
    Возвращает {location: {sector: {year: avg_wage_eur}}}
    """
    xl = pd.ExcelFile(RAW / "Зарлпаты_по_муниципалитетам_и_типам.xlsx")
    result = {}

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(RAW / "Зарлпаты_по_муниципалитетам_и_типам.xlsx",
                           sheet_name=sheet_name, header=None, skiprows=7)
        df.columns = range(len(df.columns))
        series = _extract_location_series(df)

        for loc, years_data in series.items():
            result.setdefault(loc, {})
            result[loc][sheet_name] = years_data

    return result


# ── занятость по профессиям ───────────────────────────────────────────────────

def load_occupations() -> dict:
    """
    Файл: Рабочая_область_по_муниципалитетам.xlsx
    Листы — типы занятости (Managers, Professionals, ...).
    Возвращает {location: {occupation: {year: count}}}
    """
    xl = pd.ExcelFile(RAW / "Рабочая_область_по_муниципалитетам.xlsx")
    result = {}

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(RAW / "Рабочая_область_по_муниципалитетам.xlsx",
                           sheet_name=sheet_name, header=None, skiprows=7)
        df.columns = range(len(df.columns))
        series = _extract_location_series(df)

        for loc, years_data in series.items():
            result.setdefault(loc, {})
            result[loc][sheet_name] = years_data

    return result


# ── национальности (М/Ж) ──────────────────────────────────────────────────────

def load_nationalities() -> dict:
    """
    Файл: Националььности__МЖ.xlsx
    Листы без _1 = мужчины, с _1 = женщины.
    Возвращает {location: {nationality: {sex: {year: count}}}}
    """
    xl = pd.ExcelFile(RAW / "Националььности__МЖ.xlsx")
    result = {}

    for sheet_name in xl.sheet_names:
        is_female = sheet_name.endswith("_1")
        nationality = sheet_name.removesuffix("_1")
        sex = "female" if is_female else "male"

        df = pd.read_excel(RAW / "Националььности__МЖ.xlsx",
                           sheet_name=sheet_name, header=None, skiprows=6)
        df.columns = range(len(df.columns))
        if df.shape[0] == 0:
            continue
        series = _extract_location_series(df)

        for loc, years_data in series.items():
            result.setdefault(loc, {})
            result[loc].setdefault(nationality, {})
            result[loc][nationality][sex] = years_data

    return result


# ── семейное положение ────────────────────────────────────────────────────────

def load_marital_status() -> dict:
    """
    Файл: Популяция_по_браку.xlsx
    Листы: Total, Single, Married, Divorced, Widowed, Unknown
    Возвращает {location: {status: {year: count}}}
    """
    xl = pd.ExcelFile(RAW / "Популяция_по_браку.xlsx")
    result = {}

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(RAW / "Популяция_по_браку.xlsx",
                           sheet_name=sheet_name, header=None, skiprows=7)
        df.columns = range(len(df.columns))
        series = _extract_location_series(df)

        for loc, years_data in series.items():
            result.setdefault(loc, {})
            result[loc][sheet_name] = years_data

    return result


# ── образование ───────────────────────────────────────────────────────────────

def load_education() -> dict:
    """
    Файл: Популяция_по_образованию.xlsx
    Листы: Total, Without education, Basic, Secondary*, University, Unknown
    Возвращает {location: {edu_level: {year: count}}}
    """
    xl = pd.ExcelFile(RAW / "Популяция_по_образованию.xlsx")
    result = {}

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(RAW / "Популяция_по_образованию.xlsx",
                           sheet_name=sheet_name, header=None, skiprows=8)
        df.columns = range(len(df.columns))
        series = _extract_location_series(df)

        for loc, years_data in series.items():
            result.setdefault(loc, {})
            result[loc][sheet_name] = years_data

    return result


# ── студенты и школьники по городам ──────────────────────────────────────────

def load_students() -> dict:
    """
    Файл: Студенты_и_школьники_по_городам.xlsx
    Один лист, строки — показатели по городам.
    Возвращает {city: {indicator: {year: value}}}
    """
    df = _read("Студенты_и_школьники_по_городам.xlsx", skiprows=4)
    df.columns = range(len(df.columns))

    years = []
    for v in df.iloc[0, 3:]:
        n = _parse_number(v)
        if n is not None:
            years.append(int(n))

    result = {}
    for _, row in df.iloc[1:].iterrows():
        city = str(row.iloc[0]).strip()
        indicator = str(row.iloc[1]).strip()
        sex_group = str(row.iloc[2]).strip()
        if city == "nan":
            continue
        key = f"{indicator} | {sex_group}"
        result.setdefault(city, {})
        result[city][key] = {
            years[i]: _parse_number(row.iloc[3 + i])
            for i in range(len(years))
        }
    return result


# ── население городов (Urban Audit) ──────────────────────────────────────────

def load_city_population() -> dict:
    """
    Файл: City_population.xlsx
    Города + возрастные группы + пол.
    Возвращает {city: {age_group: {sex: {year: value}}}}
    """
    df = _read("City_population.xlsx", skiprows=4)
    df.columns = range(len(df.columns))

    years = []
    for v in df.iloc[0, 3:]:
        n = _parse_number(v)
        if n is not None:
            years.append(int(n))

    result = {}
    for _, row in df.iloc[1:].iterrows():
        city = str(row.iloc[0]).strip()
        age_group = str(row.iloc[1]).strip()
        sex = str(row.iloc[2]).strip()
        if city == "nan":
            continue
        result.setdefault(city, {})
        result[city].setdefault(age_group, {})
        result[city][age_group][sex] = {
            years[i]: _parse_number(row.iloc[3 + i])
            for i in range(len(years))
        }
    return result


# ── население регионов (Kraje) ────────────────────────────────────────────────

def load_kraje_population() -> dict:
    """
    Файл: Kraje_population.xlsx
    Регионы + возрастные группы + пол.
    Возвращает {region: {age_group: {sex: {year: value}}}}
    """
    df = _read("Kraje_population.xlsx", skiprows=5)
    df.columns = range(len(df.columns))

    years = []
    for v in df.iloc[0, 2:]:
        n = _parse_number(v)
        if n is not None:
            years.append(int(n))

    result = {}
    for _, row in df.iloc[1:].iterrows():
        region = str(row.iloc[0]).strip()
        age_group = str(row.iloc[1]).strip()
        sex = str(row.iloc[2]).strip()
        if region == "nan":
            continue
        result.setdefault(region, {})
        result[region].setdefault(age_group, {})
        result[region][age_group][sex] = {
            years[i]: _parse_number(row.iloc[3 + i])
            for i in range(len(years))
        }
    return result


# ── рынок труда по регионам ───────────────────────────────────────────────────

def load_labour() -> dict:
    """
    Файл: Labour_kraje.xlsx
    Один лист: регион + индикатор занятости + значения по годам.
    Возвращает {region: {indicator: {year: value}}}
    """
    df = _read("Labour_kraje.xlsx", skiprows=4)
    df.columns = range(len(df.columns))

    years = []
    for v in df.iloc[0, 1:]:
        n = _parse_number(v)
        if n is not None:
            years.append(int(n))

    result = {}
    for _, row in df.iloc[1:].iterrows():
        region = str(row.iloc[0]).strip()
        indicator = str(row.iloc[1]).strip()
        if region == "nan":
            continue
        result.setdefault(region, {})
        result[region][indicator] = {
            years[i]: _parse_number(row.iloc[2 + i])
            for i in range(len(years))
        }
    return result


# ── цены на жильё по регионам ────────────────────────────────────────────────

def load_housing() -> dict:
    """
    Файл: Flats_prices_kraje.xlsx
    Один лист: регион + показатель + значения по годам.
    Возвращает {region: {indicator: {year: value}}}
    """
    df = _read("Flats_prices_kraje.xlsx", skiprows=4)
    df.columns = range(len(df.columns))

    years = []
    for v in df.iloc[0, 1:]:
        n = _parse_number(v)
        if n is not None:
            years.append(int(n))

    result = {}
    for _, row in df.iloc[1:].iterrows():
        region = str(row.iloc[0]).strip()
        indicator = str(row.iloc[1]).strip()
        if region == "nan":
            continue
        result.setdefault(region, {})
        result[region][indicator] = {
            years[i]: _parse_number(row.iloc[2 + i])
            for i in range(len(years))
        }
    return result


# ── брак по регионам (М/Ж × статус) ─────────────────────────────────────────

_MARITAL_SHEET_STATUS = {
    "Men":     ("Single person",   "male"),
    "Women":   ("Single person",   "female"),
    "Men_1":   ("Married person",  "male"),
    "Women_1": ("Married person",  "female"),
    "Men_2":   ("Divorced person", "male"),
    "Women_2": ("Divorced person", "female"),
    "Men_3":   ("Widowed person",  "male"),
    "Women_3": ("Widowed person",  "female"),
    "Men_4":   ("Unknown",         "male"),
    "Women_4": ("Unknown",         "female"),
}

def load_marital_by_region() -> dict:
    """
    Файл: Брак_по_регионам_МЖ.xlsx
    Листы: Men/Women × 5 статусов.
    Возвращает {region: {status: {sex: {year: count}}}}
    """
    xl = pd.ExcelFile(RAW / "Брак_по_регионам_МЖ.xlsx")
    result = {}

    for sheet_name in xl.sheet_names:
        if sheet_name not in _MARITAL_SHEET_STATUS:
            continue
        status, sex = _MARITAL_SHEET_STATUS[sheet_name]

        df = pd.read_excel(RAW / "Брак_по_регионам_МЖ.xlsx",
                           sheet_name=sheet_name, header=None, skiprows=7)
        df.columns = range(len(df.columns))
        series = _extract_location_series(df)

        for region, years_data in series.items():
            result.setdefault(region, {})
            result[region].setdefault(status, {})
            result[region][status][sex] = years_data

    return result


# ── брак по возрастным группам (национальный уровень) ────────────────────────

def load_marital_by_age() -> dict:
    """
    Файл: Брак_по_возрастным_группам_МЖ.xlsx
    Статус × возрастная группа × пол × год. Уровень: Slovak Republic.
    Возвращает {status: {age_group: {sex: {year: count}}}}
    """
    df = pd.read_excel(RAW / "Брак_по_возрастным_группам_МЖ.xlsx",
                       header=None, skiprows=4)
    df.columns = range(len(df.columns))

    # row 0: years (2024 2024 2023 2023 ...)
    # row 1: Males/Females alternating
    # row 2+: data
    year_row = df.iloc[0, 2:].tolist()
    sex_row  = df.iloc[1, 2:].tolist()

    col_meta = []
    for i, (y, s) in enumerate(zip(year_row, sex_row)):
        sex  = "male" if str(s).strip() == "Males" else "female"
        year = int(_parse_number(y)) if _parse_number(y) else None
        col_meta.append((i, year, sex))

    result = {}
    current_status = None

    for _, row in df.iloc[2:].iterrows():
        status_val = str(row.iloc[0]).strip()
        age_group  = str(row.iloc[1]).strip()

        if status_val and status_val != "nan":
            current_status = status_val
        if not current_status or age_group == "nan":
            continue
        if age_group == "Age groups total":
            continue

        result.setdefault(current_status, {})
        result[current_status].setdefault(age_group, {})

        for col_i, year, sex in col_meta:
            if year is None:
                continue
            val = _parse_number(row.iloc[2 + col_i])
            result[current_status][age_group].setdefault(sex, {})
            result[current_status][age_group][sex][year] = val

    return result
