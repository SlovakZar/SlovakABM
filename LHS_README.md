# LHS-тестирование параметров SlovakABM

## Что сделано

### 1. `lhs_parameters.json` — Спецификация параметров
Структурированы **~50+ параметров** в 6 категорий:

| Категория | Описание | Кол-во |
|-----------|----------|--------|
| `TWO_BARRIER_MODEL` | Двухбарьерная модель (aspirations, decay, smoothing) | 5 |
| `HEURISTIC_SEARCH` | Эвристический поиск (зарплатные ожидания, жильё, адаптация) | 11 |
| `GRAPH_ENVIRONMENT` | Реакция среды (housing/wage alpha, weak_ties) | 4 |
| `SIGNAL_SYSTEM` | Сигнальная система v2 (social_boost, penalty, decay) | 14 |
| `AGENT_CREATION` | Инициализация агентов (Rogers-Castro, education, shock, type_modifiers) | 18 |
| `AGENT_CORRELATIONS` | Условные связи параметров (pc→econ_pc, digital→info) | 4 |

Каждый параметр имеет:
- `default` — текущее значение в коде
- `range: [min, max]` — границы для LHS-сэмплирования
- `description` — что делает параметр

### 2. `lhs_runner.py` — Фреймворк для LHS
Ключевые классы/функции:
- `load_param_specs()` — загрузка спецификации
- `lhs_sample()` — генерация LHS-матрицы
- `ParamPatcher` — подмена констант в engine/graph/signals на время прогона
- `create_patched_dispatcher()` — пересоздание сигнальной шины с новыми параметрами
- `lhs_test()` — главный цикл: сэмплы → прогоны → сбор метрик
- `sensitivity_analysis()` — корреляция параметров с метриками

## Быстрый старт

```bash
# Тестовый прогон: 20 сэмплов × 3000 агентов × 24 тика (~5 мин)
python lhs_runner.py --samples 20 --agents 3000 --ticks 24 --output test_results.csv

# Полный прогон: 100 сэмплов × 5000 агентов × 36 тиков (~30-40 мин)
python lhs_runner.py --samples 100 --agents 5000 --ticks 36 --output lhs_results.csv --sensitivity
```

## Выходные метрики (20+)
- `unemployment_rate`, `n_commuters`, `commuter_share`
- `avg_wage_employed`, `median_wage_employed`, `wage_q25`, `wage_q75`, `wage_std`
- `avg_sat_economic/social/family/place`
- `avg_aspirations`, `avg_signal_reduction`, `tpb_active_share`
- `n_moved_economic`, `n_moved_place`, `n_commute_started`, `n_adapted`, `n_stay_decision`
- `regional_population_gini`, `regional_wage_std`
- `jobs_pressure_max_final`, `avg_dissat_final`
- Динамические: `avg_econ_penalty`, `avg_inertia_mobility_penalty` и др.

## Что НЕ вынесено (структурные константы)
Эти параметры фиксированы, т.к. они определяют архитектуру модели:
- Структура графа (79 районов, commuting-матрица)
- Формулы dissatisfaction/D_instant/TPB (менять можно только веса)
- Типы агентов и их модификаторы (вынесены как варьируемые ranges)
- Распределения из опроса SASD (global_mean/std вынесены)
- Шкалы опросных переменных

При необходимости любой параметр можно добавить в `lhs_parameters.json` и прописать маппинг в `ParamPatcher.MAPPING`.

## Анализ результатов

```python
import pandas as pd
from lhs_runner import sensitivity_analysis

df = pd.read_csv("lhs_results.csv")
sens = sensitivity_analysis(df, top_n=30)
print(sens)

# Топ-параметры по влиянию на безработицу:
print(df.corr()["unemployment_rate"].abs().sort_values(ascending=False).head(10))
```
