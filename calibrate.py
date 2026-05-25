#!/usr/bin/env python3
"""
calibrate.py – Калибровка параметров ABM для достижения 1.5–2% годовой миграции.
Запускать: python calibrate.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

# Импорт из вашего проекта
from graph import build_graph, print_graph_summary
from agents import create_agents
from engine import run_simulation

# ============================================================
# 1. КОНФИГУРАЦИЯ КАЛИБРОВКИ
# ============================================================
# Целевой уровень миграции за 6 тиков (полгода) = 1.75% годовых * 0.5 = 0.875%
TARGET_RATE = 0.00875   # 0.875%

# Параметры быстрой симуляции
N_AGENTS_CALIB = 5000   # вместо 70000
N_TICKS_CALIB  = 6      # полугодие
N_SEEDS = 3             # число сидов для усреднения (стохастичность)

# Параметры оптимизации
N_INIT_POINTS = 5       # случайных стартовых точек
N_CALLS = 20            # общее число итераций (включая init)

# Диапазоны параметров калибровки
PARAM_NAMES = ['inertia_reduction', 'threshold_offset', 'control_boost']
PARAM_BOUNDS = [
    (0.7, 1.0),   # inertia_reduction
    (-0.2, 0.2),  # threshold_offset
    (0.0, 0.2)    # control_boost
]

# Пути к файлам (такие же, как в run.py по умолчанию)
ENV_PATH = "environment.json"
COMMUTING_PATH = "commuting_filtered_with_travel.csv"
AGENT_DIST_PATH = "agent_init_distributions.json"

# ============================================================
# 2. ФУНКЦИЯ ЗАПУСКА СИМУЛЯЦИИ С ЗАДАННЫМИ ПАРАМЕТРАМИ
# ============================================================
def run_with_params(param_dict, seed, verbose=False):
    """
    Запускает один прогон симуляции с модифицированными параметрами агентов.
    Возвращает долю переездов (moves / total_agents).
    """
    # 1. Строим граф (один раз на всё время, можно закешировать, но для простоты оставим)
    G = build_graph(ENV_PATH, COMMUTING_PATH)
    
    # 2. Создаём агентов (уменьшенное количество)
    df = create_agents(AGENT_DIST_PATH, n_agents=N_AGENTS_CALIB, seed=seed)
    
    # 3. Модифицируем атрибуты агентов в соответствии с параметрами калибровки
    # inertia
    df['inertia'] = np.clip(df['inertia'] * param_dict['inertia_reduction'], 0.05, 0.95)
    # perceived_control
    df['perceived_control'] = np.clip(df['perceived_control'] + param_dict['control_boost'], 0.0, 1.0)
    # пороги (thr_economic, thr_social, thr_family, thr_place)
    for col in ['thr_economic', 'thr_social', 'thr_family', 'thr_place']:
        df[col] = np.clip(df[col] + param_dict['threshold_offset'], 0.0, 1.0)
    
    # 4. Запускаем симуляцию (без вывода в консоль, чтобы не захламлять)
    #    run_simulation возвращает (df_final, snapshots, tick_stats)
    _, _, tick_stats = run_simulation(
        df, G,
        n_ticks=N_TICKS_CALIB,
        snapshot_ticks=[],       # не нужны снимки
        seed=seed,
        verbose=False
    )
    
    # 5. Суммируем переезды за все тики
    total_moves = sum(stat['moves'] for stat in tick_stats)
    move_rate = total_moves / N_AGENTS_CALIB
    
    if verbose:
        print(f"    seed={seed}: moves={total_moves}, rate={move_rate:.4f}")
    
    return move_rate


def objective(params_list):
    """
    Целевая функция для оптимизации.
    params_list: список значений параметров в порядке PARAM_NAMES
    Возвращает квадратичное отклонение от TARGET_RATE.
    """
    param_dict = dict(zip(PARAM_NAMES, params_list))
    rates = []
    for seed in range(N_SEEDS):
        rate = run_with_params(param_dict, seed=seed + 10000)
        rates.append(rate)
    avg_rate = np.mean(rates)
    loss = (avg_rate - TARGET_RATE) ** 2
    print(f"Params: {param_dict} -> avg_rate={avg_rate:.4f} (target {TARGET_RATE:.4f}) loss={loss:.6f}")
    return loss

# ============================================================
# 3. ЗАПУСК ОПТИМИЗАЦИИ
# ============================================================
def run_calibration():
    print("=" * 70)
    print("КАЛИБРОВКА МОДЕЛИ МИГРАЦИИ")
    print(f"Целевая доля переездов за {N_TICKS_CALIB} тиков: {TARGET_RATE:.4f}")
    print(f"Агентов на прогон: {N_AGENTS_CALIB}")
    print(f"Сидов на точку: {N_SEEDS}")
    print(f"Итераций оптимизации: {N_CALLS}")
    print("=" * 70)
    
    # Используем байесовскую оптимизацию из scikit-optimize
    try:
        from skopt import gp_minimize
        from skopt.space import Real
        from skopt.utils import use_named_args
        
        dimensions = [Real(low, high, name=name) for (low, high), name in zip(PARAM_BOUNDS, PARAM_NAMES)]
        
        @use_named_args(dimensions)
        def skopt_loss(**params):
            return objective([params[name] for name in PARAM_NAMES])
        
        start_time = time.time()
        res = gp_minimize(
            skopt_loss,
            dimensions=dimensions,
            n_calls=N_CALLS,
            n_initial_points=N_INIT_POINTS,
            acq_func='EI',
            random_state=42,
            verbose=True
        )
        elapsed = time.time() - start_time
        
        best_params = dict(zip(PARAM_NAMES, res.x))
        best_loss = res.fun
        best_rate = np.sqrt(best_loss) + TARGET_RATE  # приблизительная достигнутая доля
        
    except ImportError:
        print("scikit-optimize не установлен. Установите: pip install scikit-optimize")
        print("Использую простой случайный поиск (100 точек)")
        best_loss = float('inf')
        best_params = None
        start_time = time.time()
        for i in range(100):
            params = [np.random.uniform(low, high) for (low, high) in PARAM_BOUNDS]
            loss = objective(params)
            if loss < best_loss:
                best_loss = loss
                best_params = dict(zip(PARAM_NAMES, params))
                print(f"  Новый лучший: loss={loss:.6f}, params={best_params}")
        elapsed = time.time() - start_time
        best_rate = np.sqrt(best_loss) + TARGET_RATE
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТ КАЛИБРОВКИ")
    print(f"  Лучшие параметры:")
    for k, v in best_params.items():
        print(f"    {k}: {v:.4f}")
    print(f"  Достигнутая доля переездов: {best_rate:.4f} (цель {TARGET_RATE:.4f})")
    print(f"  Время выполнения: {elapsed/60:.2f} мин")
    
    # Сохраняем результат в JSON
    result = {
        'best_params': best_params,
        'target_rate': TARGET_RATE,
        'achieved_rate': best_rate,
        'n_agents': N_AGENTS_CALIB,
        'n_ticks': N_TICKS_CALIB,
        'n_seeds': N_SEEDS,
        'iterations': N_CALLS,
        'elapsed_minutes': elapsed/60
    }
    with open('calibration_output.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("\nРезультаты сохранены в calibration_output.json")
    
    return best_params

# ============================================================
# 4. ЗАПУСК
# ============================================================
if __name__ == '__main__':
    best = run_calibration()
    print("\nЧтобы использовать эти параметры в основной симуляции, добавьте в run.py:")
    print("  # После создания агентов:")
    print("  df['inertia'] = df['inertia'] * {:.4f}".format(best['inertia_reduction']))
    print("  df['perceived_control'] = np.clip(df['perceived_control'] + {:.4f}, 0, 1)".format(best['control_boost']))
    for col in ['thr_economic', 'thr_social', 'thr_family', 'thr_place']:
        print(f"  df['{col}'] = np.clip(df['{col}'] + {best['threshold_offset']:.4f}, 0, 1)")
