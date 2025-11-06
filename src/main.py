import numpy as np
import math
import random
import time
import classes as cl
import visual as viz

# ----------------------------- ПАРАМЕТРЫ ТЕСТА ------------------------------
SUCCESS_RADIUS_M = 2.0    # (м) Целевой радиус для успешной посадки
CRITICAL_COLLISION_TOLERANCE = 1e-4 # (м) Порог для "разбились" (почти нулевое расстояние)

# Начальные условия
INITIAL_R_MIN = 500.0      # (м) Минимальный радиус старта
INITIAL_R_MAX = 5000.0     # (м) Максимальный радиус старта
INITIAL_Z_MIN = 200.0      # (м) Минимальная высота старта
INITIAL_Z_MAX = 2000.0      # (м) Максимальная высота старта
INITIAL_V_MAX = 55.0      # (м/с) Максимальная начальная скорость
# ---------------------------------------------------------------------------


def initialize_random_scenario() -> tuple[cl.Beacon, cl.Drone, cl.Environment, cl.Controller]:
    beacon = cl.Beacon(x=0.0, y=0.0, z=0.0)

    R_start = np.random.uniform(INITIAL_R_MIN, INITIAL_R_MAX)
    Z_start = np.random.uniform(INITIAL_Z_MIN, INITIAL_Z_MAX)
    angle = np.random.uniform(0, 2 * math.pi)

    X_start = R_start * math.cos(angle)
    Y_start = R_start * math.sin(angle)

    V_start = np.random.uniform(5.0, INITIAL_V_MAX)

    drone = cl.Drone(x=X_start, y=Y_start, z=Z_start, speed=V_start)

    env = cl.Environment(beacon)          # один-единственный env
    drone.point_toward_beacon(env)        # ориентируемся в том же env

    controller = cl.Controller()
    return beacon, drone, env, controller



def run_landing_test(viz_on: bool = False):
    """Главный цикл симуляции с подробным логированием и валидацией."""
    vis = viz.Visualizer(success_radius_m=SUCCESS_RADIUS_M) if viz_on else None
    # 1. Инициализация сценария
    beacon, drone, env, controller = initialize_random_scenario()
    
    print("--- НАЧАЛО ТЕСТА ПОСАДКИ (M1) ---")
    print(f"  Начальная позиция (X, Y, Z): [{drone.pos[0]:.2f}, {drone.pos[1]:.2f}, {drone.pos[2]:.2f}]")
    print(f"  Начальная скорость: {drone.speed:.2f} м/с")
    print(f"  Условие успеха: R < {SUCCESS_RADIUS_M} м")
    print("-" * 50)
    
    # Вспомогательные переменные для логирования
    log_timer = 0.0
    total_sim_time = 0.0
    vis = viz.Visualizer(success_radius_m=SUCCESS_RADIUS_M, fmt="mp4", fps=24)

    while True: # Бесконечный цикл, который прервется по условию посадки
        
        dt = cl.SIM_DT
        
        # --- БЛОК СИМУЛЯЦИИ ---
        env.update(dt)
        vis.capture(beacon, drone, env, total_sim_time)
        drone.integrate(dt)
        total_sim_time += dt

        # --- КРИТИЧЕСКАЯ ПРОВЕРКА (Сразу после интеграции) ---
        if drone.pos[2] == 0.0 and drone.speed == 0.0:
            print(f"[{total_sim_time:.1f}с] Посадка обнаружена. Переход к валидации.")
            break
        
        # --- БЛОК КОНТРОЛЛЕРА ---

        controller.update(env, drone, dt)
        
        # --- БЛОК ЛОГИРОВАНИЯ (раз в оборот маяка ~1.0 сек) ---
        log_timer += dt
        if log_timer >= cl.TDM_SWITCH_RATE:
            log_timer = 0.0
            
            # Логика из контроллера (если там были собраны 2 сектора)
            log_sectors = list(controller.power_readings.items())
            
            sector_log = "НЕТ ДАННЫХ"
            if len(log_sectors) == 2:
                s0_id, p0 = log_sectors[0]
                s1_id, p1 = log_sectors[1]
                sector_log = f"S{s0_id}:{p0:.2f} | S{s1_id}:{p1:.2f} (Сумма: {controller.last_total_power:.2f} W/m²)"
            elif len(log_sectors) == 1:
                s_id, p = log_sectors[0]
                sector_log = f"S{s_id}:{p:.2f} (Сумма: {controller.last_total_power:.2f} W/m²). ИЩУ ШОВ."
            
            pos = drone.pos
            spd = drone.speed
            dir_mag = np.linalg.norm(drone.direction)
            
            print(f" Маяк(x,y,z) = {beacon.pos} | T={total_sim_time:.1f}с | X:{pos[0]:.2f} | Y:{pos[1]:.2f} | Z:{pos[2]:.2f}м | V:{spd:.2f} м/с | Мощность: {sector_log}")
            # print(f"    Pos XY: [{pos[0]:.2f}, {pos[1]:.2f}] | Dir Z: {drone.direction[2]:.2f}")
            print("-" * 50)


    # ========================== ВАЛИДАЦИЯ РЕЗУЛЬТАТА ==========================
    
    landing_pos_xy = drone.pos[:2]
    distance_from_beacon = np.linalg.norm(landing_pos_xy)
    
    print("\n--- ФИНАЛЬНЫЙ ВЕРДИКТ ---")
    print(f"  Финальная позиция (X, Y): [{landing_pos_xy[0]:.4f}, {landing_pos_xy[1]:.4f}]")
    print(f"  Дистанция до маяка (0,0): {distance_from_beacon:.4f} м")
    
    if distance_from_beacon < CRITICAL_COLLISION_TOLERANCE:
        print(f"  💥 РЕЗУЛЬТАТ: КРИТИЧЕСКИЙ ПРОВАЛ (КОЛЛИЗИЯ)")
        print(f"    Дрон упал *прямо* на маяк (R < {CRITICAL_COLLISION_TOLERANCE} м).")
    elif distance_from_beacon <= SUCCESS_RADIUS_M:
        print(f"  ✅ РЕЗУЛЬТАТ: УСПЕХ (ПОСАДКА)")
        print(f"    Посадка в целевом радиусе {SUCCESS_RADIUS_M} м.")
    else:
        print(f"  ❌ РЕЗУЛЬТАТ: ПРОВАЛ (МИМО ЦЕЛИ)")
        print(f"    Посадка за пределами радиуса {SUCCESS_RADIUS_M} м.")
    
    print("-" * 50)
    
    if viz_on:
        vis.plot_all(beacon_xy=(0.0, 0.0))

# --- Запуск теста ---
if __name__ == "__main__":
    run_landing_test(True)