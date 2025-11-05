import numpy as np
import math
import random
# Импорт классов и константы SIM_DT из файла с логикой
from classes import Beacon, Environment, Drone, Controller, SIM_DT

# ---------------------------------- СЦЕНАРИЙ СИМУЛЯЦИИ ---------------------------------

def run_full_simulation():
    """Запускает симуляцию с выводом в консоль."""

    # --- Инициализация ---
    beacon = Beacon(0.0, 0.0, 0.0)
    env = Environment(beacon)

    # Случайная стартовая позиция
    x0 = random.uniform(50, 100)
    y0 = random.uniform(-100, 100)
    z0 = random.uniform(20, 40)
    
    # 1. Начальная скорость (импортирована из classes.py, по умолчанию 20.0)
    initial_speed = 20.0 
    drone = Drone(x0, y0, z0, speed=initial_speed)

    # 2. Вводим ошибку курса (45 градусов)
    initial_direction = beacon.pos - drone.pos
    yaw_to_beacon = math.atan2(initial_direction[1], initial_direction[0])
    initial_yaw_error = math.radians(45.0)
    initial_yaw = yaw_to_beacon + initial_yaw_error

    # Задаем направление с ошибкой и тангажем на снижение
    pitch_down_5deg = math.radians(-5.0)
    drone.direction[0] = math.cos(pitch_down_5deg) * math.cos(initial_yaw)
    drone.direction[1] = math.cos(pitch_down_5deg) * math.sin(initial_yaw)
    drone.direction[2] = math.sin(pitch_down_5deg)
    drone.direction /= np.linalg.norm(drone.direction)


    controller = Controller(drone, env)

    max_steps = 15000 # Ограничение по времени (15000 * 0.1 с = 1500 секунд)
    t = 0.0
    
    # --- Вывод заголовка ---
    print("=================================================================================================")
    print(f"НАЧАЛО СИМУЛЯЦИИ: Дрон ({x0:.1f}, {y0:.1f}, {z0:.1f}) -> Маяк (0, 0, 0)")
    print(f"Начальная скорость: {drone.speed:.1f} м/с, Ошибка курса: {math.degrees(initial_yaw_error):.1f}°")
    print(f"Целевая скорость (V_target): {controller.SPEED_TARGET:.1f} м/с")
    print("=================================================================================================")
    print(f"{'Time (s)':<10}{'X, Y, Z (m)':<30}{'Speed (m/s)':<15}{'Flux (W/m²)' :<15}{'Sectors (P_curr / P_prev)'}")
    print("-------------------------------------------------------------------------------------------------")
    
    # --- Основной цикл ---
    for step in range(max_steps):

        # 1. Шаг управления
        controller.run_control_cycle(SIM_DT)
        
        # 2. Измерение (для вывода)
        current_power, current_sector = drone.measure_flux(env)
        
        # 3. Вывод состояния
        if step % 10 == 0: # Выводим каждую секунду
            
            p_prev_info = f"{controller.last_sector_id} ({controller.last_power_flux:.3f})" if controller.is_landing_activated else "N/A"
                
            print(
                f"{t:<10.1f}"
                f"({drone.pos[0]:.1f}, {drone.pos[1]:.1f}, {drone.pos[2]:.1f}){chr(8214):<2}"
                f"{drone.speed:<15.1f}"
                f"{current_power:<15.3f}"
                f"{current_sector} ({current_power:.3f}) / {p_prev_info}"
            )
        
        # 4. Проверка завершения
        dist_xy = np.linalg.norm(drone.pos[:2] - beacon.pos[:2])
        if dist_xy < 3.0 and drone.pos[2] <= 0.05:
            # Успешная посадка
            print("\n=================================================================================================")
            print("✅ ПОСАДКА ВЫПОЛНЕНА УСПЕШНО!")
            print(f"Финальная позиция (X, Y, Z): ({drone.pos[0]:.3f}, {drone.pos[1]:.3f}, {drone.pos[2]:.3f})")
            print(f"Пройдено времени: {t:.1f} с")
            print("=================================================================================================")
            return True
        
        if drone.pos[2] <= 0.0 and dist_xy > 3.0:
            # Неуспешная посадка (упал или приземлился далеко)
            print("\n=================================================================================================")
            print("❌ ПОСАДКА НЕ УСПЕШНА: Дрон достиг земли слишком далеко от цели.")
            print(f"Финальная позиция (X, Y, Z): ({drone.pos[0]:.3f}, {drone.pos[1]:.3f}, {drone.pos[2]:.3f})")
            print(f"Расстояние до цели: {dist_xy:.1f} м")
            print("=================================================================================================")
            return False


        if t >= max_steps * SIM_DT:
            # Превышен лимит времени
            print("\n=================================================================================================")
            print("❌ ПОСАДКА НЕ УСПЕШНА: Превышен лимит времени.")
            print(f"Финальная позиция (X, Y, Z): ({drone.pos[0]:.3f}, {drone.pos[1]:.3f}, {drone.pos[2]:.3f})")
            print(f"Финальная скорость: {drone.speed:.1f} м/с")
            print("=================================================================================================")
            return False

        t += SIM_DT


if __name__ == "__main__":
    for _ in range(10):
        run_full_simulation()