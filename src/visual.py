import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches  
from matplotlib.animation import FuncAnimation
# Импорт из classes.py
from classes import Drone, Beacon, Environment, axis_unit, WIDTH, N, STEP 

# --- КОНСТАНТЫ ВИЗУАЛИЗАЦИИ ---
FRAME_RATE = 100  # Количество кадров анимации
TIME_STEP = 0.1  # Шаг симуляции dt (сек)
# Длительность цикла TDM: 8 секторов * 10 шагов на сектор = 80 шагов
SECTORS_PER_FRAME = 10 

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def get_sector_patch(beacon_pos, sector_id, color):
    """Создает сегмент (конус) для визуализации активного сектора."""
    center_yaw = sector_id * STEP
    
    start_angle = math.degrees(center_yaw - WIDTH / 2.0)
    end_angle = math.degrees(center_yaw + WIDTH / 2.0)
    
    return patches.Wedge( 
        (beacon_pos[0], beacon_pos[1]), 
        r=1000, 
        theta1=start_angle, 
        theta2=end_angle, 
        alpha=0.2, 
        color=color, 
        edgecolor='none'
    )

def setup_scene(beacon):
    """Инициализация сцены Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Визуализация TDM Маяка и Траектории Дрона (Вид Сверху)")
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)

    # 1. Позиция Маяка
    ax.plot(beacon.pos[0], beacon.pos[1], 'o', color='red', markersize=10, label='Маяк (0, 0)')

    # 2. Инициализация активного сектора
    sector_patch = get_sector_patch(beacon.pos, 0, 'green')
    ax.add_patch(sector_patch)
    
    # 3. Инициализация траектории дрона
    line, = ax.plot([], [], 'b--', linewidth=1, label='Траектория Дрона')
    point, = ax.plot([], [], 's', color='blue', markersize=8, label='Дрон')
    
    ax.legend(loc='upper right')
    
    return fig, ax, sector_patch, line, point

def animate(frame, beacon, drone, env, sector_patch, line, point, trajectory_data):
    """
    Основная функция анимации, вызывается для каждого кадра.
    """
    # --- 1. ЛОГИКА TDM МАЯКА ---
    sector_id = (frame // SECTORS_PER_FRAME) % N
    beacon.set_sector(sector_id)

    # Обновляем визуализацию активного сектора
    if sector_patch in plt.gca().patches:
        sector_patch.remove() 
        
    new_patch = get_sector_patch(beacon.pos, sector_id, 'green')
    # Это важно: мы должны вернуть новый патч в конце функции, поэтому сохраняем его
    # в локальной переменной, чтобы он был доступен в следующем кадре.
    # В FuncAnimation это достигается через список возвращаемых объектов.
    sector_patch = new_patch
    plt.gca().add_patch(sector_patch)

    # --- 2. ЛОГИКА ДРОНА (Случайное движение) ---
    
    # Имитация случайного управления:
    if frame % 50 == 0:
        drone.change_yaw(np.random.uniform(-math.radians(5), math.radians(5)))
    
    if frame % 20 == 0:
        drone.change_speed(np.random.uniform(-5.0, 5.0))
        drone.change_altitude_rate(np.random.uniform(-math.radians(0.5), math.radians(0.5)))
    
    drone._update_kinematics(TIME_STEP)

    # --- 3. ИЗМЕРЕНИЕ СИГНАЛА ---
    flux_data = drone.measure_flux(env)
    
    # --- 4. ОБНОВЛЕНИЕ ТРАЕКТОРИИ ---
    trajectory_data.append(drone.pos.copy())
    
    x_data = [p[0] for p in trajectory_data]
    y_data = [p[1] for p in trajectory_data]

    # Обновление графических объектов
    line.set_data(x_data, y_data)
    
    # ИСПРАВЛЕНИЕ ОШИБКИ: Передаем координаты как последовательности [x], [y]
    point.set_data([drone.pos[0]], [drone.pos[1]]) 
    
    # Выводим текущее состояние в консоль для отладки
    if frame % 50 == 0:
        print(f"Кадр {frame:4d}: Сектор {sector_id}, Мощность {flux_data[0]:.2f} Вт/м², Позиция {drone.pos[:2]}")
    
    # Возвращаем обновленные объекты
    return [sector_patch, line, point]

# --- ИСПОЛНЕНИЕ ---

if __name__ == "__main__":
    # Инициализация объектов
    beacon = Beacon(x=0.0, y=0.0, z=0.0) 
    drone = Drone(x=400.0, y=-300.0, z=500.0) 
    env = Environment([beacon]) 

    # Инициализация траектории
    trajectory_data = [drone.pos.copy()]

    # Подготовка сцены
    fig, ax, sector_patch, line, point = setup_scene(beacon)
    
    # Запуск анимации
    # Увеличим FRAME_RATE, чтобы получить более длинную траекторию
    anim = FuncAnimation(
        fig, 
        lambda frame: animate(frame, beacon, drone, env, sector_patch, line, point, trajectory_data), 
        frames=500, # Увеличиваем количество кадров
        interval=TIME_STEP * 1000, 
        blit=False, 
        repeat=False
    )

    plt.show()