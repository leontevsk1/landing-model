import numpy as np
import math
import random

# ----------------------------- КОНСТАНТЫ СИСТЕМЫ -----------------------------
N = 8
STEP  = math.pi / 4               # 45° -> π/4
WIDTH = math.radians(60.0)        # ширина лепестка (рад) -> 60°
HALF  = WIDTH / 2.0               # полуширина -> 30°
POWER = 15.0                      # мощность излучения маяка (Вт)

# Симуляция/время
SIM_DT = 0.1                      # шаг симуляции (с)
TDM_SWITCH_RATE = 1.0             # период переключения сектора (с)

# Контроль/безопасность
R_MIN = 0.5                       # минимально учитываемая дистанция до маяка (м)

# ----------------------------- ПАРАМЕТРЫ ПОСАДКИ (эвристики) -------------------------
# Скорости (м/с)
LANDING_SPEED_MPS = 20.0 / 3.6      # 20 км/ч -> 5.55 м/с
LANDING_FINAL_SPEED_MPS = 2.0 / 3.6 # 2 км/ч -> 0.55 м/с

# Высоты (м)
LANDING_FINAL_ALTITUDE = 4.0        # 4 м

# Пороги мощности (Вт/м^2) - ЭТИ ЗНАЧЕНИЯ НУЖНО ПОДБИРАТЬ!
LANDING_DECEL_POWER = 5.0           # Порог для замедления до 20 км/ч
LANDING_SAFE_POWER = 1.0           # Порог "безопасной зоны" для остановки

# Параметры контроллера (M0)
POWER_TOLERANCE = 0.01              # Допуск для P(S0) == P(S1)
STRAFE_DISTANCE_M = 0.5             # (м) Величина "шага" вбок для коррекции
DESCENT_ANGLE_RAD = math.radians(5.0) # Угол снижения (5 градусов)

# ----------------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ -----------------------
def axis_unit(yaw_idx: int, axis_pitch_rad: float) -> np.ndarray:
    """
    Единичный вектор оси активного сектора.
    """
    yaw = (yaw_idx % N) * STEP
    cp, sp = math.cos(axis_pitch_rad), math.sin(axis_pitch_rad)
    return np.array([cp * math.cos(yaw), cp * math.sin(yaw), sp], dtype=float)

def energy_flux(R: float, alpha: float, w: float, P: float) -> float:
    """
    Плотность потока (Вт/м^2).
    """
    if R <= 0.0:
        return 0.0
    G = max(0.0, 1.0 - 4.0 * (alpha ** 2) / (w ** 2))
    return 9.0 * P * G / (4.0 * (R ** 2) * (w ** 2))

# ----------------------------------- МАЯК ------------------------------------
class Beacon:
    def __init__(self, x=0.0, y=0.0, z=0.0, power_w: float = POWER,
                 width_rad: float = WIDTH, axis_pitch_rad: float = HALF):
        self.pos = np.array([x, y, z], dtype=float)
        self.power_w = float(power_w)
        self.width_rad = float(width_rad)
        self.axis_pitch_rad = float(axis_pitch_rad)
        self.sector_id = 0
        self._tdm_timer = 0.0

    def update(self, dt: float):
        """Переключение активного сектора по таймеру."""
        self._tdm_timer += dt
        if self._tdm_timer >= TDM_SWITCH_RATE:
            steps = int(self._tdm_timer // TDM_SWITCH_RATE)
            self._tdm_timer -= steps * TDM_SWITCH_RATE
            self.sector_id = (self.sector_id + steps) % N

    def axis_vector(self) -> np.ndarray:
        return axis_unit(self.sector_id, self.axis_pitch_rad)

# ---------------------------------- СРЕДА ------------------------------------
class Environment:
    def __init__(self, beacon: Beacon):
        self.beacon = beacon

    def update(self, dt: float):
        self.beacon.update(dt)

    def power_lin_at(self, pos: np.ndarray) -> tuple[float, int]:
        b = self.beacon
        r_vec = pos - b.pos
        R = float(np.linalg.norm(r_vec))
        R = max(R, R_MIN)  # клиппинг сингулярности

        v = r_vec / R
        a = b.axis_vector()
        cosd = float(np.clip(np.dot(a, v), -1.0, 1.0))
        delta = math.acos(cosd)
        total = energy_flux(R, delta, b.width_rad, b.power_w)
        return total, b.sector_id

# ----------------------------------- ДРОН ------------------------------------
class Drone:
    def __init__(self, x=0.0, y=0.0, z=0.0, speed: float = 20.0):
        self.pos = np.array([x, y, z], dtype=float)
        self.direction = np.array([1.0, 0.0, 0.0], dtype=float)
        self.speed = float(speed)
        self._eps_xy = 1e-6

    def measure_flux(self, env: Environment) -> tuple[float, int]:
        return env.power_lin_at(self.pos)

    def change_altitude_rate(self, delta_pitch_rad: float):
        """Меняем тангаж, сохраняя азимут."""
        dir_norm = np.linalg.norm(self.direction)
        if dir_norm < self._eps_xy: dir_norm = 1.0; self.direction = np.array([1.0, 0.0, 0.0], dtype=float)

        d = self.direction / dir_norm
        current_pitch = math.asin(np.clip(d[2], -1.0, 1.0))
        new_pitch = current_pitch + float(delta_pitch_rad)
        xy_mag = max(0.0, math.cos(new_pitch))
        z_component = math.sin(new_pitch)

        xy = d[:2]
        xy_norm = float(np.linalg.norm(xy))
        xy_dir = xy / xy_norm if xy_norm > self._eps_xy else np.array([1.0, 0.0], dtype=float)

        self.direction[:2] = xy_dir * xy_mag
        self.direction[2] = z_component
        self.direction /= np.linalg.norm(self.direction)

    def change_yaw(self, delta_yaw_rad: float):
        """Поворот вокруг оси Z."""
        c, s = math.cos(delta_yaw_rad), math.sin(delta_yaw_rad)
        x, y, z = self.direction
        x_new = x * c - y * s
        y_new = x * s + y * c
        self.direction = np.array([x_new, y_new, z], dtype=float)
        self.direction /= np.linalg.norm(self.direction)

    def change_speed(self, delta_v: float):
        self.speed = max(0.0, self.speed + float(delta_v))

    def integrate(self, dt: float):
        """
        Обновление позиции и КРИТИЧЕСКАЯ ПРОВЕРКА ПОСАДКИ.
        """
        self.pos += self.direction * self.speed * dt
        
        # ОСТАНОВКА НА ЗЕМЛЕ
        if self.pos[2] < 0.0:
            self.pos[2] = 0.0
            self.speed = 0.0       # Остановка

            self.direction[2] = 0.0
            if np.linalg.norm(self.direction[:2]) > self._eps_xy:
                 self.direction /= np.linalg.norm(self.direction)
            else:
                 self.direction = np.array([1.0, 0.0, 0.0]) # Деградация до +X
    def point_toward_beacon(self, env: Environment):
        """
        Устанавливает направление дрона прямо на маяк.
        Вычисляет единичный вектор от дрона к маяку и устанавливает его как направление движения.
        """
        beacon_pos = env.beacon.pos
        direction_to_beacon = beacon_pos - self.pos
        
        # Проверка на нулевое расстояние
        distance = np.linalg.norm(direction_to_beacon)
        if distance < self._eps_xy:
            # Если дрон уже у маяка - сохраняем текущее направление
            return
        
        # Нормализуем вектор направления
        self.direction = direction_to_beacon / distance
        
class Controller:
    def __init__(self):
        self.state_timer = 0.0
        self.power_readings = {}
        self.last_total_power = 0.0

    def update(self, env: Environment, drone: Drone, dt: float):
        self.state_timer += dt
        p, s_id = drone.measure_flux(env)
        self.power_readings[s_id] = p
        self.last_total_power = p

        if self.state_timer < TDM_SWITCH_RATE + dt:
            return

        drone.point_toward_beacon(env)

        if self.last_total_power > LANDING_DECEL_POWER:
            if drone.speed > LANDING_SPEED_MPS:
                drone.speed = LANDING_SPEED_MPS
            current_dir_xy = drone.direction[:2]
            xy_norm = np.linalg.norm(current_dir_xy)
            if xy_norm > 1e-6:
                current_dir_xy /= xy_norm
            z_comp = math.sin(-DESCENT_ANGLE_RAD)
            xy_comp = math.cos(-DESCENT_ANGLE_RAD)
            drone.direction[:2] = current_dir_xy * xy_comp
            drone.direction[2] = z_comp
            drone.direction /= np.linalg.norm(drone.direction)

        if drone.pos[2] <= LANDING_FINAL_ALTITUDE and self.last_total_power > LANDING_SAFE_POWER:
            drone.speed = LANDING_FINAL_SPEED_MPS
            drone.direction = np.array([0.0, 0.0, -1.0])
            
        if self.last_total_power > LANDING_SAFE_POWER:
            drone.speed = 0.0

        if drone.speed == 0.0:
            self.power_readings = {}
            self.state_timer = 0.0
            return

        visible_sectors = list(self.power_readings.keys())
        forward_vec = drone.direction.copy()
        up_vec = np.array([0.0, 0.0, 1.0])
        right_vec = np.cross(forward_vec, up_vec)
        right_vec /= np.linalg.norm(right_vec)

        if len(visible_sectors) < 2:
            drone.pos += right_vec * STRAFE_DISTANCE_M
        else:
            s0, s1 = visible_sectors[0], visible_sectors[1]
            p0 = self.power_readings[s0]
            p1 = self.power_readings[s1]
            if abs(p0 - p1) < POWER_TOLERANCE:
                pass
            elif p0 > p1:
                drone.pos += right_vec * STRAFE_DISTANCE_M
            else:
                drone.pos -= right_vec * STRAFE_DISTANCE_M

        self.power_readings = {}
        self.state_timer = 0.0
        

