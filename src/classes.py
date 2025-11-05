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
        """Переключение активного сектора по таймеру TDM."""
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
        
        # --- ИСПРАВЛЕНИЕ КИНЕМАТИКИ: ОСТАНОВКА НА ЗЕМЛЕ ---
        if self.pos[2] < 0.0:
            self.pos[2] = 0.0      # Клип Z
            self.speed = 0.0       # Остановка
            # Выравниваем направление по горизонту, чтобы не "тыкать носом" в землю
            self.direction[2] = 0.0
            if np.linalg.norm(self.direction[:2]) > self._eps_xy:
                 self.direction /= np.linalg.norm(self.direction)
            else:
                 self.direction = np.array([1.0, 0.0, 0.0]) # Деградация до +X
        # --------------------------------------------------

# -------------------------------- КОНТРОЛЛЕР ---------------------------------
class Controller:
    def __init__(self, drone: Drone, env: Environment):
        self.drone = drone
        self.env = env

        # ---- КУРСОВОЙ (УЛУЧШЕННЫЕ КОНСТАНТЫ) ----
        self.LOCALIZER_P_GAIN = math.radians(8.0)
        self.FLUX_MIN_THRESHOLD = 1e-5
        self.INITIAL_SHIFT_YAW = math.radians(12.0)
        self.P_DIFF_EPS = 1e-6
        self.MAX_CORRECTION_ANGLE = math.radians(6.0)

        # ---- ПОИСК (НОВАЯ ЛОГИКА) ----
        self.SEEK_ENABLED = True
        self.SEEK_YAW_RATE = math.radians(2.0)
        self.SIGNAL_LOST_TIMEOUT = 2.5
        self._time_since_seen = 1e9

        # ---- ГЛИССАДА/СКОРОСТЬ ----
        self.PITCH_TARGET_RAD = math.radians(-5.0)
        self.GLIDE_PITCH_GAIN = 0.12
        self.SPEED_TARGET = 1.0 # Целевая скорость
        self.POWER_FALL_BOOST_DEG = 0.5
        self.POWER_FALL_RATIO = 0.95
        self.FLARE_ALT = 4.0
        self.FLARE_PITCH_TARGET = math.radians(-1.0)
        self.FLARE_BLEND = 0.6

        # ---- ПАМЯТЬ ----
        self.is_landing_activated = False
        self.last_power_flux = 0.0
        self.last_sector_id = -1

    def _current_pitch(self) -> float:
        d = self.drone.direction
        d_norm = np.linalg.norm(d)
        return math.asin(np.clip(d[2] / d_norm, -1.0, 1.0)) if d_norm > 1e-6 else 0.0

    def _compute_yaw_delta(self, power_flux: float, sector_id: int, dt: float) -> float:
        dyaw = 0.0
        seen_now = power_flux > self.FLUX_MIN_THRESHOLD
        
        # 1) Вход в перекрытие
        if self.last_power_flux <= self.FLUX_MIN_THRESHOLD and seen_now:
            # Смещаемся CCW (+yaw) для гарантированного попадания в зону overlap
            return +self.INITIAL_SHIFT_YAW

        # 2) Сравнение двух последовательных измерений
        if seen_now and self.last_power_flux > self.FLUX_MIN_THRESHOLD:
            denom = max(power_flux, self.last_power_flux, self.P_DIFF_EPS)
            diff_norm = (self.last_power_flux - power_flux) / denom
            dyaw = diff_norm * self.LOCALIZER_P_GAIN
            dyaw = float(np.clip(dyaw, -self.MAX_CORRECTION_ANGLE, self.MAX_CORRECTION_ANGLE))
            return dyaw

        # 3) Режим поиска (работает только после активации посадки)
        if self.SEEK_ENABLED and self.is_landing_activated: 
            # Начинаем крутиться, чтобы найти сигнал
            if self._time_since_seen < self.SIGNAL_LOST_TIMEOUT:
                dyaw += self.SEEK_YAW_RATE * dt   # CCW
            else:
                dyaw += 2.0 * self.SEEK_YAW_RATE * dt # Ускоренный поиск

        return dyaw

    def _compute_vertical_delta(self, power_flux: float) -> float:
        pitch_now = self._current_pitch()
        pitch_target = self.PITCH_TARGET_RAD
        
        # Flare у земли
        if self.drone.pos[2] <= self.FLARE_ALT:
            pitch_target = (1.0 - self.FLARE_BLEND) * pitch_target + self.FLARE_BLEND * self.FLARE_PITCH_TARGET
            
        error = pitch_target - pitch_now
        delta_pitch = error * self.GLIDE_PITCH_GAIN
        
        # Эвристика: просел поток - увеличиваем снижение
        if power_flux < self.last_power_flux * self.POWER_FALL_RATIO and self.last_power_flux > self.FLUX_MIN_THRESHOLD:
            delta_pitch -= math.radians(self.POWER_FALL_BOOST_DEG)
            
        return delta_pitch

    def _compute_speed_delta(self, power_flux: float) -> float:
        delta_v = 0.0
        if self.drone.speed > self.SPEED_TARGET:
            delta_v += -(self.drone.speed - self.SPEED_TARGET) * 0.12
        if power_flux > 0.5:
            delta_v += -0.05 * power_flux
        return delta_v

    def run_control_cycle(self, dt: float):
        self.env.update(dt)
        power_flux, sector_id = self.drone.measure_flux(self.env)

        # Обновляем таймер «как давно видели сигнал»
        if power_flux > self.FLUX_MIN_THRESHOLD:
            self._time_since_seen = 0.0
        else:
            self._time_since_seen += dt

        # Активация посадки
        if not self.is_landing_activated and power_flux > self.FLUX_MIN_THRESHOLD:
            self.is_landing_activated = True
            print(f"--- АКТИВАЦИЯ ПОСАДКИ (Поток: {power_flux:.4f}) ---")

        # Управление применяется только после активации
        if self.is_landing_activated:
            dyaw   = self._compute_yaw_delta(power_flux, sector_id, dt)
            dpitch = self._compute_vertical_delta(power_flux)
            dspeed = self._compute_speed_delta(power_flux)

            self.drone.change_yaw(dyaw)
            self.drone.change_altitude_rate(dpitch)
            self.drone.change_speed(dspeed)
        
        # Если посадка не активирована, но надо искать сигнал (допустим, медленный полет по прямой)
        # Добавлять сюда сложнее. Пока оставляем: дрон летит по прямой, пока не увидит сигнал.

        self.drone.integrate(dt)
        self.last_power_flux = power_flux
        self.last_sector_id = sector_id