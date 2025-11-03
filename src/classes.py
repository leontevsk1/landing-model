import numpy as np
import math

# Константы диаграммы (в РАДИАНАХ)
N = 8
STEP  = math.pi / 4       # 45° -> π/4
WIDTH = math.radians(60.0)       # ширина лепестка -> π/3
HALF  = WIDTH / 2.0              # 30° -> π/6
POWER = 15.0

def axis_unit(i):
    # единичный вектор оси луча по текущему сектору (yaw+pitch)
    yaw   = i * STEP
    pitch = WIDTH/2.0
    return np.array([math.cos(pitch)*math.cos(yaw),
                     math.cos(pitch)*math.sin(yaw),
                     math.sin(pitch)], float)

# Определение плотности потока энергии
def energy_flux(R:float,alpha:float,w:float,P:float ) -> float:
   '''
   Определение плотности потока энергии электромагнитного излучения на расстоянии R (метров) от истоника под углом alpha (радиан).
   w - ширина диаграммы направленности (радиан),
   P - мощность излучения (Вт),
   результат (Вт/м**2)
   
   '''
   return 9*P*max(0,(1-4*alpha**2/w**2))/(4*R**2*w**2)

class Beacon:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.pos = np.array([x, y, z], float)
        self.sector_id = 0  # активный сектор (0..7)
        self.power_w = POWER          # << берём мощность из объекта, не из глобала в расчётах
        self.width_rad = WIDTH        

    def set_sector(self, i):
        self.sector_id = int(i) % N

class Environment:
    """Мир: считает поле(мощность) в точке по источникам."""
    def __init__(self, beacons):
        self.beacons = list(beacons)

    def power_lin_at(self, pos):
        """Суммарная мощность (линейная) в точке (x,y,z) от всех маяков."""
        total = 0.0
        for b in self.beacons:
            r_vec = pos - b.pos
            R = float(np.linalg.norm(r_vec))
            if R == 0.0:
                # Точка совпадает с позицией маяка — вклад не определён, пропускаем
                continue
            v = r_vec / R
            a = axis_unit(b.sector_id)
            cosd = float(np.clip(np.dot(a, v), -1.0, 1.0))
            delta = math.acos(cosd)  # рад
            total += energy_flux(R, delta, b.width_rad, b.power_w)
        return [total,b.sector_id]

# Добавим в класс Drone в classes.py

class Drone:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.pos = np.array([x, y, z], float)
        self.landing_power = 10.0
        self.safe_power = 1.0
        self.landing_flag = False
        self.speed = 55.5
        # Инициализируем direction как единичный вектор (для старта в +X)
        self.direction = np.array([1.0, 0.0, 0.0], float) 
        
    def measure_flux(self, env: Environment): # Убираем -> float, т.к. возвращается список
        # Теперь возвращает список [мощность, ID_сектора]
        return env.power_lin_at(self.pos)
    
    # --- БАЗОВЫЕ ФУНКЦИИ УПРАВЛЕНИЯ ---

    # 1. Снижение (Изменение вертикальной составляющей вектора направления)
    def change_altitude_rate(self, delta_pitch_rad: float):
        """
        Изменяет вертикальный угол (тангаж) вектора направления дрона.
        delta_pitch_rad: изменение угла в радианах (отрицательное - снижение).
        """
        # Преобразуем текущий вектор в сферические координаты (или просто прикладываем матрицу поворота)
        
        # Получаем текущие углы (для упрощения, только наклон, курс оставляем)
        # pitch - угол между вектором и XY-плоскостью.
        current_pitch = math.asin(self.direction[2] / np.linalg.norm(self.direction))
        
        # Новый угол
        new_pitch = current_pitch + delta_pitch_rad
        
        # Ограничиваем угол, чтобы дрон не летел назад
        # max_pitch = math.radians(45) # можно добавить константы
        # new_pitch = np.clip(new_pitch, -max_pitch, max_pitch) 
        
        # Пересчитываем Z-компоненту и нормируем XY-компоненты
        xy_magnitude = math.cos(new_pitch)
        z_component = math.sin(new_pitch)
        
        # Нормируем горизонтальную составляющую
        current_xy_magnitude = np.linalg.norm(self.direction[:2])
        if current_xy_magnitude > 1e-6:
             self.direction[:2] = self.direction[:2] * (xy_magnitude / current_xy_magnitude)
        else:
             # Если дрон летел строго вверх/вниз, то горизонтальное направление (Yaw) сохраняется
             self.direction[:2] = np.array([xy_magnitude, 0.0]) # Примем, что курс идет по оси X
             
        self.direction[2] = z_component
        self.direction /= np.linalg.norm(self.direction) # Гарантируем единичный вектор


    # 2. Изменение скорости
    def change_speed(self, delta_v: float):
        """
        Изменяет скалярную скорость дрона (м/с).
        delta_v: изменение скорости (может быть отрицательным).
        """
        self.speed = max(0.0, self.speed + delta_v) # Скорость не может быть отрицательной


    # 3. Изменение курса
    def change_yaw(self, delta_yaw_rad: float):
        """
        Изменяет горизонтальный курс (рысканье) вектора направления дрона.
        delta_yaw_rad: изменение угла в радианах (положительное - против часовой стрелки).
        """
        # Применяем матрицу поворота вокруг оси Z к горизонтальным компонентам (X, Y)
        cos_d = math.cos(delta_yaw_rad)
        sin_d = math.sin(delta_yaw_rad)
        
        x_new = self.direction[0] * cos_d - self.direction[1] * sin_d
        y_new = self.direction[0] * sin_d + self.direction[1] * cos_d
        
        self.direction[0] = x_new
        self.direction[1] = y_new
        
        # Вектор все еще должен быть единичным (просто повернулся)
        self.direction /= np.linalg.norm(self.direction) 

    # --- ФУНКЦИЯ ПРОДВИЖЕНИЯ В СИМУЛЯЦИИ ---
    
    def _update_kinematics(self, dt: float):
        """Выполняет один шаг симуляции за время dt (секунды)."""
        # V = speed * direction
        velocity_vector = self.speed * self.direction
        
        # P_new = P_old + V * dt
        self.pos += velocity_vector * dt
    
    
    
if __name__ == "__main__":
    b = Beacon()
    b2 = Beacon(1.0,1.0,1.0)
    e = Environment([b,b2])
    d_plusX  = Drone( 10.0, 1.0, 3.0)  # +X
    d_minusX = Drone(-10.0, 1.0, 3.0)  # -X

    b.set_sector(0)    # ось на +X
    print("sector 0:", d_plusX.measure_flux(e), d_minusX.measure_flux(e))   # >0, 0

    b.set_sector(4)    # ось на -X
    print("sector 4:", d_plusX.measure_flux(e), d_minusX.measure_flux(e))   # 0, >0