import numpy as np
import math

# Константы диаграммы (в РАДИАНАХ)
N = 8
STEP  = 2.0 * math.pi / N        # 45° -> π/4
WIDTH = math.radians(60.0)       # ширина лепестка -> π/3
HALF  = WIDTH / 2.0              # 30° -> π/6
POWER = 15.0

def axis_unit(i):
    # единичный вектор оси луча по текущему сектору (yaw+pitch)
    yaw   = WIDTH/2.0 + i * STEP
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
        return total

class Drone:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.pos = np.array([x, y, z], float)
        self.landing_power = 10.0
        self.safe_power = 1.0
        salf.landing_flag = False
    def measure_flux(self, env: Environment) -> float:
        return env.power_lin_at(self.pos)
    
if __name__ == "__main__":
    b = Beacon()
    e = Environment([b])
    d_plusX  = Drone( 10.0, 1.0, 0.0)  # +X
    d_minusX = Drone(-20.0, 0.0, 0.0)  # -X

    b.set_sector(0)    # ось на +X
    print("sector 0:", d_plusX.measure_flux(e), d_minusX.measure_flux(e))   # >0, 0

    b.set_sector(4)    # ось на -X
    print("sector 4:", d_plusX.measure_flux(e), d_minusX.measure_flux(e))   # 0, >0
