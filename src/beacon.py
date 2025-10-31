import numpy as np
import matplotlib.pyplot as plt
import time as time
class Beacon:
    
    def __init__(self):
        self.pos = np.array([0.0, 0.0, 0.0])
        self.p = 0.0
        self.n = 8
        self.a = 60
        self.t = 0.8 / 1000
        self.base = 0.0
        
        self._step = 360.0 / self.n
        self._half = self.a / 2.0

    def directional_gain(self, delta_deg):
        x = np.abs(np.asarray(delta_deg, dtype=float))
        return np.where(x >= self._half, 0.0, 1.0 - x / self._half)
    
    def enable_sector_by_id(self, i):
        i = int(i) % self.n
        c = i * self._step
        a = (c - self._half) % 360.0
        b = (c + self._half) % 360.0
        info = {"sector_id": i, "center_deg": c, "bounds_deg": (a, b)}

        # "включение" на время t
        print(f"[ON] sector {i} → {a:.1f}°–{b:.1f}°  (duration {self.t*1000:.1f} ms)")
        time.sleep(self.t)
        print(f"[OFF] sector {i}")
        return info
    
    def rotation(self):
        i = 0
        while True:
            self.enable_sector_by_id(i)
            time.sleep(self.t)
            i = (i + 1) % self.n
            
    def signal_power_at(self, t, point_xyz):
        s = self.active_sector(t)

        # Вектор от маяка к точке (мы как симуляция — знаем)
        dx, dy, dz = np.array(point_xyz) - self.pos
        r = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9  # чтобы не делить на ноль

        # Направление точки относительно оси антенны
        ang = np.degrees(np.arctan2(dy, dx))
        phi = s * self.beam_deg
        diff = (ang - phi + 180) % 360 - 180

        # Коэффициент усиления (0..1)
        gain = self.directional_gain(diff)

        # Простейшая модель затухания
        #   RSSI = P_tx - 20*log10(r) + G(θ)
        rssi = self.p - 20*np.log10(r) + gain*10
        return s, rssi

if __name__ == "__main__":
    b = Beacon()

    print("enable_sector_by_id:")
    for i in range(8):
        print(i, "→", b.enable_sector_by_id(i))

    
    # Пример gain: отклонения от центра сектора
    deltas = np.array([-40, -30, -15, 0, 15, 30, 40], dtype=float)
    print("\ndirectional_gain on deltas:", deltas, "→", b.directional_gain(deltas))
    
    print("\nrotation(θ):")
    
    print(b.rotation())
    
