import numpy as np
import matplotlib.pyplot as plt
import time as time
class Beacon:
    
    def __init__(self):
        self.pos = np.array([0.0, 0.0, 0.0])
        self.p = 0.0
        self.n = 8
        self.ang = 60
        self.t = 0.8 / 1000
        self.base = 0.0
        
        self._step = 360.0 / self.n
        self._half = self.ang / 2.0

    def directional_gain(self, delta_deg):
        x = np.abs(np.asarray(delta_deg, dtype=float))
        return np.where(x >= self._half, 0.0, 1.0 - x / self._half)

    def signal_power_at(self, sector_id, point_xyz):
        s = int(sector_id) % self.n
        dx, dy, dz = np.array(point_xyz, dtype=float) - self.pos
        r = float(np.sqrt(dx*dx + dy*dy + dz*dz)) + 1e-12

        ang  = np.degrees(np.arctan2(dy, dx))
        phi  = (self.base + s * self._step)
        diff = ((ang - phi + 180.0) % 360.0) - 180.0

        g_lin = float(self.directional_gain(diff))    # 0..1
        if g_lin == 0.0:
            return s, 0.0

        Ptx_lin = 10.0**(self.p/10.0)
        rx_lin  = Ptx_lin * g_lin / (r*r)
        return s, rx_lin


if __name__ == "__main__":
    b = Beacon()

    # Точки-дроны для демонстрации
    points = {
        "A": np.array([20.0, 7.0, 0.0]),
        "B": np.array([-20.0, 0.0, 0.0]),
    }
    i=0
    while True: 
        time.sleep(1) 
        print(f"Мощность сигнала в точке {points['A']}", b.signal_power_at(i, points['A'])) 
        print(f"Мощность сигнала в точке {points['B']}", b.signal_power_at(i, points['B'])) 
        i+=1 
        if i == 8: 
            break


    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location('E')      # 0° = +X (вправо)
    ax.set_theta_direction(-1)           # CCW = +
    ax.set_thetagrids(np.arange(0,360,45))
    r_max = 40.0
    ax.set_rmax(r_max)

    theta = np.linspace(-np.pi, np.pi, 720)

    i = 0
    while True:
        ax.clear()
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.arange(0,360,45))
        ax.set_rmax(r_max)
        ax.set_title(f"Активный сектор {i}", va='bottom')

        # текущий сектор — центр и границы
        phi_deg  = (b.base + i*b._step)
        phi      = np.deg2rad(phi_deg)
        half     = np.deg2rad(b._half)

        # рисуем треугольный лепесток
        d_angles = (np.degrees(theta) - phi_deg + 180.0) % 360.0 - 180.0
        gains    = b.directional_gain(d_angles)
        ax.plot(theta, gains*r_max, linewidth=2)
        ax.fill_between(theta, 0, gains*r_max, alpha=0.3)

        # границы сектора
        ax.plot([phi-half, phi-half], [0, r_max], ls='--', lw=0.8)
        ax.plot([phi+half, phi+half], [0, r_max], ls='--', lw=0.8)

        # --- точки ---
        for name, pt in points.items():
            dx, dy = pt[0]-b.pos[0], pt[1]-b.pos[1]
            r = float(np.hypot(dx, dy))
            ang_rad = np.arctan2(dy, dx)   # [-π, π]
            ang_deg = np.degrees(ang_rad)
            # отклонение от центра активного сектора
            delta = ((ang_deg - phi_deg + 180.0) % 360.0) - 180.0
            gain_lin = float(b.directional_gain(delta))

            # цвет точки: внутри лепестка — зелёный, вне — серый
            color = 'tab:green' if gain_lin > 0.0 else 'tab:gray'

            # текущая мощность (линейная модель, как в методе)
            _, rx = b.signal_power_at(i, pt)

            ax.scatter([ang_rad], [min(r, r_max)], s=50, marker='o', color=color)
            ax.text(ang_rad, min(r, r_max)*1.02,
                    f"{name}\nrx={rx:.3g}", ha='center', va='bottom')

        plt.pause(0.6)
        i = (i + 1) % b.n
