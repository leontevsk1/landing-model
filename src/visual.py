from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Wedge, Circle

# Подтягиваем параметры системы
import classes as cl


# ------------------------------ ДАННЫЕ КАДРА ------------------------------

@dataclass
class FrameState:
    t: float
    drone_pos: np.ndarray         # [x, y, z]
    drone_dir: np.ndarray         # [dx, dy, dz] (единичный)
    drone_speed: float
    beacon_pos: np.ndarray        # [0,0,0]
    beacon_sector: int            # активный сектор в момент кадра
    tdm_timer: float              # таймер TDM внутри периода [0..TDM_SWITCH_RATE)
    power_last: float             # последняя измеренная мощность (для справки)


# ------------------------------ ВИЗУАЛИЗАТОР ------------------------------

@dataclass
class Visualizer:
    success_radius_m: float = 2.0
    out_dir: str = "/home/artem/dev/bpla/export_data"
    out_name: str = "landing"
    fmt: str = "mp4"          # "mp4" | "gif"
    fps: int = 24
    dpi: int = 120
    world_pad: float = 20.0   # отступ к авто-лимитам
    trail_alpha: float = 0.9
    trail_width: float = 1.5
    _frames: List[FrameState] = field(default_factory=list)

    def capture(self, beacon: cl.Beacon, drone: cl.Drone, env: cl.Environment, t: float):
        """Снимок состояния для кадра анимации."""
        self._frames.append(
            FrameState(
                t=float(t),
                drone_pos=drone.pos.copy(),
                drone_dir=drone.direction.copy(),
                drone_speed=float(drone.speed),
                beacon_pos=env.beacon.pos.copy(),
                beacon_sector=int(env.beacon.sector_id),
                tdm_timer=float(getattr(env.beacon, "_tdm_timer", 0.0)),
                power_last=float(getattr(env, "last_power_for_debug", 0.0))  # не обязательно
            )
        )

    # -------------------------- ПОМОЩНИКИ РИСОВАНИЯ --------------------------

    def _compute_limits(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        xs = [f.drone_pos[0] for f in self._frames] + [self._frames[0].beacon_pos[0]]
        ys = [f.drone_pos[1] for f in self._frames] + [self._frames[0].beacon_pos[1]]
        zs = [f.drone_pos[2] for f in self._frames] + [self._frames[0].beacon_pos[2]]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(0.0, min(zs)), max(zs)

        # симметричный квадрат для XY, чтобы полярная сетка не выглядела сжатой
        r = max(x_max - x_min, y_max - y_min)
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        x_lim = (cx - 0.5 * r - self.world_pad, cx + 0.5 * r + self.world_pad)
        y_lim = (cy - 0.5 * r - self.world_pad, cy + 0.5 * r + self.world_pad)

        # для XZ берём X из XY и Z с отступом
        z_lim = (z_min - 5.0, z_max + 5.0)
        return x_lim, y_lim, z_lim

    def _draw_axes_xy(self, ax, x_lim, y_lim):
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_aspect("equal")
        ax.set_xlabel("X, м")
        ax.set_ylabel("Y, м")
        ax.set_title("Вид сверху (XY) • полярная сетка, секторная диаграмма")
        ax.grid(True, alpha=0.15)

        # Полярная поверх декартовой
        cx = 0.5 * (x_lim[0] + x_lim[1])
        cy = 0.5 * (y_lim[0] + y_lim[1])
        r_max = 0.5 * max(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0])

        # окружности
        rings = 6
        for i in range(1, rings + 1):
            rad = r_max * i / rings
            circ = plt.Circle((0.0, 0.0), rad, fill=False, ls=":", lw=0.8, alpha=0.35)
            ax.add_patch(circ)

        # лучи
        spokes = 12
        for k in range(spokes):
            ang = 2 * math.pi * k / spokes
            ax.plot([0.0, r_max * math.cos(ang)], [0.0, r_max * math.sin(ang)], ls=":", lw=0.8, alpha=0.35)

        # Целевой радиус успеха
        target = plt.Circle((0.0, 0.0), self.success_radius_m, fill=False, ec="tab:green", lw=1.2, alpha=0.9)
        ax.add_patch(target)

    def _draw_axes_xz(self, ax, x_lim, z_lim):
        ax.set_xlim(*x_lim)
        ax.set_ylim(*z_lim)
        ax.set_aspect("auto")
        ax.set_xlabel("X, м")
        ax.set_ylabel("Z, м")
        ax.set_title("Вид сбоку (XZ) • высота и угол к земле")
        ax.grid(True, alpha=0.15)
        # линия земли
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.5)

    def _make_sector_patches(self) -> List[Wedge]:
        """Создаём контуры секторов для вида сверху (без добавления в Axes)."""
        patches = []
        # Радиус косметический: берём не бесконечность, а долю от охвата сцены.
        # Фактический радиус будет обновляться при отрисовке.
        base_radius = 1.0
        for i in range(cl.N):
            start_deg = math.degrees(i * cl.STEP - cl.HALF)
            end_deg = math.degrees(i * cl.STEP + cl.HALF)
            w = Wedge(center=(0.0, 0.0), r=base_radius, theta1=start_deg, theta2=end_deg,
                      width=None, fill=False, ec="tab:gray", lw=1.0, alpha=0.7)
            patches.append(w)
        return patches

    def _update_sector_patches(self, patches: List[Wedge], ax_xy, active_idx: int,
                               show_active: bool, radius: float):
        """Обновить вид секторов под текущий радиус сцены.
        Активный сектор подсвечиваем мягким зелёным, остальные — контуром.
        Если show_active == False, не подсвечиваем ни один (пауза OFF)."""
        for i, w in enumerate(patches):
            w.set_radius(radius)
            w.set_center((0.0, 0.0))
            w.set_fill(False)
            w.set_linewidth(1.0)
            w.set_edgecolor("tab:gray")
            w.set_alpha(0.7)
            if w.axes is None:
                ax_xy.add_patch(w)

        if show_active and 0 <= active_idx < len(patches):
            # Сверху добавим «залитую» копию активного сектора
            active = patches[active_idx]
            w_fill = Wedge(center=(0.0, 0.0),
                           r=radius,
                           theta1=active.theta1,
                           theta2=active.theta2,
                           width=None,
                           fill=True,
                           fc=(0.0, 0.9, 0.0, 0.15),  # мягкий зелёный
                           ec="none")
            ax_xy.add_patch(w_fill)
            # чтобы не плодить их бесконечно, вернём ссылку для удаления в конце кадра
            return [w_fill]
        return []

    @staticmethod
    def _descent_angle_deg(dir_vec: np.ndarray) -> float:
        """Угол к земле (горизонтали), градусы. Положительный при снижении."""
        horiz = float(np.linalg.norm(dir_vec[:2]) + 1e-12)
        angle = math.degrees(math.atan2(max(0.0, -dir_vec[2]), horiz))
        return angle

    # ------------------------------ ОСНОВНОЙ РЕНДЕР ------------------------------

    def plot_all(self, beacon_xy: Tuple[float, float] = (0.0, 0.0)) -> Path:
        if not self._frames:
            raise RuntimeError("Нет данных для визуализации. Вызови vis.capture(...) внутри петли симуляции.")

        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(self.out_dir) / f"{self.out_name}.{self.fmt.lower()}"

        x_lim, y_lim, z_lim = self._compute_limits()

        fig = plt.figure(figsize=(12, 6), dpi=self.dpi)
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25)
        ax_xy = fig.add_subplot(gs[0, 0])
        ax_xz = fig.add_subplot(gs[0, 1])

        self._draw_axes_xy(ax_xy, x_lim, y_lim)
        self._draw_axes_xz(ax_xz, x_lim, z_lim)

        # маяк
        beacon_dot_xy, = ax_xy.plot([0.0], [0.0], marker="*", ms=10, color="tab:orange", label="Маяк")
        beacon_dot_xz, = ax_xz.plot([0.0], [0.0], marker="*", ms=10, color="tab:orange")

        # траектория
        trail_xy, = ax_xy.plot([], [], lw=self.trail_width, alpha=self.trail_alpha, label="Траектория")
        proj_xy, = ax_xy.plot([], [], lw=1.0, ls="--", alpha=0.5, label="Проекция Z на XY")
        trail_xz, = ax_xz.plot([], [], lw=self.trail_width, alpha=self.trail_alpha)

        # дрон
        drone_xy, = ax_xy.plot([], [], marker="o", ms=6, color="tab:blue", label="Дрон")
        drone_xz, = ax_xz.plot([], [], marker="o", ms=6, color="tab:blue")

        # скорость и угол
        speed_text = ax_xy.text(0.02, 0.98, "", transform=ax_xy.transAxes,
                                va="top", ha="left", fontsize=10,
                                bbox=dict(boxstyle="round", fc=(1,1,1,0.6), ec="none"))
        angle_text = ax_xz.text(0.02, 0.98, "", transform=ax_xz.transAxes,
                                va="top", ha="left", fontsize=10,
                                bbox=dict(boxstyle="round", fc=(1,1,1,0.6), ec="none"))

        # Сектора: статические контуры + динамическая подсветка активного
        sector_patches = self._make_sector_patches()

        # Легенда
        ax_xy.legend(loc="lower right", fontsize=9)

        # Предрасчёт траектории
        xs = np.array([f.drone_pos[0] for f in self._frames])
        ys = np.array([f.drone_pos[1] for f in self._frames])
        zs = np.array([f.drone_pos[2] for f in self._frames])
        # Для XZ возьмём X по оси X
        xz_x = xs
        xz_z = zs

        # Радиус для клиппинга лепестков относительно сцены
        radius_for_sectors = 0.48 * max(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0])

        # Функции анимации
        removable_fills: List[matplotlib.patches.Patch] = []

        def init():
            trail_xy.set_data([], [])
            proj_xy.set_data([], [])
            trail_xz.set_data([], [])
            drone_xy.set_data([], [])
            drone_xz.set_data([], [])
            speed_text.set_text("")
            angle_text.set_text("")
            # первичная инициализация контуров секторов
            self._update_sector_patches(sector_patches, ax_xy, active_idx=-1, show_active=False, radius=radius_for_sectors)
            return (trail_xy, proj_xy, trail_xz, drone_xy, drone_xz, speed_text, angle_text, *sector_patches)

        def update(frame_idx: int):
            # очистить заливки с прошлого кадра
            nonlocal removable_fills
            for p in removable_fills:
                p.remove()
            removable_fills = []

            f = self._frames[frame_idx]

            # траектория до текущего кадра
            trail_xy.set_data(xs[:frame_idx+1], ys[:frame_idx+1])
            # проекция снижения: рисуем пунктиром от текущей точки до земли в XY через короткий вектор
            proj_xy.set_data([xs[frame_idx], xs[frame_idx]], [ys[frame_idx], ys[frame_idx]])

            trail_xz.set_data(xz_x[:frame_idx+1], xz_z[:frame_idx+1])

            # дрон
            drone_xy.set_data([f.drone_pos[0]], [f.drone_pos[1]])
            drone_xz.set_data([f.drone_pos[0]], [f.drone_pos[2]])

            # скорость и угол к земле
            angle_deg = self._descent_angle_deg(f.drone_dir)
            speed_text.set_text(f"t = {f.t:6.2f} с\nV = {f.drone_speed:5.2f} м/с")
            angle_text.set_text(f"θ к земле = {angle_deg:5.2f}°\nZ = {f.drone_pos[2]:.2f} м")

            # TDM: off-фаза
            frac = (f.tdm_timer % cl.TDM_SWITCH_RATE) / cl.TDM_SWITCH_RATE if cl.TDM_SWITCH_RATE > 0 else 0.0
            is_on = frac >= cl.TDM_OFF_FRACTION  # сначала короткая «тишина», затем сектор включён

            # обновляем контуры и зелёную заливку активного сектора
            removable_fills = self._update_sector_patches(
                sector_patches, ax_xy, active_idx=f.beacon_sector, show_active=is_on, radius=radius_for_sectors
            )

            return (trail_xy, proj_xy, trail_xz, drone_xy, drone_xz, speed_text, angle_text, *sector_patches, *removable_fills)

        ani = FuncAnimation(fig, update, frames=len(self._frames), init_func=init,
                            blit=False, interval=1000.0 / self.fps)

        # Сохранение
        if self.fmt.lower() == "mp4":
            try:
                writer = FFMpegWriter(fps=self.fps, bitrate=2400)
                ani.save(out_path.as_posix(), writer=writer, dpi=self.dpi)
            except Exception as e:
                # если ffmpeg не доступен — fallback на gif
                fallback = out_path.with_suffix(".gif")
                writer = PillowWriter(fps=self.fps)
                ani.save(fallback.as_posix(), writer=writer, dpi=self.dpi)
                plt.close(fig)
                return fallback
        else:
            writer = PillowWriter(fps=self.fps)
            ani.save(out_path.as_posix(), writer=writer, dpi=self.dpi)

        plt.close(fig)
        return out_path