from classes import Beacon, Drone, Environment
    
if __name__ == "__main__":
    b = Beacon()
    e = Environment([b])
    d_plusX  = Drone( 10.0, 1.0, 0.0)  # +X
    d_minusX = Drone(-20.0, 0.0, 0.0)  # -X

    b.set_sector(0)    # ось на +X
    print("sector 0:", d_plusX.measure_flux(e), d_minusX.measure_flux(e))   # >0, 0

    b.set_sector(4)    # ось на -X
    print("sector 4:", d_plusX.measure_flux(e), d_minusX.measure_flux(e))   # 0, >0
