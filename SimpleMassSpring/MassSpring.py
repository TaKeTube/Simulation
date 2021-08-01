import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)

pixels = 400
dt = 1e-3
substeps = 10

x = ti.field(dtype=float, shape=3)
v = ti.field(dtype=float, shape=3)
m = ti.field(dtype=float, shape=())
k = ti.field(dtype=float, shape=())

@ti.func
def forward_euler():
    new_v = v[0] - dt*k[None]*x[0]/m[None]
    new_x = x[0] + dt*v[0]
    v[0] = new_v
    x[0] = new_x

@ti.func
def symplectic_euler():
    new_v = v[1] - dt*k[None]*x[1]/m[None]
    new_x = x[1] + dt*new_v
    v[1] = new_v
    x[1] = new_x

@ti.func
def backward_euler():
    dv = -dt*k[None]*(x[2]+dt*v[2])/(m[None]+dt**2*k[None])
    new_v = v[2] + dv
    new_x = x[2] + dt*(v[2]+dv)
    v[2] = new_v
    x[2] = new_x

@ti.kernel
def substep():
    forward_euler()
    symplectic_euler()
    backward_euler()

def main():
    # drawing stuff
    simulation_window = ti.GUI("Simple 1-D Mass Spring", res=(pixels,pixels), background_color=0xDDDDDD)
    draw_orig = np.array([[0.25, 0.5],[0.5,0.5],[0.75,0.5]])
    draw_pos = draw_orig.copy()

    k[None] = 5000
    m[None] = 20
    
    for method in range(3):
        x[method] = 100
        v[method] = 0

    for i in range(1000000):
        for step in range(substeps):
            substep()

        for method in range(3):
            draw_pos[method][1] = (pixels/2 + x[method])/pixels
            simulation_window.line(begin = draw_orig[method], end = draw_pos[method], radius = 2, color = 0x444444)
            simulation_window.circle(pos = draw_pos[method], color = 0x111111, radius = 5)

        simulation_window.show()

if __name__ == '__main__':
    main()