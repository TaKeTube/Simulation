import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

scale = 40
l = 2
m = 1
G = 100
M = 100
C = -G*M*m/2*l

pixels = 400
dt = 1e-4
substeps = 50

x = ti.field(dtype=float, shape=6)
y = ti.field(dtype=float, shape=6)
temp = ti.field(dtype=float, shape=6)
k1 = ti.field(dtype=float, shape=6)
k2 = ti.field(dtype=float, shape=6)
k3 = ti.field(dtype=float, shape=6)
k4 = ti.field(dtype=float, shape=6)

# r: x[0]
# phi: x[1]
# theta: x[2]
# \dot r: x[3]
# \dot phi: x[4]
# \dot theta: x[5]

@ti.func
def f_x(t: float, i: int):
    ret = 0.0
    if i < 3:
        ret = x[i+3]
    elif i == 3:
        dUdr = C*(-l)/x[0]*(
            1/ti.sqrt(x[0]**2-2*x[0]*l*ti.cos(x[1])+l**2) +
            1/ti.sqrt(x[0]**2+2*x[0]*l*ti.cos(x[1])+l**2)
            )
        ret = x[0]*x[5]**2 - dUdr/m
    elif i == 4:
        dUdphi = C/ti.sin(x[1])*(
            (x[0]-l*ti.cos(x[1]))/ti.sqrt(x[0]**2-2*x[0]*l*ti.cos(x[1])+l**2) -
            (x[0]+l*ti.cos(x[1]))/ti.sqrt(x[0]**2+2*x[0]*l*ti.cos(x[1])+l**2)
            )
        ret = -(3/(m*l**2)+1/(m*x[0]**2))*dUdphi + 2*x[3]*x[5]/x[0]
    elif i == 5:
        dUdphi = C/ti.sin(x[1])*(
            (x[0]-l*ti.cos(x[1]))/ti.sqrt(x[0]**2-2*x[0]*l*ti.cos(x[1])+l**2) -
            (x[0]+l*ti.cos(x[1]))/ti.sqrt(x[0]**2+2*x[0]*l*ti.cos(x[1])+l**2)
            )
        ret = dUdphi/(m*x[0]**2) - 2*x[3]*x[5]/x[0]
    return ret

@ti.kernel
def RK4_x(t: float):
    for i in range(6):
        temp[i] = x[i]
    for i in range(6):
        k1[i] = f_x(t, i)
    for i in range(6):
        x[i] = temp[i] + 0.5 * k1[i] * dt
    for i in range(6):
        k2[i] = f_x(t + dt / 2., i)
    for i in range(6):
        x[i] = temp[i] + 0.5 * k2[i] * dt
    for i in range(6):
        k3[i] = f_x(t + dt / 2., i)
    for i in range(6):
        x[i] = temp[i] + k3[i] * dt
    for i in range(6):
        k4[i] = f_x(t + dt, i)
    for i in range(6):
        x[i] += 1.0 / 6.0 * dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
    x[1] = x[1] if x[1] < 2*np.pi else x[1] - 2*np.pi
    x[2] = x[2] if x[2] < 2*np.pi else x[2] - 2*np.pi

@ti.kernel
def forward_euler_x(t: float):
    for i in range(6):
        k1[i] = f_x(t, i)
    for i in range(6):
        x[i] += dt * k1[i]

@ti.func
def f_y(t: float, i: int):
    ret = 0.0
    if i < 3:
        ret = y[i+3]
    elif i == 3:
        dUdr = C*(-l)/y[0]*(
            1/ti.sqrt(y[0]**2-2*y[0]*l*ti.cos(y[1])+l**2) +
            1/ti.sqrt(y[0]**2+2*y[0]*l*ti.cos(y[1])+l**2)
            )
        ret = y[0]*y[5]**2 - dUdr/m
    elif i == 4:
        dUdphi = C/ti.sin(y[1])*(
            (y[0]-l*ti.cos(y[1]))/ti.sqrt(y[0]**2-2*y[0]*l*ti.cos(y[1])+l**2) -
            (y[0]+l*ti.cos(y[1]))/ti.sqrt(y[0]**2+2*y[0]*l*ti.cos(y[1])+l**2)
            )
        ret = -(3/(m*l**2)+1/(m*y[0]**2))*dUdphi + 2*y[3]*y[5]/y[0]
    elif i == 5:
        dUdphi = C/ti.sin(y[1])*(
            (y[0]-l*ti.cos(y[1]))/ti.sqrt(y[0]**2-2*y[0]*l*ti.cos(y[1])+l**2) -
            (y[0]+l*ti.cos(y[1]))/ti.sqrt(y[0]**2+2*y[0]*l*ti.cos(y[1])+l**2)
            )
        ret = dUdphi/(m*y[0]**2) - 2*y[3]*y[5]/y[0]
    return ret

@ti.kernel
def RK4_y(t: float):
    for i in range(6):
        temp[i] = y[i]
    for i in range(6):
        k1[i] = f_y(t, i)
    for i in range(6):
        y[i] = temp[i] + 0.5 * k1[i] * dt
    for i in range(6):
        k2[i] = f_y(t + dt / 2., i)
    for i in range(6):
        y[i] = temp[i] + 0.5 * k2[i] * dt
    for i in range(6):
        k3[i] = f_y(t + dt / 2., i)
    for i in range(6):
        y[i] = temp[i] + k3[i] * dt
    for i in range(6):
        k4[i] = f_y(t + dt, i)
    for i in range(6):
        y[i] += 1.0 / 6.0 * dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
    y[1] = y[1] if y[1] < 2*np.pi else y[1] - 2*np.pi
    y[2] = y[2] if y[2] < 2*np.pi else y[2] - 2*np.pi

@ti.kernel
def forward_euler_y(t: float):
    for i in range(6):
        k1[i] = f_y(t, i)
    for i in range(6):
        y[i] += dt * k1[i]

@ti.kernel
def init():
    x[0] = 6
    x[1] = np.pi/3
    x[2] = np.pi/3
    x[3] = 0
    x[4] = 0
    # for forward Euler
    # x[5] = -15
    # for RK4
    # x[5] = -15.1
    x[5] = 17

    y[0] = 6
    y[1] = np.pi/3
    y[2] = np.pi/3
    y[3] = 0
    y[4] = 0
    y[5] = 17.001
    # y[5] = 17
    

def substep():
    # forward_euler_x(0)
    RK4_x(0)
    RK4_y(0)


def get_ends(r, phi, theta):
    begin = np.array([-l*np.cos(phi+theta)+r*np.cos(theta), -l*np.sin(phi+theta)+r*np.sin(theta)])/scale + 0.5
    end = np.array([l*np.cos(phi+theta)+r*np.cos(theta), l*np.sin(phi+theta)+r*np.sin(theta)])/scale + 0.5
    return begin, end

def main():
    gui = ti.GUI('Simulation', res=(pixels, pixels))

    init()

    centers_x = []
    centers_y = []

    for i in range(1000000):
        for step in range(substeps):
            substep()
        gui.circle(pos=[0.5,0.5], radius=5, color=0xFFFFFF)
        # Draw rod x
        begin_x, end_x = get_ends(x[0], x[1], x[2])
        centers_x.append((begin_x+end_x)/2)
        for c in centers_x:
            gui.circle(pos=c, radius=1, color=0x46b5d7)
        gui.line(begin_x, end_x, radius=2, color=0x46b5d7)

        # Draw rod y
        begin_y, end_y = get_ends(y[0], y[1], y[2])
        centers_y.append((begin_y+end_y)/2)
        for c in centers_y:
            gui.circle(pos=c, radius=1, color=0xffa946)
        gui.line(begin_y, end_y, radius=2, color=0xffa946)
    
        gui.show()

if __name__ == '__main__':
    main()
