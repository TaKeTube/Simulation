import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

Using_RK4 = True
Using_NeoHookean = True
pixels = 512
dt = 3e-5
substeps = 20
g = 9.8

# N: #elements per edge 
# E: Young's modulus
# nu: Poisson's ratio
# rho: density of an element
# pos: initial position
# length: length of edge
# angle: initial rotation angle
@ti.data_oriented
class Cube():
    def __init__(self, N, E, nu, rho, pos, length = 0.15, angle = np.pi/8):
        self.N = N
        self.num_elem = 2 * N * N
        self.num_vert = (N + 1) ** 2
        self.pos = pos
        self.angle = angle
        self.length = length
        self.dx = length / N
        self.rho = rho
        self.m = self.rho * self.dx ** 2

        self.E = E
        self.nu = nu
        # Lamé constants
        self.mu = 0.5 * E / (1 + nu)
        self.la = E * nu / ((1 + nu) * (1 - 2 * nu))

        # vertex position
        self.x = ti.Vector.field(2, ti.f32, self.num_vert, needs_grad=True)
        self.x_temp = ti.Vector.field(2, ti.f32, self.num_vert, needs_grad=True)
        # vertex velocity
        self.v = ti.Vector.field(2, ti.f32, self.num_vert)
        self.v_temp = ti.Vector.field(2, ti.f32, self.num_vert)
        # RK4
        self.F1_x = ti.Vector.field(2, ti.f32, self.num_vert)
        self.F2_x = ti.Vector.field(2, ti.f32, self.num_vert)
        self.F3_x = ti.Vector.field(2, ti.f32, self.num_vert)
        self.F4_x = ti.Vector.field(2, ti.f32, self.num_vert)
        self.F1_v = ti.Vector.field(2, ti.f32, self.num_vert)
        self.F2_v = ti.Vector.field(2, ti.f32, self.num_vert)
        self.F3_v = ti.Vector.field(2, ti.f32, self.num_vert)
        self.F4_v = ti.Vector.field(2, ti.f32, self.num_vert)

        # element
        self.e = ti.Vector.field(3, ti.i32, self.num_elem)
        # inverse of edge vectors matrix in material space
        self.Dm_inv = ti.Matrix.field(2, 2, ti.f32, self.num_elem)
        # deformation gradient of element
        self.F = ti.Matrix.field(2, 2, ti.f32, self.num_elem, needs_grad=True)
        # volume of element
        self.V = ti.field(ti.f32, self.num_elem)
        
        # total energy
        self.U = ti.field(ti.f32, (), needs_grad=True)

    @ti.kernel
    def init(self):
        for i, j in ti.ndrange(self.N, self.N):
            idx = (self.N * i + j) * 2
            x1 = i * (self.N + 1) + j
            x2 = x1 + 1
            x3 = x1 + self.N + 1
            x4 = x3 + 1
            self.e[idx] = [x1, x2, x3]
            self.e[idx+1] = [x2, x3, x4]

        rotation = ti.Matrix([[ti.cos(self.angle), -ti.sin(self.angle)], [ti.sin(self.angle), ti.cos(self.angle)]])
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            idx = (self.N + 1) * i + j
            self.x[idx] = (ti.Vector([i, j]) / self.N - ti.Vector([0.5, 0.5])) * self.length + ti.Vector([self.pos[0], self.pos[1]])
            self.x[idx] = rotation @ self.x[idx]
            self.v[idx] = ti.Vector([0, 0])
        for i in range(self.num_elem):
            x1_idx = self.e[i][0]
            x2_idx = self.e[i][1]
            x3_idx = self.e[i][2]
            x1 = self.x[x1_idx]
            x2 = self.x[x2_idx]
            x3 = self.x[x3_idx]
            Dm = ti.Matrix.cols([x1 - x3, x2 - x3])
            self.Dm_inv[i] = Dm.inverse()

    @ti.kernel
    def update_U(self):
        for i in range(self.num_elem):
            x1_idx = self.e[i][0]
            x2_idx = self.e[i][1]
            x3_idx = self.e[i][2]
            x1 = self.x[x1_idx]
            x2 = self.x[x2_idx]
            x3 = self.x[x3_idx]
            self.V[i] = 0.5 * abs((x1 - x3).cross(x2 - x3))
            Dw_i = ti.Matrix.cols([x1 - x3, x2 - x3])
            self.F[i] = Dw_i @ self.Dm_inv[i]
        for i in range(self.num_elem):
            if not Using_NeoHookean:
                C_i = self.F[i].transpose() @ self.F[i]
                E_i = 0.5 * (C_i - ti.Matrix.identity(ti.f32, 2))
                phi_i = self.mu * (E_i @ E_i).trace() + 0.5 * self.la * E_i.trace() ** 2
                # self.U[None] += self.V[i] * phi_i + self.V[i] * self.m * g * self.x[i][1] * 100000
                self.U[None] += self.V[i] * phi_i
            else:
                F_i = self.F[i]
                J_i = F_i.determinant()
                phi_i = self.mu / 2 * ((F_i.transpose() @ F_i).trace() - 2)
                phi_i -= self.mu * ti.log(J_i)
                phi_i += self.la / 2 * (J_i - 1) ** 2
                self.U[None] += self.V[i] * phi_i
    
    @ti.kernel
    def Euler(self):
        for i in range(self.num_vert):
            a = -self.x.grad[i] / self.m
            self.v[i] += dt * (a + ti.Vector([0, -40]))
            self.v[i] *= ti.exp(-dt * 1.25)

        for i in range(self.num_vert):
            if self.x[i][0] < 0 and self.v[i][0] < 0 or self.x[i][0] > 1 and self.v[i][0] > 0:
                self.v[i][0] = 0
            if self.x[i][1] < 0 and self.v[i][1] < 0 or self.x[i][1] > 1 and self.v[i][1] > 0:
                self.v[i][1] = 0
            self.x[i] += dt * self.v[i]

    @ti.kernel
    def RK4_0(self):
        for i in range(self.num_vert):
            self.x_temp[i] = self.x[i]
            self.v_temp[i] = self.v[i]

    @ti.kernel
    def RK4_1(self):
        for i in range(self.num_vert):
            self.F1_x[i] = dt * self.v_temp[i]
            self.F1_v[i] = dt * (-self.x.grad[i] / self.m + ti.Vector([0, -40]))
            self.x[i] = self.x_temp[i] + self.F1_x[i] * 0.5

    @ti.kernel
    def RK4_2(self):
        for i in range(self.num_vert):
            self.F2_x[i] = dt * (self.v_temp[i] + self.F1_v[i] * 0.5)
            self.F2_v[i] = dt * (-self.x.grad[i] / self.m + ti.Vector([0, -40]))
            self.x[i] = self.x_temp[i] + self.F2_x[i] * 0.5

    @ti.kernel
    def RK4_3(self):
        for i in range(self.num_vert):
            self.F3_x[i] = dt * (self.v_temp[i] + self.F2_v[i] * 0.5)
            self.F3_v[i] = dt * (-self.x.grad[i] / self.m + ti.Vector([0, -40]))
            self.x[i] = self.x_temp[i] + self.F3_x[i]

    @ti.kernel
    def RK4_4(self):
        for i in range(self.num_vert):
            self.F4_x[i] = dt * (self.v_temp[i] + self.F3_v[i])
            self.F4_v[i] = dt * (-self.x.grad[i] / self.m + ti.Vector([0, -40]))

        for i in range(self.num_vert):
            self.x[i] = self.x_temp[i] + 1/6 * (self.F1_x[i] + 2 * self.F2_x[i] + 2 * self.F3_x[i] + self.F4_x[i])
            self.v[i] = self.v_temp[i] + 1/6 * (self.F1_v[i] + 2 * self.F2_v[i] + 2 * self.F3_v[i] + self.F4_v[i])
            self.v[i] *= ti.exp(-dt * 1.25)
            if self.x[i][0] < 0 and self.v[i][0] < 0 or self.x[i][0] > 1 and self.v[i][0] > 0:
                self.v[i][0] = 0
            if self.x[i][1] < 0 and self.v[i][1] < 0 or self.x[i][1] > 1 and self.v[i][1] > 0:
                self.v[i][1] = 0

    def substep(self):
        global dt
        if Using_RK4:
            dt *= 3
            self.RK4_0()
            with ti.ad.Tape(loss=self.U):
                self.update_U()
            self.RK4_1()
            with ti.ad.Tape(loss=self.U):
                self.update_U()
            self.RK4_2()
            with ti.ad.Tape(loss=self.U):
                self.update_U()
            self.RK4_3()
            with ti.ad.Tape(loss=self.U):
                self.update_U()
            self.RK4_4()
        else:
            with ti.ad.Tape(loss=self.U):
                self.update_U()
            self.Euler()
        

def main():
    cube1 = Cube(N=10,
                 E=20000,
                 nu=0.2,
                 rho=5,
                 pos=[0.7, 0.3])
    cube1.init()

    gui = ti.GUI('Simulation', res=(pixels, pixels))
    while gui.running:
        for i in range(substeps):
            cube1.substep()
        gui.circles(pos=cube1.x.to_numpy(),
                    radius=3)
        
        if Using_RK4:
            gui.text(content=f'RK4', pos=(0, 0.95), color=0xFFFFFF)
        else:
            gui.text(content=f'Explicit Euler', pos=(0, 0.95), color=0xFFFFFF)
        if Using_NeoHookean:
            gui.text(content=f'Neo-Hookean Model', pos=(0, 0.85), color=0xFFFFFF)
        else:
            gui.text(content=f'Saint Venant-Kirchhoff Model', pos=(0, 0.85), color=0xFFFFFF)
        gui.show()

if __name__ == '__main__':
    main()