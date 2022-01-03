import taichi as ti
from functools import reduce
import numpy as np

ti.init(arch=ti.cuda)
result_dir = "./data"
video_manager = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)


@ti.data_oriented
class SPHSolver:
    material_bound = 0
    material_fluid = 1
    material_rigid = 2

    def __init__(self,
                 res,
                 screen_to_world_ratio,
                 bound,
                 rigid_num,
                 alpha=0.5,
                 dx=0.1):
        self.dim = 2
        self.res = res,
        self.screen_to_world_ratio = screen_to_world_ratio
        self.padding = 2 * dx
        self.g = ti.Vector([0.0, -9.8])
        self.c_0 = 200.0
        self.alpha = alpha
        self.rho_0 = 1000.0
        self.df_fac = 1.3
        self.dx = dx
        self.dh = self.dx * self.df_fac
        # self.dt = ti.field(ti.f32, shape=())
        self.dt = 0.00015
        self.m = self.dx ** self.dim * self.rho_0
        self.mr = self.m
        self.mrt = self.mr * 50 * 8

        self.grid_size = 2 * self.dh
        self.grid_pos = np.ceil(np.array(res) / self.screen_to_world_ratio / self.grid_size).astype(int)
        self.top_bound = bound[0]
        self.bottom_bound = bound[1]
        self.left_bound = bound[2]
        self.right_bound = bound[3]

        # PCISPH parameters
        self.s_f = ti.field(ti.f32, shape=())  # scaling factor
        self.it = 0
        self.max_it = 0
        self.sub_max_iteration = 3
        self.rho_err = ti.field(ti.f32, shape=())
        self.max_rho_err = ti.field(ti.f32, shape=())

        # Dynamic Fill particles use
        self.source_bound = ti.Vector.field(self.dim, dtype=ti.f32, shape=2)
        self.source_velocity = ti.Vector.field(self.dim, dtype=ti.f32, shape=())
        self.source_pressure = ti.field(ti.f32, shape=())
        self.source_density = ti.field(ti.f32, shape=())

        self.rigid_num = rigid_num
        self.jelly_pos = ti.Vector.field(2, dtype=ti.f32, shape=self.rigid_num)
        self.jelly_vel = ti.Vector.field(2, dtype=ti.f32, shape=self.rigid_num)
        self.rotation = ti.field(ti.f32, shape=self.rigid_num)
        self.relative_pos = ti.Vector.field(self.dim, dtype=ti.f32)
        self.rigid_idx = ti.field(ti.i32) # particle -> jelly
        self.jelly_idx = ti.field(dtype=ti.f32, shape=(self.rigid_num, 50*8))
        self.angularMomentum = ti.field(ti.f32, shape=self.rigid_num)
        self.angularVelocity = ti.field(ti.f32, shape=self.rigid_num)

        self.particle_num = ti.field(ti.i32, shape=())
        self.particle_positions = ti.Vector.field(self.dim, dtype=ti.f32)
        self.particle_velocity = ti.Vector.field(self.dim, dtype=ti.f32)
        self.particle_positions_new = ti.Vector.field(self.dim, dtype=ti.f32)
        self.particle_velocity_new = ti.Vector.field(self.dim, dtype=ti.f32)
        self.particle_pressure = ti.field(ti.f32)
        self.particle_pressure_acc = ti.Vector.field(self.dim, dtype=ti.f32)
        self.particle_density = ti.field(ti.f32)

        self.color = ti.field(ti.f32)
        self.material = ti.field(ti.i32)

        self.d_velocity = ti.Vector.field(self.dim, dtype=ti.f32)
        self.d_density = ti.field(ti.f32)

        self.grid_num_particles = ti.field(ti.i32)
        self.grid2particles = ti.field(ti.i32)
        self.particle_num_neighbors = ti.field(ti.i32)
        self.particle_neighbors = ti.field(ti.i32)

        self.max_num_particles_per_cell = 100
        self.max_num_neighbors = 100

        ti.root.dense(ti.i, 2 ** 16).place(
            self.particle_positions, self.particle_velocity,
            self.particle_pressure, self.particle_density,
            self.d_velocity, self.d_density, self.rigid_idx, self.relative_pos,
            self.material, self.color, self.particle_positions_new,
            self.particle_velocity_new, self.particle_pressure_acc)

        grid_snode = ti.root.dense(ti.ij, self.grid_pos)
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.k, self.max_num_particles_per_cell).place(self.grid2particles)

        nb_node = ti.root.dynamic(ti.i, 2**16)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, self.max_num_neighbors).place(self.particle_neighbors)

        self.s_f.from_numpy(np.array(1.0, dtype=np.float32))
        # self.dt.from_numpy(np.array(0.00015, dtype=np.float32))

    @ti.func
    def compute_grid_index(self, pos):
        return (pos / (2 * self.dh)).cast(int)

    @ti.kernel
    def allocate_particles(self):
        # Ref to pbf2d example from by Ye Kuang (k-ye)
        # https://github.com/taichi-dev/taichi/blob/master/examples/pbf2d.py
        # allocate particles to grid
        for p_i in range(self.particle_num[None]):
            # Compute the grid index
            cell = self.compute_grid_index(self.particle_positions[p_i])
            offs = self.grid_num_particles[cell].atomic_add(1)
            self.grid2particles[cell, offs] = p_i

    @ti.func
    def is_in_grid(self, c):
        res = 1
        for i in ti.static(range(self.dim)):
            res = ti.atomic_and(res, (0 <= c[i] < self.grid_pos[i]))
        return res

    @ti.func
    def is_fluid(self, p):
        # check fluid particle or bound particle
        return self.material[p]

    @ti.kernel
    def search_neighbors(self):
        # Ref to pbf2d example from by Ye Kuang (k-ye)
        # https://github.com/taichi-dev/taichi/blob/master/examples/pbf2d.py
        for p_i in range(self.particle_num[None]):
            pos_i = self.particle_positions[p_i]
            nb_i = 0
            cell = self.compute_grid_index(self.particle_positions[p_i])
            for offs in ti.static(
                    ti.grouped(ti.ndrange(*((-1, 2),) * self.dim))):
                cell_to_check = cell + offs
                if self.is_in_grid(cell_to_check) == 1:
                    for j in range(self.grid_num_particles[cell_to_check]):
                        p_j = self.grid2particles[cell_to_check, j]
                        if nb_i < self.max_num_neighbors and p_j != p_i and (
                                pos_i - self.particle_positions[p_j]
                        ).norm() < self.dh * 2.00:
                            self.particle_neighbors[p_i, nb_i] = p_j
                            nb_i.atomic_add(1)
            self.particle_num_neighbors[p_i] = nb_i

    @ti.func
    def cubic_kernel(self, r, h):
        # value of cubic spline smoothing kernel
        k = 10. / (7. * np.pi * h ** self.dim)
        q = r / h
        # assert q >= 0.0  # Metal backend is not happy with assert
        res = ti.cast(0.0, ti.f32)
        if q <= 1.0:
            res = k * (1 - 1.5 * q ** 2 + 0.75 * q ** 3)
        elif q < 2.0:
            res = k * 0.25 * (2 - q) ** 3
        return res

    @ti.func
    def cubic_kernel_derivative(self, r, h):
        # derivative of cubcic spline smoothing kernel
        k = 10. / (7. * np.pi * h ** self.dim)
        q = r / h
        # assert q > 0.0
        res = ti.cast(0.0, ti.f32)
        if q < 1.0:
            res = (k / h) * (-3 * q + 2.25 * q ** 2)
        elif q < 2.0:
            res = -0.75 * (k / h) * (2 - q) ** 2
        return res

    @ti.func
    def pressure_force(self, ptc_i, ptc_j, r, r_mod):
        # Compute the pressure force contribution, Symmetric Formula
        res = -self.m * (self.particle_pressure[ptc_i] / self.particle_density[ptc_i] ** 2
                         + self.particle_pressure[ptc_j] / self.particle_density[ptc_j] ** 2) \
              * self.cubic_kernel_derivative(r_mod, self.dh) * r / r_mod
        return res

    @ti.func
    def viscosity_force(self, ptc_i, ptc_j, r, r_mod):
        # Compute the viscosity force contribution, artificial viscosity
        res = ti.Vector([0.0 for _ in range(self.dim)], dt=ti.f32)
        v_xy = (self.particle_velocity[ptc_i] -
                self.particle_velocity[ptc_j]).dot(r)
        if v_xy < 0:
            # Artifical viscosity
            vmu = -2.0 * self.alpha * self.dx * self.c_0 / (
                    self.particle_density[ptc_i] +
                    self.particle_density[ptc_j])
            res = -self.m * vmu * v_xy / (
                    r_mod ** 2 + 0.01 * self.dx ** 2) * self.cubic_kernel_derivative(
                r_mod, self.dh) * r / r_mod
        return res

    @ti.func
    def simulate_collisions(self, ptc_i, vec, d):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.3
        self.particle_positions[ptc_i] += vec * d
        self.particle_velocity[ptc_i] -= (1.0 + c_f) * self.particle_velocity[ptc_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary(self):
        for p_i in range(self.particle_num[None]):
            if self.is_fluid(p_i) == 1:
                pos = self.particle_positions[p_i]
                if pos[0] < self.left_bound + 0.5 * self.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([1.0, 0.0], dt=ti.f32),
                        self.left_bound + 0.5 * self.padding - pos[0])
                if pos[0] > self.right_bound - 0.5 * self.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([-1.0, 0.0], dt=ti.f32),
                        pos[0] - self.right_bound + 0.5 * self.padding)
                if pos[1] > self.top_bound - self.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([0.0, -1.0], dt=ti.f32),
                        pos[1] - self.top_bound + self.padding)
                if pos[1] < self.bottom_bound + self.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([0.0, 1.0], dt=ti.f32),
                        self.bottom_bound + self.padding - pos[1])
            elif self.is_fluid(p_i) == 2:
                pos = self.particle_positions[p_i]
                if pos[1] > self.top_bound - self.padding:
                    idx = self.rigid_idx[p_i]
                    self.jelly_vel[idx] *= ti.Vector([0.0, -0.0])
                    for i in range(self.jelly_idx.shape[1]):
                        self.particle_velocity[self.jelly_idx[idx, i]] *= ti.Vector([0, -0.0])
                elif pos[1] < self.bottom_bound + self.padding:
                    idx = self.rigid_idx[p_i]
                    self.jelly_vel[idx] *= ti.Vector([0.0, -0.0])
                    for i in range(self.jelly_idx.shape[1]):
                        self.particle_velocity[self.jelly_idx[idx, i]] *= ti.Vector([0.0, -0.0])

            self.d_velocity[p_i] = ti.Vector([0.0 for _ in range(self.dim)], dt=ti.f32)

    @ti.kernel
    def pci_scaling_factor(self):
        grad_sum = ti.Vector([0.0 for _ in range(self.dim)], dt=ti.f32)
        grad_dot_sum = 0.0
        range_num = ti.cast(self.dh * 2.0 / self.dx, ti.i32)
        half_range = ti.cast(0.5 * range_num, ti.i32)
        for x in range(-half_range, half_range):
            for y in range(-half_range, half_range):
                r = ti.Vector([-x * self.dx, -y * self.dx])
                r_mod = r.norm()
                if 2.0 * self.dh > r_mod > 1e-5:
                    grad = self.cubic_kernel_derivative(r_mod,
                                                        self.dh) * r / r_mod
                    grad_sum += grad
                    grad_dot_sum += grad.dot(grad)

        beta = 2 * (self.dt * self.m / self.rho_0) ** 2
        self.s_f[None] = 1.0 / ti.max(
            beta * (grad_sum.dot(grad_sum) + grad_dot_sum), 1e-6)

    @ti.kernel
    def pci_pos_vel_prediction(self):
        for p_i in range(self.particle_num[None]):
            if self.is_fluid(p_i) == 1:
                self.particle_velocity_new[p_i] = self.particle_velocity[p_i] + self.dt * (
                        self.d_velocity[p_i] + self.particle_pressure_acc[p_i])
                self.particle_positions_new[p_i] = self.particle_positions[p_i] + self.dt * self.particle_velocity_new[p_i]
        # Initialize the max_rho_err
        self.max_rho_err[None] = 0.0

    @ti.kernel
    def pci_update_pressure(self):
        for p_i in range(self.particle_num[None]):
            if self.is_fluid(p_i) == 1:
                pos_i = self.particle_positions_new[p_i]
                d_rho = 0.0
                for j in range(self.particle_num_neighbors[p_i]):
                    p_j = self.particle_neighbors[p_i, j]
                    if self.is_fluid(p_j) == 1:
                        pos_j = self.particle_positions_new[p_j]

                        # Compute distance and its mod
                        r = pos_i - pos_j
                        r_mod = r.norm()
                        if r_mod > 1e-5:
                            # Compute Density change
                            d_rho += self.cubic_kernel_derivative(r_mod, self.dh) \
                                     * (self.particle_velocity_new[p_i] - self.particle_velocity_new[p_j]).dot(r / r_mod)
                    elif self.is_fluid(p_j) == 2:
                        pos_j = self.particle_positions[p_j]
                        r = pos_i - pos_j
                        r_mod = r.norm()
                        v_ab = self.particle_velocity[p_i] - self.particle_velocity[p_j]
                        delta = 0.0
                        if r_mod > 1e-5:
                            for r_nb_idx in range(self.particle_num_neighbors[p_j]):
                                pk_idx = self.particle_neighbors[p_j, r_nb_idx]
                                if self.is_fluid(pk_idx) == 2:
                                    pos_k = self.particle_positions[pk_idx]
                                    delta += self.cubic_kernel((pos_j - pos_k).norm(), self.dh)
                            d_rho += self.rho_0 / delta * v_ab.dot(r/r_mod)*self.cubic_kernel_derivative(r_mod, self.dh) / 10
                self.d_density[p_i] = d_rho
                # Avoid negative density variation
                self.rho_err[None] = max(0, self.particle_density[p_i] + self.dt * d_rho - self.rho_0)
                self.max_rho_err[None] = max(abs(self.rho_err[None]),self.max_rho_err[None])
                self.particle_pressure[p_i] += self.s_f[None] * self.rho_err[None]

    @ti.kernel
    def pci_update_pressure_force(self):
        for p_i in range(self.particle_num[None]):
            if self.is_fluid(p_i) == 1:
                pos_i = self.particle_positions_new[p_i]
                d_vp = ti.Vector([0.0 for _ in range(self.dim)], dt=ti.f32)
                for j in range(self.particle_num_neighbors[p_i]):
                    p_j = self.particle_neighbors[p_i, j]
                    pos_j = self.particle_positions_new[p_j]
                    # Compute distance and its mod
                    r = pos_i - pos_j
                    r_mod = r.norm()
                    if self.is_fluid(p_j) == 1:
                        if r_mod > 1e-5:
                            # Compute Pressure force contribution
                            d_vp += self.pressure_force(p_i, p_j, r, r_mod)
                self.particle_pressure_acc[p_i] = d_vp


    def pci_pc_iteration(self):
        self.pci_pos_vel_prediction()
        self.pci_update_pressure()
        self.pci_update_pressure_force()

    @ti.kernel
    def pci_compute_deltas(self):
        for p_i in range(self.particle_num[None]):
            # self.d_velocity[p_i] = ti.Vector([0.0 for _ in range(self.dim)], dt=ti.f32)
            if self.is_fluid(p_i) == 1:
                pos_i = self.particle_positions[p_i]
                d_v = ti.Vector([0.0 for _ in range(self.dim)], dt=ti.f32)

                for j in range(self.particle_num_neighbors[p_i]):
                    p_j = self.particle_neighbors[p_i, j]
                    if self.is_fluid(p_j) == 1:
                        pos_j = self.particle_positions[p_j]

                        # Compute distance and its mod
                        r = pos_i - pos_j
                        r_mod = r.norm()

                        if r_mod > 1e-5:
                            # Compute Viscosity force contribution
                            d_v += self.viscosity_force(p_i, p_j, r, r_mod)
                    elif self.is_fluid(p_j) == 2:
                        pos_j = self.particle_positions[p_j]
                        delta = 0.0
                        for r_nb_idx in range(self.particle_num_neighbors[p_j]):
                            pk_idx = self.particle_neighbors[p_j, r_nb_idx]
                            if self.is_fluid(pk_idx) == 2:
                                pos_k = self.particle_positions[pk_idx]
                                delta += self.cubic_kernel((pos_j - pos_k).norm(), self.dh)
                        x_ab = pos_i-pos_j
                        v_ab = self.particle_velocity[p_i] - self.particle_velocity[p_j]
                        v_dot_x = v_ab.dot(x_ab)
                        if v_dot_x < 0:
                            mu = self.alpha * self.dh * self.c_0 / (2 * self.particle_density[p_i])
                            PI_ab = -mu * (v_dot_x / (x_ab.norm() ** 2 + 0.01 * self.dh ** 2))
                            d_v += - self.rho_0 / delta * PI_ab * self.cubic_kernel_derivative(
                                x_ab.norm(), self.dh) * x_ab / x_ab.norm()
                            self.d_velocity[p_j] += self.m * self.rho_0 / delta * PI_ab * self.cubic_kernel_derivative(
                                x_ab.norm(), self.dh) * x_ab / x_ab.norm()
                # Add body force
                d_v += self.g
                self.d_velocity[p_i] = d_v
                # Initialize the pressure
            self.particle_pressure[p_i] = 0.0
            self.particle_pressure_acc[p_i] = ti.Vector([0.0 for _ in range(self.dim)], dt=ti.f32)

    @ti.kernel
    def pci_update_time_step(self):
        # Final position and velocity update
        for p_i in range(self.particle_num[None]):
            if self.is_fluid(p_i) == 1:
                self.particle_velocity[p_i] += self.dt * (
                        self.d_velocity[p_i] + self.particle_pressure_acc[p_i])
                self.particle_velocity[p_i][0] = min(self.particle_velocity[p_i][0], 5.0)
                self.particle_velocity[p_i][0] = max(self.particle_velocity[p_i][0], -5.0)
                self.particle_velocity[p_i][1] = min(self.particle_velocity[p_i][1], 5.0)
                self.particle_velocity[p_i][1] = max(self.particle_velocity[p_i][1], -5.0)

                self.particle_positions[p_i] += self.dt * self.particle_velocity[p_i]
            # Update density
            self.particle_density[p_i] += self.dt * self.d_density[p_i]
            # new_rho = 0.0
            # for nb_idx in range(self.particle_num_neighbors[p_i]):
            #     p_j = self.particle_neighbors[p_i, nb_idx]
            #     pos_i = self.particle_positions[p_i]
            #     pos_j = self.particle_positions[p_j]
            #     r = pos_i - pos_j
            #     if self.is_fluid(p_i) == 1:
            #         new_rho += self.m * self.cubic_kernel(r.norm(), self.dh)
            #     elif self.is_fluid(p_j) == 2:
            #         delta = 0.0
            #         for r_nb_idx in range(self.particle_num_neighbors[p_j]):
            #             pk_idx = self.particle_neighbors[p_j, r_nb_idx]
            #             if self.is_fluid(pk_idx) == 2:
            #                 pos_k = self.particle_positions[pk_idx]
            #                 delta += self.cubic_kernel((pos_j - pos_k).norm(), self.dh)
            #         new_rho += self.rho_0 / delta * self.cubic_kernel(r.norm(), self.dh)
            # self.particle_density[p_i] = new_rho

    @ti.kernel
    def get_relativepos(self):
        for i in range(self.particle_num[None]):
            if self.is_fluid(i) == 2:
                idx = self.rigid_idx[i]
                self.relative_pos[i] = self.particle_positions[i] - self.jelly_pos[idx]

    @ti.kernel
    def integraterigidbody(self):
        for i in range(self.particle_num[None]):
            if self.is_fluid(i) == 2:
                idx = self.rigid_idx[i]
                self.jelly_vel[idx] += ((self.particle_pressure_acc[i] + self.d_velocity[i])/ self.mrt + self.g*self.mr/self.mrt) * self.dt
                torque = self.relative_pos[i].cross(self.particle_pressure_acc[i] + self.d_velocity[i] + self.g*self.mr)
                self.angularMomentum[idx] += torque * self.dt
        for i in range(self.rigid_num):
            self.jelly_vel[i][0] = min(self.jelly_vel[i][0], 5.0)
            self.jelly_vel[i][0] = max(self.jelly_vel[i][0], -5.0)
            self.jelly_vel[i][1] = min(self.jelly_vel[i][1], 5.0)
            self.jelly_vel[i][1] = max(self.jelly_vel[i][1], -5.0)

        for idx in range(self.rigid_num):
            self.jelly_pos[idx] += self.jelly_vel[idx] * self.dt
            self.angularVelocity[idx] = self.angularMomentum[idx] / self.mrt
            self.rotation[idx] = self.angularVelocity[idx] * self.dt * (180.0 / np.pi)

    @ti.kernel
    def updaterigidparticle(self):
        for i in range(self.particle_num[None]):
            if self.is_fluid(i) == 2:
                idx = self.rigid_idx[i]
                rot1 = ti.Vector([ti.cos(self.rotation[idx]), -ti.sin(self.rotation[idx])])
                rot2 = ti.Vector([ti.sin(self.rotation[idx]), ti.cos(self.rotation[idx])])
                new_relativepos = ti.Vector([rot1.dot(self.relative_pos[i]), rot2.dot(self.relative_pos[i])])
                self.particle_positions[i] = new_relativepos + self.jelly_pos[idx]
                radius = ti.Vector(
                    [self.particle_positions[i][0] - self.jelly_pos[idx][0], self.particle_positions[i][1] - self.jelly_pos[idx][1], 0])
                point_vel = ti.Vector([self.jelly_vel[idx][0], self.jelly_vel[idx][1], 0]) + ti.Vector(
                    [0, 0, self.angularVelocity[idx]]).cross(radius)
                self.particle_velocity[i] = ti.Vector([point_vel[0], point_vel[1]])

    @ti.kernel
    def compute_force(self):
        for p_i in range(self.particle_num[None]):
            if self.is_fluid(p_i) == 1:
                pos_i = self.particle_positions[p_i]
                for j in range(self.particle_num_neighbors[p_i]):
                    p_j = self.particle_neighbors[p_i, j]
                    pos_j = self.particle_positions[p_j]
                    r = pos_i - pos_j
                    r_mod = r.norm()
                    if self.is_fluid(p_j) == 2:
                        delta = 0.0
                        for r_nb_idx in range(self.particle_num_neighbors[p_j]):
                            pk_idx = self.particle_neighbors[p_j, r_nb_idx]
                            if self.is_fluid(pk_idx) == 2:
                                pos_k = self.particle_positions[pk_idx]
                                delta += self.cubic_kernel((pos_j - pos_k).norm(), self.dh)
                        self.particle_pressure_acc[p_i] += -self.rho_0 / delta * (
                                self.particle_pressure[p_i] / self.particle_density[p_i] ** 2) * \
                                self.cubic_kernel_derivative(r_mod, self.dh) * r / r_mod
                        self.particle_pressure_acc[p_j] += self.m * self.rho_0 / delta * (
                                self.particle_pressure[p_i] / self.particle_density[p_i] ** 2) * \
                                                           self.cubic_kernel_derivative(r_mod, self.dh) * r / r_mod

    def step(self):
        self.grid_num_particles.fill(0)
        self.particle_neighbors.fill(-1)
        self.allocate_particles()
        self.search_neighbors()

        self.pci_compute_deltas()
        self.pci_scaling_factor()
        self.it = 0
        self.max_it = 0
        while self.max_rho_err[None] >= 0.01 * self.rho_0 or self.it < self.sub_max_iteration:
            self.pci_pc_iteration()
            self.it += 1
            self.max_it += 1
            self.max_it = max(self.it, self.max_it)
            if self.it > 30:
                print("Warning: PCISPH density does not converge, iterated %d steps"% self.it)
                break
        # Compute new velocity, position and density
        self.compute_force()
        self.get_relativepos()
        self.integraterigidbody()
        self.pci_update_time_step()
        self.updaterigidparticle()
        self.enforce_boundary()

    @ti.func
    def fill_particle(self, i, x, material, color, velocity, pressure,
                      density):
        self.particle_positions[i] = x
        self.particle_positions_new[i] = x
        self.particle_velocity[i] = velocity
        self.particle_velocity_new[i] = velocity
        self.d_velocity[i] = ti.Vector([0.0 for _ in range(self.dim)],
                                       dt=ti.f32)
        self.particle_pressure[i] = pressure
        self.particle_pressure_acc[i] = ti.Vector(
            [0.0 for _ in range(self.dim)], dt=ti.f32)
        self.particle_density[i] = density
        self.d_density[i] = 0.0
        self.color[i] = color
        self.material[i] = material

    @ti.kernel
    def fill(self, new_particles: ti.i32, new_positions: ti.ext_arr(),
             new_material: ti.i32, color: ti.i32):
        for i in range(self.particle_num[None], self.particle_num[None] + new_particles):
            self.material[i] = new_material
            x = ti.Vector.zero(ti.f32, self.dim)
            for k in ti.static(range(self.dim)):
                x[k] = new_positions[k, i - self.particle_num[None]]
            self.fill_particle(i, x, new_material, color,
                               self.source_velocity[None],
                               self.source_pressure[None],
                               self.source_density[None])

    def set_source_velocity(self, velocity):
        if velocity is not None:
            velocity = list(velocity)
            assert len(velocity) == self.dim
            self.source_velocity[None] = velocity
        else:
            for i in range(self.dim):
                self.source_velocity[None][i] = 0

    def set_source_pressure(self, pressure):
        if pressure is not None:
            self.source_pressure[None] = pressure
        else:
            self.source_pressure[None] = 0.0

    def set_source_density(self, density):
        if density is not None:
            self.source_density[None] = density
        else:
            self.source_density[None] = 0.0

    def add_cube(self,
                 lower_corner,
                 cube_size,
                 material,
                 color=0xFFFFFF,
                 density=None,
                 pressure=None,
                 velocity=None):
        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          self.dx))
        num_new_particles = reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])
        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(
            -1, reduce(lambda x, y: x * y, list(new_positions.shape[1:])))

        for i in range(self.dim):
            self.source_bound[0][i] = lower_corner[i]
            self.source_bound[1][i] = cube_size[i]

        self.set_source_velocity(velocity=velocity)
        self.set_source_pressure(pressure=pressure)
        self.set_source_density(density=density)

        self.fill(num_new_particles, new_positions, material, color)
        # Add to current particles count
        self.particle_num[None] += num_new_particles

    def add_rigid(self, center, material,  index, color=0xFFFFFF,
                  density=None, pressure=None, velocity=None):
        self.jelly_pos[index] = center
        self.jelly_vel[index] = velocity
        for n in range(50):
            t = n / 50.0 * 2 * np.pi - np.pi
            x1 = np.cos(t)
            x2 = np.sin(t)
            for m in range(8):
                i = self.particle_num[None]
                l = (m + 0.1) * self.dh * 0.3
                self.particle_positions[i] = ti.Vector([center[0]+x1*l, center[1]+x2*l])
                self.particle_positions_new[i] = ti.Vector([center[0]+x1*l, center[1]+x2*l])
                self.particle_velocity[i] = velocity
                self.particle_velocity_new[i] = velocity
                self.d_velocity[i] = ti.Vector([0.0, 0.0])
                self.particle_pressure[i] = 0.0
                self.particle_pressure_acc[i] = ti.Vector([0.0, 0.0])
                self.particle_density[i] = density
                self.d_density[i] = 0.0
                self.color[i] = color
                self.material[i] = material
                self.rigid_idx[i] = index
                self.jelly_idx[index, n*8+m] = i
                self.particle_num[None] += 1
    @ti.kernel
    def copy_dynamic_nd(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in range(self.particle_num[None]):
            for j in ti.static(range(self.dim)):
                np_x[i, j] = input_x[i][j]

    @ti.kernel
    def copy_dynamic(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in range(self.particle_num[None]):
            np_x[i] = input_x[i]

    def particle_info(self):
        np_x = np.ndarray((self.particle_num[None], self.dim),
                          dtype=np.float32)
        self.copy_dynamic_nd(np_x, self.particle_positions)
        np_v = np.ndarray((self.particle_num[None], self.dim),
                          dtype=np.float32)
        self.copy_dynamic_nd(np_v, self.particle_velocity)
        np_material = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_dynamic(np_material, self.material)
        np_color = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_dynamic(np_color, self.color)
        return {
            'position': np_x,
            'velocity': np_v,
            'material': np_material,
            'color': np_color
        }


def main():
    res = (800, 800)
    screen_to_world_ratio = 80
    dx = 0.1
    u, b, l, r = np.array([res[1], 0, 0, res[0]]) / screen_to_world_ratio
    rigid_num = 5
    gui = ti.GUI('SPH', (800, 800), background_color=0x112F41)
    sph = SPHSolver(res,
                    screen_to_world_ratio, [u, b, l, r],
                    rigid_num=rigid_num,
                    alpha=0.30,
                    dx=dx)

    # Add fluid particles
    sph.add_cube(lower_corner=[res[0] / 2 / screen_to_world_ratio - 3, 4 * dx],
                 cube_size=[6, 6],
                 velocity=[0.0, 0.0],
                 density=1000,
                 color=0x068587,
                 material=SPHSolver.material_fluid)
    colors = np.array([0xED553B, 0x068587, 0xEEEEF0, 0xFFFF00], dtype=np.uint32)
    add_cnt = 0.0
    add = True
    t = 0.0
    frame = 0
    dt=0.0001
    while gui.running and not gui.get_event(gui.ESCAPE):
        sph.step()
        # if frame == 1000:
        if add and add_cnt > 0.8:
            for i in range(rigid_num):
                sph.add_rigid(center=ti.Vector([1.8+1.5*i, 4.0]),
                              velocity=[0.0, 0.0],
                              density=1000.0,
                              color=0xED553B,
                              material=SPHSolver.material_rigid, index=i)
            add = False

        if frame % 50 == 0:
            particles = sph.particle_info()

            for pos in particles['position']:
                for j in range(len(res)):
                    pos[j] *= screen_to_world_ratio / res[j]

            gui.circles(particles['position'],
                        radius=2.0,
                        color=particles['color'])
            video_manager.write_frame(gui.get_image())
            gui.show()

        frame += 1
        t += dt
        add_cnt += dt
    video_manager.make_video(gif=True, mp4=True)
    print('done')


if __name__ == '__main__':
    main()