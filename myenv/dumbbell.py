"""
Rigid dumbbell spacecraft orbiting a central body.

System (Paper 1 — Sanyal, Shen, McClamroch, ACC 2004, "Dynamics and control
of a 3D pendulum") specialized to the 2D planar rigid dumbbell with the
spring DOF frozen (q = ell, constant). Central-body gravity is a point mass
with gravitational parameter mu.

Generalized coordinates:
    r    — distance from central body to dumbbell center of mass
    phi  — orbital angle of CoM about central body
    theta — attitude of dumbbell relative to local radial direction

Mass matrix (rigid dumbbell, two equal masses m at +/- ell along the body):
    M(r) = diag( 2m , 2m(r^2 + ell^2) , 2m ell^2 ) with off-diagonal
    M_{phi,theta} = M_{theta,phi} = 2 m ell^2.

Potential (exact, Paper 3 style decomposition):
    V(r, theta) = - mu m / r1  - mu m / r2
    where
        r1 = sqrt(r^2 + ell^2 - 2 r ell cos(theta))
        r2 = sqrt(r^2 + ell^2 + 2 r ell cos(theta))
    Legendre expansion: V = -2 mu m / r  -  (mu m ell^2 / r^3)(3 cos^2 theta - 1)
                           + O( (ell/r)^4 ).
    V_base   = -2 mu m / r                           (point-mass / Kepler part)
    V_perturb= V - V_base                            (gravity-gradient / attitude coupling)
"""

import numpy as np
import scipy.integrate

solve_ivp = scipy.integrate.solve_ivp


class DumbbellEnv:
    """
    Gym-like environment for a 2D rigid dumbbell in a central gravity field.

    State: y = [r, phi, theta, r_dot, phi_dot, theta_dot]  (length 6)
    """

    def __init__(self, mu=10.0, m=1.0, ell=1.0):
        self.mu = mu
        self.m = m
        self.ell = ell
        self.dt = 0.05

        # Validity bounds. The inner mass distance from origin is |r - ell|
        # (minimum over theta, achieved at theta = pi). For the mass disk
        # (radius particle_radius_px = 2) to clear the planet pixel by ~1 px,
        # need |r - ell| >= ~1.1 world units, i.e. r_min >= ell + 1.1.
        # Outer mass distance r + ell must stay within world_bound.
        self.r_min = 2.5
        self.r_max = 3.5
        self.max_speed = 20.0

        # Rendering
        self.world_bound = 5.0
        self.img_size = 32
        self.particle_radius_px = 2.0
        self.planet_radius_px = 0.5

        self.state = None
        self.rng = np.random.RandomState(0)

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def dynamics(self, t, y):
        """
        Equations of motion for rigid dumbbell in central gravity.
        y = [r, phi, theta, r_dot, phi_dot, theta_dot]

        Derivation (Euler-Lagrange on L = T - V with M as above):
            2m r_ddot - 2m r phi_dot^2 + dV/dr = 0
            d/dt[ 2m(r^2 + ell^2) phi_dot + 2m ell^2 theta_dot ] + dV/dphi = 0
            d/dt[ 2m ell^2 (phi_dot + theta_dot) ] + dV/dtheta = 0
        V is independent of phi, so dV/dphi = 0.
        Subtracting the theta equation from the phi equation and solving:
            r_ddot      = r phi_dot^2 - (1/(2m)) dV/dr
            phi_ddot    = - 2 r_dot phi_dot / r  +  (1/(2 m r^2)) dV/dtheta
            theta_ddot  = - phi_ddot  -  (1/(2 m ell^2)) dV/dtheta
        """
        r, phi, theta, r_dot, phi_dot, theta_dot = y
        mu, m, ell = self.mu, self.m, self.ell

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        r1 = np.sqrt(r*r + ell*ell - 2.0 * r * ell * cos_t)
        r2 = np.sqrt(r*r + ell*ell + 2.0 * r * ell * cos_t)
        r1_3 = r1 ** 3
        r2_3 = r2 ** 3

        dV_dr = mu * m * (r - ell * cos_t) / r1_3 + mu * m * (r + ell * cos_t) / r2_3
        dV_dtheta = mu * m * r * ell * sin_t * (1.0 / r1_3 - 1.0 / r2_3)

        r_ddot = r * phi_dot * phi_dot - dV_dr / (2.0 * m)
        phi_ddot = -2.0 * r_dot * phi_dot / r + dV_dtheta / (2.0 * m * r * r)
        theta_ddot = -phi_ddot - dV_dtheta / (2.0 * m * ell * ell)

        return [r_dot, phi_dot, theta_dot, r_ddot, phi_ddot, theta_ddot]

    def reset(self):
        """
        Sample initial conditions: near-circular orbit with small attitude
        perturbation. Circular-orbit angular rate about point mass:
            phi_dot0 = sqrt(mu / r0^3)
        """
        r0 = self.rng.uniform(2.6, 3.4)
        phi0 = self.rng.uniform(0.0, 2.0 * np.pi)
        theta0 = self.rng.uniform(-np.pi, np.pi)

        phi_dot_circ = np.sqrt(self.mu / (r0 ** 3))

        r_dot0 = self.rng.uniform(-0.05, 0.05)
        phi_dot0 = phi_dot_circ + self.rng.uniform(-0.05, 0.05)
        theta_dot0 = self.rng.uniform(-0.3, 0.3)

        self.state = np.array([r0, phi0, theta0, r_dot0, phi_dot0, theta_dot0])
        return self._get_obs()

    def step(self, u=None):
        ivp = solve_ivp(
            fun=lambda t, y: self.dynamics(t, y),
            t_span=[0, self.dt],
            y0=self.state,
            rtol=1e-9,
            atol=1e-9,
        )
        self.state = ivp.y[:, -1]
        return self._get_obs(), 0.0, False, {}

    def _get_obs(self):
        """Return full 6D state."""
        return self.state.copy()

    def compute_energy(self, state=None):
        """Total energy H = T + V for conservation checks."""
        if state is None:
            state = self.state
        r, phi, theta, r_dot, phi_dot, theta_dot = state
        m, ell, mu = self.m, self.ell, self.mu

        T = (
            m * r_dot * r_dot
            + m * (r * r + ell * ell) * phi_dot * phi_dot
            + m * ell * ell * theta_dot * theta_dot
            + 2.0 * m * ell * ell * phi_dot * theta_dot
        )
        r1 = np.sqrt(r*r + ell*ell - 2.0 * r * ell * np.cos(theta))
        r2 = np.sqrt(r*r + ell*ell + 2.0 * r * ell * np.cos(theta))
        V = -mu * m / r1 - mu * m / r2
        return T + V

    def _world_to_pixel(self, wx, wy):
        b = self.world_bound
        sz = self.img_size
        px = (wx + b) / (2.0 * b) * (sz - 1)
        py = (b - wy) / (2.0 * b) * (sz - 1)  # flip y
        return px, py

    def _draw_disk(self, img, px, py, r_px, brightness):
        sz = self.img_size
        for row in range(max(0, int(py - r_px - 1)), min(sz, int(py + r_px + 2))):
            for col in range(max(0, int(px - r_px - 1)), min(sz, int(px + r_px + 2))):
                dist = np.sqrt((col - px) ** 2 + (row - py) ** 2)
                if dist <= r_px - 0.5:
                    img[row, col] = max(img[row, col], brightness)
                elif dist <= r_px + 0.5:
                    alpha = (r_px + 0.5 - dist)
                    img[row, col] = max(img[row, col], brightness * alpha)

    def render(self, mode='rgb_array'):
        """Render current state as a 32x32 grayscale image."""
        r, phi, theta = self.state[0], self.state[1], self.state[2]
        ell = self.ell
        sz = self.img_size
        img = np.zeros((sz, sz), dtype=np.float32)

        # Single-pixel marker at origin — anchors orbital angle phi without
        # competing visually with the dumbbell masses.
        cx, cy = self._world_to_pixel(0.0, 0.0)
        self._draw_disk(img, cx, cy, self.planet_radius_px, 0.9)

        # Dumbbell CoM
        xc = r * np.cos(phi)
        yc = r * np.sin(phi)

        # Body axis direction: theta is attitude relative to local radial (phi)
        body_angle = phi + theta
        dx = ell * np.cos(body_angle)
        dy = ell * np.sin(body_angle)

        # Two masses: brighter "head" and dimmer "tail" so attitude is observable
        x_a, y_a = xc + dx, yc + dy
        x_b, y_b = xc - dx, yc - dy

        pa_x, pa_y = self._world_to_pixel(x_a, y_a)
        pb_x, pb_y = self._world_to_pixel(x_b, y_b)
        self._draw_disk(img, pa_x, pa_y, self.particle_radius_px, 1.0)
        self._draw_disk(img, pb_x, pb_y, self.particle_radius_px, 0.6)

        return img

    def close(self):
        pass
