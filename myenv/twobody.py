"""
Two-body gravitational environment for dataset generation.

Two equal-mass particles (m=1, G=1) orbiting in 2D under Newtonian gravity.
Initial conditions: opposite sides of a circular orbit with small velocity
perturbation.

Follows the same interface as myenv/pendulum.py for compatibility with
the existing dataset generation pipeline.
"""

import numpy as np
import scipy.integrate

solve_ivp = scipy.integrate.solve_ivp


class TwoBodyEnv:
    """
    Gym-like environment for a 2D gravitational two-body system.

    State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
    Hamiltonian: H = |p1|^2/(2m) + |p2|^2/(2m) - G*m1*m2/|q1-q2|
    """

    def __init__(self, G=1.0, m1=1.0, m2=1.0):
        self.G = G
        self.m1 = m1
        self.m2 = m2
        self.dt = 0.05
        self.max_speed = 50.0
        self.state = None
        self.rng = np.random.RandomState(0)

        # Rendering
        self.world_bound = 3.0
        self.img_size = 32
        self.particle_radius_px = 2.5  # pixels

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def dynamics(self, t, y):
        """
        Equations of motion for two-body gravitational system.
        y = [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        """
        x1, y1, x2, y2, vx1, vy1, vx2, vy2 = y

        dx = x2 - x1
        dy = y2 - y1
        r = np.sqrt(dx**2 + dy**2)
        r3 = r**3

        # Gravitational acceleration: a_i = G * m_j * (q_j - q_i) / |q_j - q_i|^3
        ax1 = self.G * self.m2 * dx / r3
        ay1 = self.G * self.m2 * dy / r3
        ax2 = self.G * self.m1 * (-dx) / r3
        ay2 = self.G * self.m1 * (-dy) / r3

        return [vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2]

    def reset(self):
        """
        Initialize particles on opposite sides of a circular orbit.
        Radius R ~ Uniform[0.5, 1.5], velocities set for circular orbit + noise.
        """
        R = self.rng.uniform(0.5, 1.5)
        angle = self.rng.uniform(0, 2 * np.pi)

        # Positions: opposite sides
        x1 = R * np.cos(angle)
        y1 = R * np.sin(angle)
        x2 = -x1
        y2 = -y1

        # Circular orbit velocity: for two equal masses separated by d=2R,
        # v_circ = sqrt(G * m / (4R))  (each orbits the center of mass)
        d = 2 * R
        v_circ = np.sqrt(self.G * self.m2 / (4 * R))

        # Velocity direction: perpendicular to radius (tangent to orbit)
        # For particle at angle, tangent direction is (-sin(angle), cos(angle))
        vx1 = -v_circ * np.sin(angle) + self.rng.uniform(-0.1, 0.1)
        vy1 = v_circ * np.cos(angle) + self.rng.uniform(-0.1, 0.1)
        vx2 = v_circ * np.sin(angle) + self.rng.uniform(-0.1, 0.1)
        vy2 = -v_circ * np.cos(angle) + self.rng.uniform(-0.1, 0.1)

        self.state = np.array([x1, y1, x2, y2, vx1, vy1, vx2, vy2])
        return self._get_obs()

    def step(self, u=None):
        """Integrate dynamics one timestep forward."""
        ivp = solve_ivp(
            fun=lambda t, y: self.dynamics(t, y),
            t_span=[0, self.dt],
            y0=self.state,
            rtol=1e-10,
            atol=1e-10,
        )
        self.state = ivp.y[:, -1]
        return self._get_obs(), 0.0, False, {}

    def _get_obs(self):
        """Return full state: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]."""
        return self.state.copy()

    def compute_energy(self, state=None):
        """Compute total energy H = T + V for verification."""
        if state is None:
            state = self.state
        x1, y1, x2, y2, vx1, vy1, vx2, vy2 = state
        T = 0.5 * self.m1 * (vx1**2 + vy1**2) + 0.5 * self.m2 * (vx2**2 + vy2**2)
        r = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        V = -self.G * self.m1 * self.m2 / r
        return T + V

    def render(self, mode='rgb_array'):
        """Render current state as a 32x32 grayscale image using numpy."""
        x1, y1, x2, y2 = self.state[:4]
        b = self.world_bound
        sz = self.img_size
        r_px = self.particle_radius_px

        img = np.zeros((sz, sz), dtype=np.float32)

        # Pixel coordinate grid (y-axis flipped: row 0 = top = +y)
        # Map world [-b, b] to pixel [0, sz-1]
        for (wx, wy, brightness) in [(x1, y1, 1.0), (x2, y2, 0.65)]:
            px = (wx + b) / (2 * b) * (sz - 1)
            py = (b - wy) / (2 * b) * (sz - 1)  # flip y
            # Draw filled circle with anti-aliasing
            for row in range(max(0, int(py - r_px - 1)), min(sz, int(py + r_px + 2))):
                for col in range(max(0, int(px - r_px - 1)), min(sz, int(px + r_px + 2))):
                    dist = np.sqrt((col - px)**2 + (row - py)**2)
                    if dist <= r_px - 0.5:
                        img[row, col] = max(img[row, col], brightness)
                    elif dist <= r_px + 0.5:
                        # Anti-alias edge
                        alpha = (r_px + 0.5 - dist)
                        img[row, col] = max(img[row, col], brightness * alpha)

        return img

    def close(self):
        pass
