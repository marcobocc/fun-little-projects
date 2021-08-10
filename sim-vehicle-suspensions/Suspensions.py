import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class QuarterCarSuspension:
    def __init__(self,
                 car_mass, wheel_mass,
                 suspension_damping, suspension_stiffness, suspension_length_unloaded,
                 suspension_stretch_max, wheel_stiffness, wheel_radius):
        self.car_mass = car_mass
        self.wheel_mass = wheel_mass
        self.suspension_damping = suspension_damping
        self.suspension_stiffness = suspension_stiffness
        self.suspension_length_unloaded = suspension_length_unloaded
        self.suspension_stretch_max = suspension_stretch_max
        self.wheel_stiffness = wheel_stiffness
        self.wheel_radius = wheel_radius

    def dxdt(self, t, x, u):
        g = 9.8
        M = self.car_mass
        m = self.wheel_mass
        c = self.suspension_damping
        ks = self.suspension_stiffness
        DS = self.suspension_length_unloaded
        kw = self.wheel_stiffness
        DW = self.wheel_radius
        y_road = u
        y_body = x[0]
        y_body_speed = x[1]
        y_wheel = x[2]
        y_wheel_speed = x[3]

        susp_stretch = y_body - y_wheel - DS
        susp_stretch_speed = y_body_speed - y_wheel_speed
        wheel_stretch = y_wheel - y_road - DW

        susp_dampening_force = c * susp_stretch_speed
        susp_elastic_force = ks * susp_stretch
        wheel_elastic_force = kw * wheel_stretch

        y_body_accel = (-susp_dampening_force - susp_elastic_force) / M - g
        y_wheel_accel = (susp_dampening_force + susp_elastic_force - wheel_elastic_force) / m - g

        return np.array([y_body_speed, y_body_accel, y_wheel_speed, y_wheel_accel])

    def constrain(self, x, u):
        DS = self.suspension_length_unloaded
        DW = self.wheel_radius
        y_road = u
        y_body = x[0]
        y_wheel = x[2]

        susp_stretch = y_body - y_wheel - DS
        wheel_stretch = y_wheel - y_road - DW

        if wheel_stretch < 0:
            x[2] = y_road + DW
            x[3] = -x[3]
        if susp_stretch > self.suspension_stretch_max:
            x[0] = self.suspension_stretch_max + x[2] + DS
            x[1] = -x[1]
        if susp_stretch < -DS:
            x[0] = x[2]
            x[1] = -x[1]
        return x

class QuarterCarPlotter:
    def __init__(self, ax):
        self.ax = ax
        self.body_artists = []
        self.wheel_artists = []

    def create_body(self, **kwargs):
        vertices = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]])
        polygon = plt.Polygon([4, 2] * vertices, **kwargs)
        self.body_artists.append(self.ax.add_patch(polygon))
        return self.body_artists

    def create_wheel(self, **kwargs):
        circle = plt.Circle((0, 0), **kwargs)
        self.wheel_artists.append(self.ax.add_patch(circle))
        return self.wheel_artists

    def transform(self, x_pos, state):
        b_transform = mpl.transforms.Affine2D().translate(x_pos, state[0])
        for b in self.body_artists:
            b.set_transform(b_transform + self.ax.transData)
        w_transform = mpl.transforms.Affine2D().translate(x_pos, state[2])
        for w in self.wheel_artists:
            w.set_transform(w_transform + self.ax.transData)
        return self.body_artists, self.wheel_artists

class GroundPlotter:
    def __init__(self, ax):
        self.ax = ax
        self.ground_artist = None

    def create_ground(self, **kwargs):
        self.ground_artist = self.ax.plot([], [], **kwargs)[0]
        return self.ground_artist

    def set_data(self, line_data):
        self.ground_artist.set_data(*line_data)
        return self.ground_artist
