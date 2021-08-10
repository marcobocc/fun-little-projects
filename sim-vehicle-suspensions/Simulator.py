import scipy.integrate

class Simulator:
    def __init__(self, model, x0, timestep):
        self.model = model
        self.x = x0
        self.timestep = timestep

    def step(self, u):
        self.x = scipy.integrate.odeint(self.model.dxdt, self.x, [0, self.timestep], args=(u,), tfirst=True)[1]
        self.x = self.model.constrain(self.x, u)
        return self.x
