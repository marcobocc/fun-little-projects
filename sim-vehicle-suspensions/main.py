import numpy as np
import threading
import queue
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Suspensions import QuarterCarSuspension, QuarterCarPlotter, GroundPlotter
from Simulator import Simulator

model = QuarterCarSuspension(
    car_mass=400,
    wheel_mass=50,
    suspension_damping=1.3*1000,
    suspension_stiffness=25*1000,
    suspension_length_unloaded=2,
    suspension_stretch_max=1,
    wheel_stiffness=250*1000,
    wheel_radius=1
)
sim = Simulator(model, x0=np.array([3, 0, 1, 0]), timestep=0.01)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-10, 10), ylim=(-10, 10))

qcplotter = QuarterCarPlotter(ax)
groundplotter = GroundPlotter(ax)

qcplotter.create_wheel(radius=model.wheel_radius, alpha=0.7, facecolor='blue', edgecolor='blue', lw=0.5)
qcplotter.create_body(alpha=0.7, facecolor='red', edgecolor='red', lw=0.5)
qcplotter.transform(x_pos=0.0, state=sim.x)
groundplotter.create_ground(lw=1, color='black')

def generate_frame(out_queue, in_queue):
    u_history_len = 100
    u_history = np.array([0.0 for i in range(u_history_len)])
    begin = time.time()
    while True:
        elapsed = time.time() - begin + 1
        A = 0.1
        w = 10
        u_next = A * np.sin(2 * np.pi * w * elapsed) + A/2
        u_history[-1] = u_next

        curr_u = u_history[int(u_history_len / 2)]
        out_queue.put(curr_u)
        x = in_queue.get()
        u_history = np.roll(u_history, -1)

        yield [x, (np.linspace(-10, 10, len(u_history)), u_history)]

def thread_step(tid, out_queue, in_queue):
    while True:
        u = in_queue.get()
        x = sim.step(u)
        out_queue.put(x)
        time.sleep(0.01)

state_queue = queue.Queue()
u_queue = queue.Queue()
t1 = threading.Thread(target=thread_step, args=(1, state_queue, u_queue))
t1.start()

def animate(frame):
    return qcplotter.transform(x_pos=0.0, state=frame[0]), groundplotter.set_data(line_data=frame[1])

anim = animation.FuncAnimation(fig, animate, interval=sim.timestep*1000, blit=False,
                               frames=generate_frame(u_queue, state_queue))
plt.show()

t1.join()
