import numpy as np
from robot import Robot
from plotmap import plotMap, plotEstimate, plotMeasurement, plotError
from ekf import predict, update

# Simulation Parameters
n = 50  # Number of static landmarks
mapsize = 40

# TODO: Generate random static landmarks
# Equation: landmark_x = mapsize * (rand - 0.5)
landmark_xy = mapsize * (np.random.rand(n, 2) - 0.5)
landmark_id = np.transpose([np.linspace(0, n-1, n, dtype='uint16')])
ls = np.append(landmark_xy, landmark_id, axis=1)

# TODO: Generate dynamic landmarks
# Equation: landmark_velocity = rand - 0.5
k = 5  # Number of dynamic landmarks
vm = 5  # Velocity multiplier
landmark_xy = mapsize * (np.random.rand(k, 2) - 0.5)
landmark_v = np.random.rand(k, 2) - 0.5
landmark_id = np.transpose([np.linspace(n, n + k - 1, k, dtype='uint16')])
ld = np.append(landmark_xy, landmark_id, axis=1)
ld = np.append(ld, landmark_v, axis=1)

# Robot Initialization
fov = 80
Rt = 5 * np.array([[0.1, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
Qt = np.array([[0.01, 0], [0, 0.01]])
x_init = [0, 0, 0.5 * np.pi]

r1 = Robot(x_init, fov, Rt, Qt)

# TODO: Define control inputs
# Equations:
# u_x = step size
# u_θ = curviness
steps = 30
stepsize = 3
curviness = 0.5
u = np.zeros((steps, 3))
u[:, 0] = stepsize
u[4:12, 1] = curviness
u[18:26, 1] = curviness

# TODO: Update dynamic landmarks over time
for j in range(1, steps):
    F = np.array([
        [1, 0, 0, vm, 0], [0, 1, 0, 0, vm], [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]
    ])
    for i in range(len(ld)):
        ld[i, :] = F.dot(ld[i, :].T).T

print("SLAM Execution Complete.")
