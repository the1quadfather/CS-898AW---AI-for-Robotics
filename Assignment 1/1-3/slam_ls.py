import numpy as np
import matplotlib.pyplot as plt
from robot import Robot
from plotmap import plotMap, plotEstimate, plotMeasurement, plotError
from ekf import predict, update

# Simulation Parameters
n = 50  # Number of static landmarks
mapsize = 40

# Generate random static landmarks
landmark_xy = mapsize * (np.random.rand(n, 2) - 0.5)
landmark_id = np.transpose([np.linspace(0, n-1, n, dtype='uint16')])
ls = np.append(landmark_xy, landmark_id, axis=1)

# Generate dynamic landmarks
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

# Define control inputs
steps = 30
stepsize = 3
curviness = 0.5
u = np.zeros((steps, 3))
u[:, 0] = stepsize
u[4:12, 1] = curviness
u[18:26, 1] = curviness

# EKF-SLAM initialization
mu = np.zeros((2 * (n + k) + 3, 1))  # State vector: [x, y, theta, landmark1_x, landmark1_y, ...]
mu[0:3, 0] = x_init  # Initialize robot pose
cov = np.zeros((2 * (n + k) + 3, 2 * (n + k) + 3))  # Covariance matrix
cov[0:3, 0:3] = 1e-6 * np.eye(3)  # Initialize robot pose covariance

hist = [x_init]
mu_history = [mu.copy()]
ld_history = np.zeros((ld.shape[0], ld.shape[1], steps))
c_prob = np.ones(n + k) / 2  # Class probabilities (not used in this basic EKF-SLAM)

# EKF-SLAM Loop
for i in range(steps):
    # Update dynamic landmarks
    F = np.array([
        [1, 0, 0, vm, 0],
        [0, 1, 0, 0, vm],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ])
    for j in range(len(ld)):
        ld[j, :] = F.dot(ld[j, :].T).T
    ld_history[:, :, i] = ld

    # Robot move and sense
    x_true = r1.move(u[i, :])
    z = r1.sense(np.append(ls, ld[:, :3], axis=0))  # Get measurements of all landmarks

    hist.append(x_true)

    # EKF Prediction
    mu, cov = predict(mu, cov, u[i, :], Rt)

    # EKF Update
    if len(z) > 0:
        mu, cov, c_prob = update(mu, cov, z, c_prob, Qt)

    mu_history.append(mu.copy())

    # Visualization
    mu_history_np = np.array(mu_history)
    plotMap(ls, ld_history, hist, r1, mapsize)
    plotEstimate(mu_history_np, cov, r1, mapsize)
    plotMeasurement(mu, cov, z, n)
    plotError(mu_history_np, hist)

print("SLAM Execution Complete.")
plt.show()