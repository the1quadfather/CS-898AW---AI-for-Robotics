# Perform robot EKF-SLAM simulation

import numpy as np
import matplotlib.pyplot as plt
from ekf import predict, update
from robot import Robot
from plotmap import plotMap, plotEstimate, plotMeasurement, plotError

# Simulation Parameters
n = 50  # Number of static landmarks
mapsize = 40

# Generate random static landmarks
landmark_xy = mapsize * (np.random.rand(n, 2) - 0.5)
landmark_id = np.transpose([np.linspace(0, n - 1, n, dtype='uint16')])
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

# Initialize state and covariance
mu = np.zeros((2 * (n + k) + 3, 1))
mu[0][0] = x_init[0]
mu[1][0] = x_init[1]
mu[2][0] = x_init[2]
cov = np.eye(2 * (n + k) + 3)
cov[0:3, 0:3] = 0.1 * np.eye(3)
c_prob = np.ones((n + k, 1)) * 0.5

# Measurement storage for plotting
z_store = {}
zp_store = []
mu_store = []

# Initialize storage for the robot's true states
x_true_hist = [x_init]
ld_hist = np.expand_dims(ld[:, :2], axis=2)

landmark_measurements = {}

# SLAM Loop
for t in range(steps):
    # Robot motion
    r1.move(u[t, :])

    # Landmark detection
    #   Combine static and dynamic landmarks for sensing
    landmarks = np.concatenate((ls, ld[:, :3]), axis=0)
    z = r1.sense(landmarks)

    # Store measurements in z_store by landmark ID
    for measurement in z:
        landmark_id = int(measurement[2])
        if landmark_id not in z_store:
            z_store[landmark_id] = []
        z_store[landmark_id].append({
            "time": t,
            "range": measurement[0],
            "bearing": measurement[1]
        })

    # EKF predict step
    mu_bar, cov_bar = predict(mu, cov, u[t, :], Rt)
    mu_store.append(mu_bar.copy())  # Store the full state

    # EKF update step
    mu, cov, c_prob = update(mu_bar, cov_bar, z, c_prob, Qt)

    # Predicted measurements for plotting
    zp = r1.measurement_model(mu, ls, ld)
    zp_store.append(zp)

    # Update dynamic landmarks over time
    F = np.array([
        [1, 0, 0, vm, 0], [0, 1, 0, 0, vm], [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]
    ])
    for i in range(len(ld)):
        ld[i, :] = F.dot(ld[i, :].T).T
    ld_hist = np.append(ld_hist, np.expand_dims(ld[:, :2], axis=2), axis=2)
    x_true_hist.append(r1.x_true)

    # Visualization
    plotMap(ls, ld_hist, x_true_hist, r1, mapsize)
    plotEstimate(np.array(mu_store).T, cov, r1, mapsize)
    plotMeasurement(mu, cov, z, n)
    plotError(np.array(mu_store).T, x_true_hist)
    plt.pause(0.1)

    # Landmark detection
    #   Combine static and dynamic landmarks for sensing
    landmarks = np.concatenate((ls, ld[:, :3]), axis=0)
    z = r1.sense(landmarks)

    # Store measurements in z_store by landmark ID
    if len(z) > 0:  # Add this check
        for measurement in z:
            landmark_id = int(measurement[2])
            if landmark_id not in z_store:
                z_store[landmark_id] = 0
            z_store[landmark_id].append({
                "time": t,
                "range": measurement[0],
                "bearing": measurement[1]
            })

plt.show()
print("SLAM Execution Complete.")