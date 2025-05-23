import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def stateToArrow(state):
    x = state[0]
    y = state[1]
    dx = 0.5*np.cos(state[2])
    dy = 0.5*np.sin(state[2])
    return x,y,dx,dy

def plotMap(ls,ldt,hist,robot,mapsize):
    plt.clf()
    
    x = robot.x_true
    fov = robot.fov
    
    # Plot true environment
    plt.subplot(1,3,1).cla()
    plt.subplot(131, aspect='equal')
    
    # Plot field of view boundaries
    plt.plot([x[0], x[0]+50*np.cos(x[2] + fov/2)], [x[1], x[1]+50*np.sin(x[2] + fov/2)], color="r")
    plt.plot([x[0], x[0]+50*np.cos(x[2] - fov/2)], [x[1], x[1]+50*np.sin(x[2] - fov/2)], color="r")
    
    for state in hist:
        plt.arrow(*stateToArrow(state), head_width=0.5)
    plt.scatter(ls[:,0],ls[:,1], s=10, marker="s", color=(0,0,1))
    
    for i in range(ldt.shape[2]):
        plt.scatter(ldt[:,0,i], ldt[:,1,i], s=10, marker="s", color=(0,1,0))
    
    plt.xlim([-mapsize/2,mapsize/2])
    plt.ylim([-mapsize/2,mapsize/2])
    plt.title('True environment')
    
    
# Plot:
    # Robot state estimates (red/green)
    # Current robot state covariances
    # Field of view
    # Currently observed landmarks with covariances and lines
    # Previously observed landmarks


def plotEstimate(mu, cov, robot, mapsize):
    a = plt.subplot(132, aspect='equal')
    a.cla()

    # Plot robot state history
    for i in range(mu.shape[1]):
        # Check if mu has at least 3 rows (x, y, theta)
        if mu.shape[0] >= 3:
            # Extract robot pose (x, y, theta)
            x = mu[0, i]
            y = mu[1, i]
            theta = mu[2, i]
            state = np.array([x, y, theta])
            a.arrow(*stateToArrow(state), head_width=0.5, color=(1, 0, 0))

    # Plot current robot field of view
    # Check if mu has at least 3 rows (x, y, theta)
    if mu.shape[0] >= 3:
        # Extract robot pose (x, y, theta)
        x = mu[0, -1]
        y = mu[1, -1]
        theta = mu[2, -1]
        state = np.array([x, y, theta])
        plt.plot([state[0], state[0] + 50 * np.cos(state[2] + robot.fov / 2)],
                 [state[1], state[1] + 50 * np.sin(state[2] + robot.fov / 2)], color="r")
        plt.plot([state[0], state[0] + 50 * np.cos(state[2] - robot.fov / 2)],
                 [state[1], state[1] + 50 * np.sin(state[2] - robot.fov / 2)], color="r")

        # Plot current robot state covariance
        robot_cov = Ellipse(xy=mu[:2, -1], width=cov[0, 0], height=cov[1, 1], angle=0)
        robot_cov.set_edgecolor((0, 0, 0))
        robot_cov.set_fill(0)
        a.add_artist(robot_cov)

    # Plot all landmarks ever observed
    n = int((len(mu) - 3) / 2)
    for i in range(n):
        if cov[2 * i + 3, 2 * i + 3] < 1e6 and cov[2 * i + 3, 2 * i + 3] < 1e6:
            zx = mu[2 * i + 3, -1]
            zy = mu[2 * i + 4, -1]
            plt.scatter(zx, zy, marker='s', s=10, color=(0, 0, 1))

    # Plot settings
    plt.xlim([-mapsize / 2, mapsize / 2])
    plt.ylim([-mapsize / 2, mapsize / 2])
    plt.title('Observations and trajectory estimate')
    plt.pause(0.1)


def plotMeasurement(mu, cov, obs, n):
    a = plt.subplot(132, aspect='equal')

    for z in obs:
        j = int(z[2])
        zx = mu[2 * j + 3, 0]  # Ensure zx is a scalar
        zy = mu[2 * j + 4, 0]  # Ensure zy is a scalar
        if j < n:
            plt.plot([mu[0, 0], zx], [mu[1, 0], zy], color=(0, 0, 1))  # Ensure mu[0, 0] and mu[1, 0] are scalars
        else:
            plt.plot([mu[0, 0], zx], [mu[1, 0], zy], color=(0, 1, 0))

        landmark_cov = Ellipse(xy=[zx, zy], width=cov[2 * j + 3][2 * j + 3], height=cov[2 * j + 4][2 * j + 4], angle=0)
        landmark_cov.set_edgecolor((0, 0, 0))
        landmark_cov.set_fill(0)
        a.add_artist(landmark_cov)
        plt.pause(0.1)

    plt.pause(0.01)

def plotError(mu, x_true):
    b = plt.subplot(133)

    # Remove the trailing dimension from mu
    mu = mu.squeeze(axis=2)

    # Extract x and y from x_true_hist
    x_true_xy = np.array([[x[0], x[1]] for x in x_true]).T

    # Adjust x_true to match the number of time steps in mu
    x_true_resized = x_true_xy[:, :mu.shape[1]]  # Take only x, y and resize

    print("Shape of mu:", mu.shape)
    print("Shape of x_true_resized:", x_true_resized.shape)

    # Calculate the difference
    dif = np.power(np.abs(mu - x_true_resized), 2)
    err = dif[0, :] + dif[1, :]

    b.plot(err, color="r")
    plt.title('Squared estimation error')
    plt.xlabel('Steps')
    plt.ylabel('Squared error')
    b.plot(dif[2,:])