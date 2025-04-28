import numpy as np

def predict(mu, cov, u, Rt):
    """
    Predicts the next state based on control input and process noise.
    
    Parameters:
        mu (np.array): State estimate vector
        cov (np.array): Covariance matrix
        u (list): Control input [translation, rotation1, rotation2]
        Rt (np.array): Process noise covariance
    
    Returns:
        mu_bar (np.array): Predicted state
        cov_bar (np.array): Updated covariance
    """
    n = len(mu)
    
    # Extract motion inputs
    [dtrans, drot1, drot2] = u
    
    # TODO: Implement motion update using the control inputs
    # Equations:
    # x' = x + d * cos(θ + δθ1)
    # y' = y + d * sin(θ + δθ1)
    # θ' = θ + δθ1 + δθ2
    motion = np.array([
        [dtrans * np.cos(mu[2][0] + drot1)],  # Compute new x position
        [dtrans * np.sin(mu[2][0] + drot1)],  # Compute new y position
        [mu[2][0] + drot1 + drot2]   # Compute new orientation
    ])
    
    # Define transformation matrix F (Hint: Extend identity matrix)
    F = np.append(np.eye(3), np.zeros((3, n-3)), axis=1)
    
    # TODO: Compute new predicted state
    # Equation: μ' = μ + F^T * motion
    mu_bar = mu + F.T.dot(motion)  # Apply motion model

    # TODO: Compute motion model Jacobian
    # Equation:
    # G = I + F^T * J * F
    # J = [[0, 0, -d*sin(θ + δθ1)],
    #      [0, 0, d*cos(θ + δθ1)],
    #      [0, 0, 0]]
    J = np.array([
        [0, 0, -dtrans * np.sin(mu[2][0] + drot1)],
        [0, 0, dtrans * np.cos(mu[2][0] + drot1)],
        [0, 0, 0]
    ])
    G = np.eye(n) + (F.T).dot(J).dot(F)

    # TODO: Predict new covariance
    # Equation: Σ' = G * Σ * G^T + F^T * R * F
    cov_bar = G.dot(cov).dot(G.T) + F.T.dot(Rt).dot(F)

    print(f"Predicted location - x: {mu_bar[0][0]:.2f}, y: {mu_bar[1][0]:.2f}, theta: {mu_bar[2][0]:.2f}")
    
    return mu_bar, cov_bar


def update(mu, cov, obs, c_prob, Qt):
    """
    Updates the state based on observations.

    Parameters:
        mu (np.array): State estimate
        cov (np.array): Covariance matrix
        obs (list): Observations [range, angle, landmark_id]
        c_prob (np.array): Class probability for each landmark (static/dynamic)
        Qt (np.array): Measurement noise covariance

    Returns:
        Updated mu, cov, and landmark class probabilities.
    """
    N = len(mu)

    for [r, theta, j] in obs:
        j = int(j)

        # TODO: Initialize landmark position if it has not been observed before
        # Equations:
        # x_j = x_r + r * cos(θ + θ_r)
        # y_j = y_r + r * sin(θ + θ_r)
        if cov[2 * j + 3, 2 * j + 3] >= 1e6 and cov[2 * j + 4, 2 * j + 4] >= 1e6:
            mu[2 * j + 3][0] = mu[0][0] + r * np.cos(mu[2][0] + theta)  # Compute x position of the landmark
            mu[2 * j + 4][0] = mu[1][0] + r * np.sin(mu[2][0] + theta)  # Compute y position of the landmark

        # TODO: Compute expected observation
        # Equations:
        # r̂ = sqrt((x_j - x_r)^2 + (y_j - y_r)^2)
        # θ̂ = atan2(y_j - y_r, x_j - x_r) - θ_r
        delta = np.array([mu[2 * j + 3][0] - mu[0][0], mu[2 * j + 4][0] - mu[1][0]])  # Compute displacement
        q = delta.T.dot(delta)
        sq = np.sqrt(q)
        z_theta = np.arctan2(delta[1], delta[0])
        z_hat = np.array([[sq], [z_theta]])  # Expected range and bearing

        # TODO: Compute Jacobian H
        # Equation:
        # H =
        # [[-Δx/sq, -Δy/sq, 0, Δx/sq, Δy/sq],
        #  [Δy/q, -Δx/q, -1, -Δy/q, Δx/q]]
        H = np.zeros((2, N))  # Initialize H with the correct shape
        H[0, 0] = -delta[0] / sq
        H[0, 1] = -delta[1] / sq
        H[0, 2] = 0
        H[0, 2 * j + 3] = delta[0] / sq
        H[0, 2 * j + 4] = delta[1] / sq

        H[1, 0] = delta[1] / q
        H[1, 1] = -delta[0] / q
        H[1, 2] = -1
        H[1, 2 * j + 3] = -delta[1] / q
        H[1, 2 * j + 4] = delta[0] / q

        # TODO: Compute Kalman Gain
        # Equation: K = Σ * H^T * (H * Σ * H^T + Q)^-1
        K = cov.dot(H.T).dot(np.linalg.inv(H.dot(cov).dot(H.T) + Qt))

        # TODO: Compute difference between actual and expected observation
        z_dif = np.array([[r], [theta]]) - z_hat
        z_dif = (z_dif + np.pi) % (2 * np.pi) - np.pi  # Normalize angle

        # TODO: Update state and covariance
        # Equations:
        # μ = μ + K * (z - ẑ)
        # Σ = (I - K * H) * Σ
        mu = mu + K.dot(z_dif)
        cov = (np.eye(N) - K.dot(H)).dot(cov)

    print(f"Updated location - x: {mu[0][0]:.2f}, y: {mu[1][0]:.2f}, theta: {mu[2][0]:.2f}")

    return mu, cov, c_prob
