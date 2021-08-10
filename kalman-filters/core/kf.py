import numpy as np

class KalmanFilter:
    """
    Attributes
    ----------
    F : numpy.ndarray(n, n)
        State transition matrix of the system.

    B : numpy.ndarray(n, m)
        Control input matrix of the system.

    H : numpy.ndarray(p, n)
        Measurement matrix of the system.

    Q : numpy.ndarray(n, n)
        Covariance matrix of the process noise vector, assumed zero mean.

    R : numpy.ndarray(p, p)
        Covariance matrix of the measurement noise vector, assumed zero mean.

    x : numpy.ndarray(n, 1)
        Predicted state vector of the system. If the last call was predict(),
        this is the a priori state. If the last call was update(), this is the
        a posteriori state.

    P : numpy.ndarray(n, n)
        Covariance matrix of the predicted state. If the last call was
        predict(), this is the a priori state covariance. If the last call was
        update(), this is the a posteriori prediction covariance matrix.

    residual : numpy.ndarray(p, n)
        Error between the predicted output and the measurement after update().
        If the filter is optimal, the sequence of residuals is white-noise with
        covariance matrix S.

    S : numpy.ndarray(p, p)
        Covariance matrix of the residual computed in the last call to update().

    S_inv : numpy.ndarray(p, p)
        Pre-computed inverse of the covariance matrix S computed in the last
        call to update().

    K : numpy.ndarray(n, p)
        Kalman gain matrix computed in the last call to update().
    """
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F                                              # (n, n)
        self.B = B                                              # (n, m)
        self.H = H                                              # (p, n)
        self.Q = Q                                              # (n, n)
        self.R = R                                              # (p, p)
        self.x = x0                                             # (n, 1)
        self.P = P0                                             # (n, n)
        self.residual = np.zeros(H.shape)                       # (p, n)
        self.S = np.zeros((H.shape[0], H.shape[0]))             # (p, p)
        self.S_inv = np.zeros((H.shape[0], H.shape[0]))         # (p, p)
        self.K = np.transpose(np.zeros(H.shape))                # (n, p)

    def predict(self, control_input):
        """
        Predicts the state of the system one step ahead in time and computes
        the new prediction covariance. If the previous call was update(),
        prediction is done from the a posteriori state. If the previous call
        was predict(), prediction is done from the previously predicted state.

        Parameters
        ----------
        control_input : numpy.ndarray(m, 1)
            Control input vector.
        """
        self.x = np.dot(self.F, self.x) + np.dot(self.B, control_input)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, measurement):
        """
        Updates the residual, the residual covariance, the Kalman gain and
        computes the a posteriori state and prediction covariance. In case
        there are no available measurements, update() should be ignored and
        predict() should be used to advance the filter one step forward.

        Parameters
        ----------
        measurement : numpy.ndarray(p, 1)
            Measurement vector.
        """
        self.residual = measurement - np.dot(self.H, self.x)
        PH_T = np.dot(self.P, self.H.T)
        self.S = np.dot(self.H, PH_T) + self.R
        self.S_inv = np.linalg.inv(self.S)
        self.K = np.dot(PH_T, self.S_inv)
        self.x = self.x + np.dot(self.K, self.residual)
        I_KH = np.eye(self.H.shape[1]) - np.dot(self.K, self.H)
        # Joseph form for numerical stability
        self.P = (np.dot(np.dot(I_KH, self.P), I_KH.T) +
                  np.dot(np.dot(self.K, self.R), self.K.T))

    def predict_steady(self, control_input):
        """
        Predicts the state one step ahead in time without computing the new
        prediction covariance. If the previous call was update_steady() or
        update(), prediction is done from the a posteriori state. If the
        previous call was predict_steady() or predict(), prediction is done
        from the previously predicted state.

        Parameters
        ----------
        control_input : numpy.ndarray(m, 1)
            Control input vector.
        """
        self.x = np.dot(self.F, self.x) + np.dot(self.B, control_input)

    def update_steady(self, measurement):
        """
        Updates the residual and the a posteriori state with the given
        measurement. Does not compute the residual covariance, nor the Kalman
        gain, nor the prediction covariance. In case there are no available
        measurements, update_steady() should be ignored and predict_steady()
        should be used to advance the filter one step forward.

        Parameters
        ----------
        measurement : numpy.ndarray(p, 1)
            Measurement vector.
        """
        self.residual = measurement - np.dot(self.H, self.x)
        self.x = self.x + np.dot(self.K, self.residual)
