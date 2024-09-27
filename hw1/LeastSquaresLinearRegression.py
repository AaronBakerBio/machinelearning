'''
Test Case
---------
>>> prng = np.random.RandomState(0)
>>> N = 100

>>> true_w_F = np.asarray([1.1, -2.2, 3.3])
>>> true_b = 0.0
>>> x_NF = prng.randn(N, 3)
>>> y_N = true_b + np.dot(x_NF, true_w_F) + 0.03 * prng.randn(N)

>>> linear_regr = LeastSquaresLinearRegressor()
>>> linear_regr.fit(x_NF, y_N)

>>> yhat_N = linear_regr.predict(x_NF)
>>> np.set_printoptions(precision=3, formatter={'float':lambda x: '% .3f' % x})
>>> print(linear_regr.w_F)
[ 1.099 -2.202  3.301]
>>> print(np.asarray([linear_regr.b]))
[-0.005]
'''

import numpy as np
# No other imports allowed!

class LeastSquaresLinearRegressor(object):
    ''' A linear regression model with sklearn-like API

    Fit by solving the "least squares" optimization problem.

    Attributes
    ----------
    * self.w_F : 1D numpy array, size n_features (= F)
        vector of weights, one value for each feature
    * self.b : float
        scalar real-valued bias or "intercept"
    '''

    def __init__(self):
        ''' Constructor of an sklearn-like regressor

        Should do nothing. Attributes are only set after calling 'fit'.
        '''
        # Leave this alone
        pass

    def fit(self, x_NF, y_N):
        ''' Compute and store weights that solve least-squares problem.

        Args
        ----
        x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
            Input measurements ("features") for all examples in train set.
            Each row is a feature vector for one example.
        y_N : 1D numpy array, shape (n_examples,) = (N,)
            Response measurements for all examples in train set.
            Each row is a feature vector for one example.

        Returns
        -------
        Nothing. 

        Post-Condition
        --------------
        Internal attributes updated:
        * self.w_F (vector of weights for each feature)
        * self.b (scalar real bias, if desired)
        '''
        N, F = x_NF.shape
        X_augmented = np.hstack([np.ones((N, 1)), x_NF])
        # Solve (X^T X)^{-1} X^T y
        # Always solve theta directly, solving for
        theta = np.linalg.solve(np.dot(X_augmented.T, X_augmented), np.dot(X_augmented.T, y_N))
        self.b = theta[0]
        self.w_F = theta[1:]
        #steps for future learning, augment x_NF with 1s (add a row at the front full of 1s to account for bias.
        # This new matrix is X, use it to solve theta, which is equal to theta = [b w1 w2 w3 wf] where the first entry
        # in theta is the bias and remaining entries are the weights.
        #theta = (X^T X)^{-1} X^T y
        # index theta to get your w and b.
    def predict(self, x_MF):
        ''' Make predictions given input features for M examples

        Args
        ----
        x_MF : 2D numpy array, shape (n_examples, n_features) (M, F)
            Input measurements ("features") for all examples of interest.
            Each row is a feature vector for one example.

        Returns
        -------
        yhat_M : 1D array, size M
            Each value is the predicted scalar for one example
        '''
        # TODO FIX ME
        y_hat_M = x_MF@self.w_F+self.b
        return y_hat_M




def test_on_toy_data(N=100):
    '''
    Simple example use case
    With toy dataset with N=100 examples
    created via a known linear regression model plus small noise
    '''
    prng = np.random.RandomState(0)

    true_w_F = np.asarray([1.1, -2.2, 3.3])
    true_b = 0.0
    x_NF = prng.randn(N, 3)
    y_N = true_b + np.dot(x_NF, true_w_F) + 0.03 * prng.randn(N)

    linear_regr = LeastSquaresLinearRegressor()
    linear_regr.fit(x_NF, y_N)

    yhat_N = linear_regr.predict(x_NF)

    np.set_printoptions(precision=3, formatter={'float':lambda x: '% .3f' % x})

    print("True weights")
    print(true_w_F)
    print("Estimated weights")
    print(linear_regr.w_F)

    print("True intercept")
    print(np.asarray([true_b]))
    print("Estimated intercept")
    print(np.asarray([linear_regr.b]))

if __name__ == '__main__':
    test_on_toy_data()
