import numpy

# NOTE: This template makes use of Python classes. If 
# you are not yet familiar with this concept, you can 
# find a short introduction here: 
# http://introtopython.org/classes.html

class LinearRegression():
    """
    Linear regression implementation.
    """

    def __init__(self, lambda_ = 0, degree = 1):
        self.lambda_ = lambda_
        self.degree = degree
            
    def fit(self, X, t):
        """
        Fits the linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples, 1]
        """        

        # TODO: YOUR CODE HERE
        X = self._add_polynomial_features(X)
        X = numpy.c_[numpy.ones(X.shape[0]), X]
        I = numpy.eye(X.shape[1])
        I[0, 0] = 0 
        self.w = numpy.linalg.inv(X.T @ X + self.lambda_ * I) @ X.T @ t


    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of shape [n_samples, 1]
        """                     

        # TODO: YOUR CODE HERE
        X = self._add_polynomial_features(X)
        X = numpy.c_[numpy.ones(X.shape[0]), X] 
        return X @ self.w
    
    def _add_polynomial_features(self, X):
        """
        Adds polynomial features up to the specified degree.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        X_poly : Array of shape [n_samples, degree * n_features]
        """
        X_poly = X
        for degree in range(2, self.degree + 1):
            X_poly = numpy.c_[X_poly, X**degree]
        return X_poly