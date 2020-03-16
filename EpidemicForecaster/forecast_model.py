import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

from sympy import symbols, lambdify, diff, exp
from sympy.solvers import solve

class SISForecaster:

    def __init__(self):
        """
            See http://www.mtholyoke.edu/~ahoyerle/math333/ThreeBasicModels.pdf 
            eq. 4.4 for definitions of sigma, mu, gamma, I0

            Implementation of the SIS model where
            a = sigma/(sigma-1)
            b = (gamma+mu)/(sigma-1)
            c = 1/I0
        """

        # These are sympy expressions!
        a, b, c, x = symbols('a b c x')
        model = 1 / (a*(1-exp(-b*x)) + c*exp(-b*x))

        da = diff(model, a)
        db = diff(model, b)
        dc = diff(model, c)
        inflection_pt = solve(
            diff(model, x, 2),
            x
        ) 

        # These are python functions!
        self.model = lambdify([x, a, b, c], model, 'numpy')
        self._da_f = lambdify([x, a, b, c], da, 'numpy')
        self._db_f = lambdify([x, a, b, c], db, 'numpy')
        self._dc_f = lambdify([x, a, b, c], dc, 'numpy')
        self._inflection_point = lambdify([a, b, c], inflection_pt, 'numpy')

        self._inflection_pt_formula = inflection_pt

        # To be set by fit.
        self._fit_params = None
        self._fit_pcov = None
        self._fit_x = None
        self._fit_y = None

        # To be set by predict.
        self._pred_x = None
        self._pred_y = None
        self._sigma = None

    def fit(
        self,
        x,
        y,
        p0=[1, 1, 1]
    ):
        self._fit_params,\
        self._fit_pcov =  curve_fit(
                    self.model, 
                    x, 
                    y,
                    p0=p0
        )

        self._fit_x = x
        self._fit_y = y

        return self._fit_params, self._fit_pcov
           
    def predict(self, x):
        self._pred_y = self.model(
            x,
            self._fit_params[0],
            self._fit_params[1],
            self._fit_params[2]
        )

        self._pred_x = x
        self._sigma = self.fit_errors(x)

        return self._pred_y

    def fit_errors(
        self,
        x
    ):
        if (self._fit_params is None
        or self._fit_pcov is None):
            raise AttributeError('You have to call fit first!')

        # Just a shorthand so the expressiob 
        # below is readable.
        p = self._fit_params
        p_cov = self._fit_pcov

        # Here we assume that the errors are not
        # very large so that the O(df/dx) approximation to the error
        # is accurate.
        return np.sqrt(
            self._da_f(x, p[0], p[1], p[2])**2 * p_cov[0, 0]
            + self._db_f(x, p[0], p[1], p[2])**2 * p_cov[1, 1]
            + self._dc_f(x, p[0], p[1], p[2])**2 * p_cov[2, 2]
            + 2*self._da_f(x, p[0], p[1], p[2])*self._db_f(x, p[0], p[1], p[2])*p_cov[0, 1]
            + 2*self._da_f(x, p[0], p[1], p[2])*self._dc_f(x, p[0], p[1], p[2])*p_cov[0, 2]
            + 2*self._db_f(x, p[0], p[1], p[2])*self._dc_f(x, p[0], p[1], p[2])*p_cov[1, 2]
        )

    def inflection_point(self):
        if self._fit_params is None:
            raise AttributeError('You have to call fit first!')
        
        return np.round(
            self._inflection_point(
                self._fit_params[0],
                self._fit_params[1],
                self._fit_params[2],
            )[0],
            1
        )

    def plot(
        self,
        x_tick_labels
    ):

        if (self._fit_x is None
        or self._pred_x is None):
            raise AttributeError('You must call fit() and predict() first!')

        _, ax = plt.subplots(
            figsize=(15, 8)
        )

        ax.plot(
            self._fit_x,
            self._fit_y,
            marker='+',
            lw=0,
            markersize=15,
            color='black',
            label='data'
        )

        ax.plot(
            self._pred_x,
            self._pred_y,
            lw=2,
            c='red',
            label='fit + forecast'
        )

        # two sigma band
        ax.fill_between(
            self._pred_x,
            self._pred_y - 2*self._sigma, 
            self._pred_y + 2*self._sigma, 
            lw=2, 
            color='grey',
            alpha=0.2,
            label=r'$\pm \,2  \, \sigma$'
        )
        # one sigma band
        ax.fill_between( 
            self._pred_x,
            self._pred_y - self._sigma, 
            self._pred_y + self._sigma, 
            lw=2, 
            color='grey',
            alpha=0.4,
            label=r'$\pm \,1  \, \sigma$'
        )
        plt.grid(alpha=0.3)
        plt.legend()
        plt.xlim([0, 50])
        ax.set_xticks(self._pred_x)
        ax.set_xticklabels(x_tick_labels, rotation =90)

        plt.ylabel('Cumulative number of cases')
        plt.xlabel('Date')

        return ax  



        
