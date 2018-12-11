#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# File Name : linalg_etu.py
# Creation Date : mer. 11 oct. 2017 15:17:10 CEST
# Last Modified : mer. 22 nov. 2017 12:57:56 CET
# Created By : Cyril Desjouy
#
# Copyright © 2016-2017 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
# Distributed under terms of the BSD license.
"""

Résolution de Ax = F par différentes méthodes :

    * Gauss
    * Jacobi
    * Seidel
    * Gradient conjugué

"""

import numpy as np
import time
import os


class process:
    ''' Mise en forme des résultats !'''

    def __init__(self, name):
        self.name = name
        self.template = '| {:11} | {:^21} | {:^22} | {:^13} |\n' + 80 * '-'

    def __call__(self, function):
        def wrapper(A, F, *args):
            ti = time.clock()
            sol, info = function(A, F, *args)
            tf = time.clock() - ti
            residual = np.linalg.norm(np.dot(A, sol) - F)
            print(self.template.format(self.name, tf, residual, info))
            return sol, info
        return wrapper


def triangularise(A, F):
    """
    Description
    -----------
    Make an upper triangular matrix for Ax = F.
    Parameters
    ----------
    A: 2d numpy.array
    F: 1d numpy.array
    Returns
    -------
    2d numpy.array (upper triangular) At
    1d numpy.array Ft
    """

    At = A.copy()
    Ft = F.copy()

    n = len(F)  # Length of matrices

    for curcol in range(0, n - 1):
        for curline in range(curcol + 1, n):
            ratio = At[curline, curcol] / At[curcol, curcol]
            Ft[curline] -= ratio * Ft[curcol]
            for k in range(curcol, n):
                At[curline, k] -= ratio * At[curcol, k]
    return At, Ft


@process('Built-in')
def solve_builtin(A, F):
    """
    Description
    -----------
    Solve a linear equation Ax = F with built-in function from numpy.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    F: 1d numpy.array
    Returns
    -------
    1d numpy.array x such that Ax = F
    """
    return np.linalg.solve(A, F), 'Always'


@process('Conj. Grad.')
def conjugate_grad(A, F, x, maxit, eps):
    """
    Description
    -----------
    Solve a linear equation Ax = F with conjugate gradient method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    F: 1d numpy.array
    x: 1d numpy.array of initial point
    maxit : int for maximum number iterations
    eps : float for precision
    Returns
    -------
    1d numpy.array x such that Ax = F
    """

    i = 0
    x = F.copy()

    return x, 'It. ' + repr(i)


@process('Jacobi')
def solve_jacobi(A, F, maxit, eps):
    """
    Description
    -----------
    Solve a linear equation Ax = F with Jacobi method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    F: 1d numpy.array
    maxit : int for maximum number iterations
    eps : float for precision
    Returns
    -------
    1d numpy.array x such that Ax = F
    """

    i = 0
    x = F.copy()

    return x, 'It. ' + repr(i)


@process('Seidel')
def solve_seidel(A, F, maxit, eps):
    """
    Description
    -----------
    Solve a linear equation Ax = F with Seidel method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    F: 1d numpy.array
    maxit : int for maximum number iterations
    eps : float for precision
    Returns
    -------
    1d numpy.array x such that Ax = F
    """

    i = 0
    x = F.copy()

    return x, 'It. ' + repr(i)


@process('Gauss')
def solve_gauss(A, F):
    """
    Description
    -----------
    Solve a linear equation Ax = F with Gauss method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    F: 1d numpy.array
    Returns
    -------
    1d numpy.array x such that Ax = F
    """

    x = F.copy()

    return x, 'Always'


def matrixM(N=50, k=0):

    M = np.zeros((n, n))
    M[0, 0] = 1 - k**2
    F = np.zeros(n)
    f[0] = 1

    for i in range(1, n):
        M[i, i] = 2 - k**2
        M[i - 1, 1] = -1
        M[i + 1, i] = -1

    return M, F


if __name__ == "__main__":

    # Clear screen
    _ = os.system('clear')

    # Init M and F
    n = 50
    M, F = matrixM(n, 0)

    # Init Result table
    template = '| {:11} | {:^21} | {:^22} | {:^13} |\n' + 80 * '-'
    print(template.format('Method', 'Time [s]', 'Residual', 'Convergence'))

    # Test builtin method
    # sol_builtin, info = solve_builtin(M, F)

    # Test Gauss Method
    # sol_gauss, info = solve_gauss(M, F)

    # Test Jacobi Method
    # sol_jacob, info = solve_jacobi(M, F, 2*n, 1e-2)

    # Test Seidel Method
    # sol_seidel, info = solve_seidel(M, F, 2*n, 1e-2)

    # Test Conjugate gradient Method
    # sol_grad, info = conjugate_grad(M, F, np.ones(len(F)), 2*n, 1e-2)
