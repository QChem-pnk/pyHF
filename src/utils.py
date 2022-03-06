import argparse
import numpy as np
from scipy.special import erf
from math import floor,sqrt,pi
import time
from scipy.linalg.lapack import dsyev  # dsyev Fortran subroutine to calculate eigenvalues and eigenvectors
from scipy.linalg.blas import dgemm  # dgemm Fortran subroutine to calculate matrix multiplications
import requests

def get_random_quote():
    """
    Print a random The Simpsons quote
    """
    try:
        response = requests.get('https://thesimpsonsquoteapi.glitch.me/quotes')
        if response.status_code == 200:
            json_data = response.json()
            quote = json_data[0]['quote']
            author = json_data[0]['character']
            string = f'<<{quote.strip()}>> - {author}'
            return string
        else:
            return "<<D'oh!>> - Homer Simpson"
    except:
        return '<<Ay, caramba>> - Bart Simpson'


def parser_file():
    """
    Function to parse the arguments of the script call
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='INPUT', type=str, help='Input file')
    parser.add_argument('-o', '--output', default=None, type=str, help='Output file')
    parser.add_argument('-v', '--verbose', default=False, action="store_true", help='Verbose output')
    parser.add_argument('-m', '--maxiter', default=50, type=int, help='Max Iterations')
    parser.add_argument('-c', '--convergence', default=1e-8, type=float, help='Convergence criteria')
    parser.add_argument('-f', '--force', default=False, action="store_true", help='Force calculation of integral')
    parser.add_argument('-p', '--printonly', default=False, action="store_true", help='No output')
    arguments = parser.parse_args()
    return arguments


def _timestamp(elapsed: float, prec=0):
    """
    Time taken for the calculation
    :param elapsed: Elapsed time
    :param prec: Precision to output for the decimals of a second
    :return: Time string formatted
    """
    fr = round((elapsed - floor(elapsed)) * 10 ** prec)
    s = time.strftime("%Mm %S", time.gmtime(elapsed))
    if prec > 0:
        s += f'.{fr}s'
    return s


def triang2sym(matrix: np.ndarray, lut='L'):
    """
    Makes triangular matrix into a symmetric one
    :param matrix: Matrix to symmetrize
    :param lut: Either U or L for upper or lower triangular matrices, L by default
    """
    for i in range(len(matrix)):
        for j in range(i):
            if lut == 'L':
                matrix[j, i] = matrix[i, j]
            elif lut == 'U':
                matrix[i, j] = matrix[j, i]


def permutation_int(matrix: np.ndarray, i: int, j: int, k: int, l: int):
    """
    Makes permutations of two electron integral matrix terms
    :param matrix: two electron matrix
    :param i, j, k, l: indexes of the matrix
    """
    matrix[j, i, k, l] = matrix[i, j, k, l]
    matrix[i, j, l, k] = matrix[i, j, k, l]
    matrix[j, i, l, k] = matrix[i, j, k, l]
    matrix[k, l, i, j] = matrix[i, j, k, l]
    matrix[l, k, i, j] = matrix[i, j, k, l]
    matrix[k, l, j, i] = matrix[i, j, k, l]
    matrix[l, k, j, i] = matrix[i, j, k, l]


def invsqrt_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Function to do the inverse square matrix
    :param matrix: Matrix to perform the operation
    :return: Inversed square matrix
    """
    eigenval, eigenfunct, _ = dsyev(matrix)  # dsyev fortran subroutine implementation
    for i in range(len(eigenval)):
        eigenval[i] = 1 / sqrt(eigenval[i])
    matrix_inv = dgemm(1, dgemm(1, eigenfunct, np.diag(eigenval)), eigenfunct, trans_b=1)
    return matrix_inv


def dist_sq(coord1: list, coord2: list) -> float:
    """
    Function to calculate the square distance between two coordinates
    :param coord1, coord2: Coordinates
    :return: Square distance
    """
    sq_dist = 0.
    for c1, c2 in zip(coord1, coord2):
        sq_dist += (c1 - c2) ** 2
    return sq_dist


def f0(x: float) -> float:
    """
    Calculation of the Boys function:
    .. math:: \frac{1}{2}\sqrt{\frac{\pi}{x}}erf(\sqrt{x})

    :param x: Argument of the function
    :return: Result of the function
    """
    return 0.5 * sqrt(pi / x) * erf(sqrt(x)) if x >= 1e-10 else 1.0

def zeta_r(zeta1: float, zeta2: float, coord1, coord2, combined=False):
    """
    Calculation of zetas and reduced r
    :param zeta1, zeta2: Exponentials of the basis functions
    :param coord1, coord2: Coordinates of the atoms
    :param combined: Check if r_pq is required to be calculated
    :return: zeta, xi and r_12 parameters
    """
    zeta = zeta1 + zeta2  # Zeta
    xi = zeta1 * zeta2 / zeta  # Xi
    if not isinstance(coord1, np.ndarray):
        coord1 = np.array(coord1)
    if not isinstance(coord2, np.ndarray):
        coord2 = np.array(coord2)
    r = (zeta1 * coord1 + zeta2 * coord2) / zeta
    if combined:  # If combined is set to True, calculate r_pq^2
        rpq_sq = dist_sq(coord1, coord2)
        return zeta, xi, r, rpq_sq
    return zeta, xi, r