from src.molecule import Molecule
from src.basis import Basis
from src.progressbar import ProgressBar
from typing import Tuple
from math import sqrt, pi, exp
import numpy as np
from src.utils import dist_sq,triang2sym,zeta_r,f0

def oeint_calc(mol: Molecule, basis: Basis) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the one electron integral matrices S, T and V
    :param mol: Molecule data class
    :param basis: Basis set data class
    :return: Tuple with all three one electron integrals
    """
    # Calculation of the matrix containing the square distances of each pair of atoms in the molecule
    distances = np.zeros(shape=(mol.natoms, mol.natoms))
    for i in range(mol.natoms):
        for j in range(i, mol.natoms):
            distances[i][j] = dist_sq(mol.data[i][3], mol.data[j][3])
    triang2sym(distances, 'U')
    # Initialization of the one electron matrices
    S = np.zeros(shape=(basis.nbasisfunct, basis.nbasisfunct))
    T = np.zeros(shape=(basis.nbasisfunct, basis.nbasisfunct))
    V = np.zeros(shape=(basis.nbasisfunct, basis.nbasisfunct))
    # Max number of integrals to calculate for the initialization of the progress bar
    max_steps = (basis.nbasisfunct * (basis.nbasisfunct + 1)) / 2
    bar = ProgressBar(max_steps, title='1e integrals')  # Progress bar for the calculation
    for i in range(basis.nbasisfunct):
        for j in range(i + 1):
            for primk in basis.data[i][5]:
                for priml in basis.data[j][5]:
                    zeta, xi, rp = zeta_r(primk[0], priml[0], mol.data[basis.data[i][3] - 1][3],
                                             mol.data[basis.data[j][3] - 1][3])
                    distsquared = distances[basis.data[i][3] - 1][basis.data[j][3] - 1]
                    sijkl = s_ijkl(zeta, xi, distsquared, priml[1], primk[1])
                    S[i][j] += sijkl
                    T[i][j] += t_ijkl(sijkl, xi, distsquared)
                    for I in mol.data:  # Iterate over all atoms in molecule
                        V[i][j] += v_ijklI(rp, I[3], zeta, I[1], sijkl)
            bar(f'({i:2d},{j:2d})')
    triang2sym(S)
    triang2sym(T)
    triang2sym(V)
    return S, T, V


def v_ijklI(rp, rI, zeta: float, z: int, sijkl: float) -> float:
    """
    Calculation of the v_ijklI terms of the V matrix
    :param rp: Combined coordinates
    :param rI: Nucleus coordinates
    :param zeta: Exponent of the basis function
    :param z: Atomic charge
    :param sijkl: S_ijkl term of the S matrix
    :return: V_ijklI term
    """
    distIsq = dist_sq(rp, rI)
    return -2 * z * sqrt(zeta / pi) * sijkl * f0(zeta * distIsq)


def s_ijkl(zeta: float, xi: float, distsqr: float, cl: float, ck: float) -> float:
    """
    Calculation of the S_ijkl terms of the S matrix
    :param zeta: Exponent of the basis function
    :param xi: Reduced exponent of the basis function
    :param distsqr: Distance square of the two centers of the gaussians
    :param cl, ck: Coefficients of the basis function
    :return: S_ijkl term
    """
    exponent = exp(-xi * distsqr)
    pizeta = (pi / zeta) ** (3 / 2)
    return cl * ck * exponent * pizeta


def t_ijkl(sijkl: float, xi: float, distsqr: float):
    """
    Calculation of the T_ijkl terms of the T matrix
    :param sijkl: S_ijkl term of the S matrix
    :param xi: Xi reduced exponent of the basis function
    :param distsqr: Distance square of the two centers of the gaussians
    :return: T_ijkl term
    """
    return xi * (3 - 2 * xi * distsqr) * sijkl
