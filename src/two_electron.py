from src.molecule import Molecule
from src.basis import Basis
from src.progressbar import ProgressBar
from math import sqrt, pi, exp
import numpy as np
from src.utils import dist_sq,zeta_r,f0,permutation_int

def twoelint_calc(mol: Molecule, basis: Basis) -> np.ndarray:
    """
    Calculates the two electron integral matrix
    :param mol: Molecule data class
    :param basis: Basis set data class
    :return: Two electron integral matrix
    """
    n = basis.nbasisfunct
    TwE = np.zeros(shape=(n, n, n, n))  # Initialize two electron matrix
    max_steps = (basis.nbasisfunct * (basis.nbasisfunct + 1)) / 2  # Max number of two electron integrals
    bar = ProgressBar(max_steps ** 2, title='2e integrals')
    for i in range(n):
        for j in range(i + 1):
            for k in range(n):
                for l in range(k + 1):
                    if (i * (i + 1) / 2 + j) >= (k * (k + 1) / 2 + l):
                        for primi in basis.data[i][5]:
                            for primj in basis.data[j][5]:
                                for primk in basis.data[k][5]:
                                    for priml in basis.data[l][5]:
                                        coordlist = []
                                        for num in [i, j, k, l]:
                                            coordlist.append(mol.data[basis.data[num][3] - 1][3])  # Create coord list
                                        # Zeta, zeta' and r_x values
                                        zeta, xi, rp = zeta_r(primi[0], primj[0], coordlist[0], coordlist[1])
                                        zeta_p, xi_p, rq = zeta_r(primk[0], priml[0], coordlist[2], coordlist[3])
                                        # Rho, r_W and rpq_sq calculation
                                        pseudorho, rho, rw, rpq_sq = zeta_r(zeta, zeta_p, rp, rq, True)
                                        # K contributions
                                        k_ij = k_ijkl(zeta, xi, coordlist[0], coordlist[1])
                                        k_kl = k_ijkl(zeta_p, xi_p, coordlist[2], coordlist[3])
                                        twel = two_el_min(k_ij, k_kl, pseudorho, rho, rpq_sq)
                                        TwE[i][j][k][l] += two_fin(twel, primi[1], primj[1], primk[1], priml[1])
                    permutation_int(TwE, i, j, k, l)  # Calculate the rest of the two electron integrals
                    bar(f'({i:2d},{j:2d}|{k:2d},{l:2d})')
    return TwE


def k_ijkl(zeta: float, xi: float, coord1, coord2) -> float:
    """
    Calculation of the K_ijkl terms of two electron integral
    :param zeta: Zeta exponential
    :param xi: Xi reduced zeta exponential
    :param coord1, coord2: Coordinates
    :return: K_12 term
    """
    distsqr = dist_sq(coord1, coord2)
    exponent = exp(-xi * distsqr)
    prim = sqrt(2) * (pi ** (5 / 4)) / zeta
    return prim * exponent


def two_el_min(kij: float, kkl: float, psrho: float, rho: float, rpq_sq: float) -> float:
    """
    Calculation of a two electron integral
    :param kij, kkl: K contributions
    :param psrho: Zeta plus Zeta prime
    :param rho: Rho value (Zeta*Zeta'/psrho)
    :param rpq_sq: Combined distance of the combined distance of basis functions i, j, k and l
    :return:
    """
    return (1 / sqrt(psrho)) * kij * kkl * f0(rho * rpq_sq)


def two_fin(twe_kk: float, ci: float, cj: float, ck: float, cl: float) -> float:
    """
    Calculation of the contribution of ijkl electrons to two electron integral matrix term
    :param twe_kk: two electron integral
    :param ci, cj, ck, cl: coefficients of the primitives of the basis set
    :return: ijkl term contribution for two electron integral
    """
    return ci * cj * ck * cl * twe_kk
