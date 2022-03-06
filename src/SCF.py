from src.molecule import Molecule
from typing import Tuple, Union
import numpy as np
from scipy.linalg.lapack import dsyev  # dsyev Fortran subroutine to calculate eigenvalues and eigenvectors
from scipy.linalg.blas import dgemm  # dgemm Fortran subroutine to calculate matrix multiplications
from src.utils import invsqrt_matrix
from src.print_utils import separator,print_output,print_convergence


def coreHam(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Calculates the core hamiltonian from kinetic (T) and nuclear repulsion (V) matrices
    :param t: T matrix
    :param v: V matrix
    :return: Core hamiltonian
    """
    core_h = np.zeros(shape=t.shape)
    for i in range(len(t)):
        for j in range(len(t)):
            core_h[i][j] = t[i][j] + v[i][j]
    return core_h


def fprime(f_mat: np.ndarray, s_sqrtinv: np.ndarray) -> np.ndarray:
    """
    Calculates F' matrix from S^-1/2 and Fock matrices
    :param f_mat: Fock matrix
    :param s_sqrtinv: S^-1/2 matrix
    :return: F' matrix
    """
    return dgemm(1, dgemm(1, s_sqrtinv, f_mat, trans_a=1), s_sqrtinv)


def C_from_F(F: np.ndarray, Ssqrtinv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate coefficient matrix from S^-1/2 and Fock matrices
    :param F: Fock matrix
    :param Ssqrtinv: S^-1/2 matrix
    :return: Coefficient matrix C
    """
    Fprim = fprime(F, Ssqrtinv)
    eigenval, Cmatprime, _ = dsyev(Fprim)
    with np.printoptions(suppress=True, precision=6, sign=' ', floatmode='fixed'):
        print(f"Eigenvalues F':\n{Cmatprime}\n")
    Cmat = dgemm(1, Ssqrtinv, Cmatprime)
    return eigenval, Cmat, Fprim


def P_from_C(C: np.ndarray, occ_orb: int) -> np.ndarray:
    """
    Calculate density matrix P from coefficient matrix
    :param C: Coefficient matrix
    :param occ_orb: Number of occupied orbitals
    :return: Density matrix P
    """
    P = np.zeros(shape=C.shape)
    for i in range(len(C)):
        for j in range(len(C)):
            for e in range(int(occ_orb)):
                P[i][j] += C[i][e] * C[j][e]
    return P


def F_from_P(P: np.ndarray, core: np.ndarray, TwE: np.ndarray) -> np.ndarray:
    """
    Function to calculate the next Fock matrix from the density, core hamiltonian and two electron matrices
    :param P: Density matrix
    :param core: Core Hamiltonian matrix
    :param TwE: Two electron integral matrix
    :return: Fock matrix
    """
    F = np.zeros(shape=P.shape)
    for i in range(len(P)):
        for j in range(len(P)):
            F[i][j] = core[i][j]
            for k in range(len(P)):
                for l in range(len(P)):
                    F[i][j] += P[k][l] * (2 * TwE[i][j][k][l] - TwE[i][k][j][l])
    return F


def energy_calc(P: np.ndarray, coreH: np.ndarray, F: np.ndarray, mol: Molecule) -> Tuple[float, float]:
    """
    Function to calculate the energy and total energy
    :param P: Density matrix
    :param coreH: Core Hamiltonian
    :param F: Fock matrix
    :param mol: Molecule info class
    :return: Energy and total energy
    """
    el_energy = .0
    for i in range(len(P)):
        for j in range(len(P)):
            el_energy += P[i][j] * (coreH[i][j] + F[i][j])
    tot_energy = el_energy + mol.nuc_rep
    return el_energy, tot_energy


def check_convergence(val_old: Union[float, np.ndarray], val_new: Union[float, np.ndarray], convergence: float) -> bool:
    """
    Checks the convergence of a matrix or float value
    :param val_old: Old value
    :param val_new: New value
    :param convergence: Threshold for convergence
    :return: Boolean, True if converged, False if not
    """
    if isinstance(val_old, float):
        if abs(val_old - val_new) > convergence:
            return False
    else:
        dif_mat = np.abs(val_old - val_new)
        if np.max(dif_mat) > convergence:
            return False
    return True


def SCF(S: np.ndarray, T: np.ndarray, V: np.ndarray, TwE: np.ndarray, mol: Molecule, convergence=1e-6,
        maxiter=20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    """
    Function to perform the SCF calculation
    :param S: Overlap integrals matrix
    :param T: Kinetic integrals matrix
    :param V: Coulomb integrals matrix
    :param TwE: Two electron integrals matrix
    :param mol: Molecule info class
    :param convergence: Basis set info class
    :param maxiter: Max iterations
    :return: None
    """
    iteration = 0
    converged = [False, False, False]
    S_sqrtinv = invsqrt_matrix(S)  # Calculate S^-1/2
    coreH = coreHam(T, V)  # Calculate core Hamiltonian
    P, F = np.zeros(shape=S.shape), np.zeros(shape=S.shape)
    E, TE = .0, .0
    while True:
        if iteration == 0:  # First iteration
            Fnew = coreH.copy()  # If first iteration, copy Core Hamiltonian as Fock matrix
            separator('SCF started')
            with np.printoptions(suppress=True, precision=6, sign=' ', floatmode='fixed'):
                print(f'S^-1/2:\n{S_sqrtinv}')
                print(f'\nCore Hamiltonian:\n{coreH}')
                print(f'\nCore Hamiltonian orthogonalized:\n{fprime(coreH, S_sqrtinv)}\n')
        else:
            Fnew = F_from_P(P, coreH, TwE)  # Else, calculate Fock matrix from density matrix
            separator(f'Iteration {iteration}')
        eigenval, C, Fprim = C_from_F(Fnew, S_sqrtinv)  # Calculate coefficient matrix
        Pnew = P_from_C(C, int(mol.electrons / 2))  # Calculate new density matrix from coefficient matrix
        Enew, TEnew = energy_calc(Pnew, coreH, Fnew, mol)  # Calculate energy of the iteration
        for i, (mat, matnew) in enumerate(zip([F, P, E], [Fnew, Pnew, Enew])):  # Check convergence
            converged[i] = check_convergence(mat, matnew, convergence)
        F, P, E, TE = Fnew, Pnew, Enew, TEnew  # Store results
        if iteration != 0:
            print_output(eigenval, C, P, E, TE, F, Fprim)
            print_convergence(converged)
        else:
            print_output(eigenval, C, P, E, TE)
        if iteration >= maxiter or all(converged):  # Check convergence
            if iteration >= maxiter:
                separator('Max iterations reached')
            else:
                separator('SCF converged')
            return F, P, C, eigenval, E, TE, iteration
        iteration += 1

