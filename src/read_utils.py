from typing import TextIO, Tuple
import numpy as np
from src.molecule import Molecule
from src.basis import Basis
from src.utils import triang2sym, permutation_int

def read_input(file_name: str, force_int=False) -> Tuple[
    Molecule, Basis, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to read the input file
    :param file_name: input file
    :param force_int: Boolean to force the calculation of integrals
    :return: Molecule and Basis classes and integral matrices
    """
    OvInt, KinInt, NucInt, TwoEInt = None, None, None, None
    with open(file_name) as f:
        for line in f:
            if 'number of atoms' in line.lower():
                nat = int(f.readline().strip())
            elif 'atom labels, atom number, coords (angstrom)' in line.lower():
                data = []
                coord_unit = line.lower().split()[5].replace('(', '').replace(')', '')
                for i in range(nat):
                    data.append(f.readline().split())
            elif 'overall charge' in line.lower():
                charge = int(f.readline().strip())
            elif 'number of basis funcs' in line.lower():
                nbasisf = int(f.readline().strip())
            elif 'maximum number of primitives' in line.lower():
                maxprim = int(f.readline().strip())
            elif 'basis set:' in line.lower():
                basisdataraw = []
                for i in range(nbasisf):
                    basisfdata = f.readline().split()
                    nprim = int(f.readline())
                    basisfprim = []
                    for j in range(nprim):
                        basisfprim.append(f.readline().split())
                    basisfdata.append(nprim)
                    basisfdata.append(basisfprim)
                    basisdataraw.append(basisfdata)
            elif 'integrals' in line.lower() and not force_int:
                OvInt, KinInt, NucInt, TwoEInt = read_integrals(nbasisf, line, f)
    moldata = Molecule(data, nat, charge, coord_unit)
    basisdata = Basis(basisdataraw, nbasisf, maxprim)
    return moldata, basisdata, OvInt, KinInt, NucInt, TwoEInt


def read_integrals(nbasisf: int, line: str, data: TextIO):
    """
    Function to read all the integrals.
    :param nbasisf: Number of basis functions
    :param line: Currently read line
    :param data: Input file
    :return: S, T, V and two-electron integral matrices
    """
    OvInt, KinInt, NucInt, TwoEInt = None, None, None, None
    while line is not None:
        if 'overlap' in line.lower():
            line, OvInt = oe_int(nbasisf, data)
        elif 'kinetic' in line.lower():
            line, KinInt = oe_int(nbasisf, data)
        elif 'nuclear attraction' in line.lower():
            line, NucInt = oe_int(nbasisf, data)
        elif 'two-electron' in line.lower():
            line, TwoEInt = te_int(nbasisf, data)
    return OvInt, KinInt, NucInt, TwoEInt


def oe_int(nbasisf: int, data: TextIO) -> np.ndarray:
    """
    Read one electron integral matrix from input
    :param nbasisf: Number of basis functions
    :param data: Input file
    :return: One electron numpy array
    """
    # Initialize The Matrix
    int_matrix = np.zeros(shape=(nbasisf, nbasisf))
    for line in data:
        if 'integrals' not in line.lower():
            lsplted = line.split()
            int_matrix[int(lsplted[0]) - 1, int(lsplted[1]) - 1] = float(lsplted[2])
        else:
            break
    else:
        line = None
    triang2sym(int_matrix)
    return line, int_matrix


def te_int(nbasisf: int, data: TextIO) -> np.ndarray:
    """
    Read two electron integral matrix from input
    :param nbasisf: Number of basis functions
    :param data: Input file
    :return: Two electron numpy array
    """
    # Initialize The Matrix
    int_matrix = np.zeros(shape=(nbasisf, nbasisf, nbasisf, nbasisf))
    for line in data:
        lsplted = line.split()
        int_matrix[int(lsplted[0]) - 1, int(lsplted[1]) - 1, int(lsplted[2]) - 1, int(lsplted[3]) - 1] = \
            float(lsplted[4])
    else:
        line = None
    for i in range(nbasisf):
        for j in range(i + 1):
            for k in range(nbasisf):
                for l in range(k + 1):
                    if (i * (i + 1) / 2 + j) >= (k * (k + 1) / 2 + l):
                        permutation_int(int_matrix, i, j, k, l)  # Permutate the matrix to get the remaining integrals
    return line, int_matrix
