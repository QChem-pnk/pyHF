from math import dist, sqrt, pi, exp, floor
from scipy.special import erf
from scipy.linalg.lapack import dsyev  # dsyev Fortran subroutine to calculate eigenvalues and eigenvectors
from scipy.linalg.blas import dgemm  # dgemm Fortran subroutine to calculate matrix multiplications
import numpy as np
import argparse
from typing import Union, TextIO, Tuple
import time
import sys
import requests


__version__ = 'v1.7.7'
__author__ = 'Sergio Sánchez Pinel'
__package__ = 'SCF Script'

ang_to_bohr = 1.8897259885789

def get_random_quote():
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


# Classes


class Molecule:
    """
    Class to store molecule properties
    """

    def __init__(self, data, n_atoms, charge, unit):
        """
        Init function
        :param data: Data for the molecule
        :param n_atoms: Number of atoms of the molecule
        :param charge: Charge of the molecule
        :param unit: Units of the coordinates
        """
        self.natoms = n_atoms
        self.charge = charge
        self.data = []
        self.nuc_rep = .0
        self.electrons = 0
        # Parse molecule info
        for item in data:
            atdata = []
            coord = []
            coord_bh = []
            if unit == 'angstrom':  # Transform to bohr
                for c in range(3):
                    coord.append(float(item[c + 2]))
                    coord_bh.append(float(item[c + 2]) * ang_to_bohr)
            elif unit == 'bohr':  # Transform to angstrom
                for c in range(3):
                    coord.append(float(item[c + 2]) / ang_to_bohr)
                    coord_bh.append(float(item[c + 2]))
            atdata.append(str(item[0]))
            atdata.append(int(item[1]))
            atdata.append(coord)
            atdata.append(coord_bh)
            self.data.append(atdata)
        self.electron_count()  # Count electrons
        self.nuclear_repulsion()  # Calculate nuclear repulsion

    def nuclear_repulsion(self):
        """
        Function to calculate the nuclear repulsion
        :return: None
        """
        for i in range(self.natoms - 1):
            for j in range(i + 1, self.natoms):
                self.nuc_rep += (self.data[i][1] * self.data[j][1]) / (dist(self.data[i][3], self.data[j][3]))

    def electron_count(self):
        """
        Function to calculate the number of electrons of the molecule
        :return: None
        """
        for i in range(self.natoms):
            self.electrons += self.data[i][1]
        self.electrons -= self.charge

    def __str__(self):
        """
        Function to print the Molecule class
        :return: String with molecule info
        """
        string = f'{"MOLECULE DATA"}'
        string += f'\n{"Number of atoms":<20}\n\t{self.natoms:<8}'
        string += f'\n{"Charge:":<20}\n\t{self.charge:<8}'
        string += f'\n{"Number of electrons":<20}\n\t{self.electrons:<8}'
        string += f'\n{"Nuclear repulsion":<20}\n\t{self.nuc_rep:<8,.6f}'
        string += f'\n{"Atom label":<10}\t{"Atom number":^11}\t{"Coord (Angstrom)":^31}'
        for item in self.data:
            string += f'\n{item[0]:<10}\t{item[1]:<11}\t{item[2][0]:>9,.6f}  {item[2][1]:>9,.6f}  {item[2][2]:>9,.6f}'
        string += f'\n{"Atom label":<10}\t{"Atom number":^11}\t{"Coord (Bohr)":^31}'
        for item in self.data:
            string += f'\n{item[0]:<10}\t{item[1]:<11}\t{item[3][0]:>9,.6f}  {item[3][1]:>9,.6f}  {item[3][2]:>9,.6f}'
        return string


class Basis:
    """
    Class to store info from basis set
    """

    def __init__(self, data, nbf, maxprim):
        """
        Init function
        :param data: Data for the basis set
        :param nbf: Number of basis functions
        :param maxprim: Max number of primitives
        """
        self.maxprim = maxprim
        self.nbasisfunct = nbf
        self.data = []
        for i in range(self.nbasisfunct):
            basisfunct = []
            for j in range(5):
                if j == 1:
                    basisfunct.append(data[i][j])
                else:
                    basisfunct.append(int(data[i][j]))
            primfunctdata = []
            for c in range(data[i][4]):
                primfunct = []
                for l in range(2):
                    primfunct.append(float(data[i][5][c][l]))
                primfunctdata.append(primfunct)
            basisfunct.append(primfunctdata)
            self.data.append(basisfunct)

    def __str__(self):
        """
        Function to print the Basis class
        :return: String with basis set info
        """
        string = f'{"BASIS SET DATA"}'
        string += f'\n{"Number of basis funcs"}\n\t{self.nbasisfunct:<8}'
        string += f'\n{"Maximum number of primitives"}\n\t{self.maxprim:<8}'
        string += f'\n{"Func no"},{"At label"},{"Z"},{"Atom no"}'
        string += f'  //  {"nPrim"}'
        string += f'  //  ({"Zeta"}\t{"Cjk"})'
        for i in range(self.nbasisfunct):
            string += f'\n{self.data[i][0]:>6} {self.data[i][1]:<2}\t{self.data[i][2]:>2}\t{self.data[i][3]:>2}'
            string += f'\n\t{self.data[i][4]:>4}'
            for j in range(self.data[i][4]):
                string += f'\n\t\t{self.data[i][5][j][0]:>14,.10f}\t{self.data[i][5][j][1]:>14,.10f}'
        return string


class ProgressBar:
    """
    Class for the progress barr
    """

    def __init__(self, length, title='', size=35, fill='█', empty=' '):
        """
        Init function
        :param length: Length of the progress bar
        :param title: Title of the progress bar
        :param size: Size of the progress bar in characters
        :param fill: Fill symbol
        :param empty: Empty symbol
        """
        self.title = title
        self.lenght = int(length)
        self.size = int(size)
        self.fill = fill
        self.empty = empty
        self.value = 0
        self.init_time = 0
        self.draw()

    def draw(self, string_print=None):
        """
        Draw progress bar function
        """
        x = int(self.size * self.value / self.lenght)
        if self.value in [0, 1]:
            self.init_time = time.time()
            time_diff = 0
        else:
            new_time = time.time()
            time_diff = new_time - self.init_time
        line = f'{self.title}: |{self.fill * x}{self.empty * (self.size - x)}| ' \
               f'{self.value}/{self.lenght} ' \
               f'({((self.value / self.lenght) * 100):.2f}%)' \
               f' in {time_diff:2.2f} s'
        if string_print is not None:
            if self.value == self.lenght:
                line += '                '
            else:
                line += f' | {string_print}'
        print(line, end='\r')
        if self.value == self.lenght:
            print('')

    def __call__(self, string_print=None):
        """
        Call the progress bar class to update
        :return:
        """
        self.value += 1
        self.draw(string_print)


class PrintFile:
    """
    Class to print to one or several outputs at the same time
    """

    def __init__(self, *files):
        """
        Init function to set outputs
        :param files: Outputs
        """
        self.files = files

    def write(self, obj):
        """
        Write to output
        :param obj: Output
        """
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        """
        Flush output
        """
        for f in self.files:
            f.flush()


# Print functions


def separator(string: str, leng=36):
    """
    Function to print step
    :param string: String to be outputted
    :param leng: Length of the separator (default 36)
    """
    print(f'{"#" * leng}')
    for line in string.splitlines():
        str1 = int((leng - len(line.strip()) - 2) / 2)
        if len(line.strip()) % 2 == 0:
            str2 = str1
        else:
            str2 = str1 + 1
        print(f'#{" " * str1}{line.strip()}{" " * str2}#')
    print(f'{"#" * leng}\n')


def header():
    """
    Function to write header to the program
    """
    separator(f'{__package__}\n{__version__}\nby\n{__author__}')


def print_output(eigenvalues, C, P, e: float, te: float, F=None, Fprim=None, final=False):
    """
    Function to print the results of each iteration
    :param eigenvalues: Orbital eigenvalues
    :param C: Coefficients matrix
    :param P: Density matrix
    :param e: Electronic energy
    :param te: Total energy
    :param F: Fock matrix
    :param Fprim: F' matrix
    :param final: Boolean to check if it is the final iteration
    """
    string_dict = {
        'Eigen': 'Orbital Eigenvalues:', 'C': 'Coefficients matrix:', 'F': 'Fock matrix:',
        'Fprim': 'Fock matrix in orthogonal basis:', 'P': 'Density matrix:',
        'E': 'Electronic Energy =', 'TE': 'Total Energy ='
    }
    values = [eigenvalues, C, F, Fprim, P, e, te]
    with np.printoptions(suppress=True, precision=6, sign=' ', floatmode='fixed'):
        for (key, value), val in zip(string_dict.items(), values):
            if final:
                value = 'Final ' + value
            if key in ['E', 'TE']:
                print(f'\n{value:<26} {val:<10,.6f}')
            elif key == 'C':
                print(f'\n{value:<26}\n{val.transpose()}')
            elif key in ['F', 'Fprim']:
                if val is not None:
                    print(f'\n{value:<26}\n{val}')
            elif key in ['Eigen']:
                print(f'{value:<26}\n{val}')
            else:
                print(f'\n{value:<26}\n{val}')
    print()


def string_oeint(matrix: np.ndarray) -> str:
    """
    Function to convert one electron matrices to string
    :param matrix: One electron matrix
    :return: String for the one electron matrix
    """
    string = '( i,j ) =  integral value'
    for i in range(len(matrix)):
        for j in range(i + 1):
            string += f'\n({i + 1:>2},{j + 1:<2}) = {matrix[i, j]:< 20,.16f}'
    return string


def string_teint(matrix: np.ndarray) -> str:
    """
    Function to convert two electron matrices to string
    :param matrix: Two electron matrix
    :return: String for the two electron matrix
    """
    string = '(i   j|k   l) =  integral value'
    for i in range(len(matrix)):
        for j in range(i + 1):
            for k in range(len(matrix)):
                for l in range(k + 1):
                    if (i * (i + 1) / 2 + j) >= (k * (k + 1) / 2 + l):
                        string += f'\n({i + 1:<2} {j + 1:>2}|{k + 1:<2} {l + 1:>2}) = {matrix[i, j, k, l]:< 20,.16f}'
    return string


def print_convergence(convergence_list: list):
    """
    Print convergence table from convergence results
    :param convergence_list: List of booleans indicating convergence
    """
    names = ['Fock matrix', 'Density matrix', 'Energy', 'All']
    end = [' | ', '\n', ' | ', '\n\n']
    converged = ['Yes' if x else 'No' for x in convergence_list]
    converged.append('Yes' if all(convergence_list) else 'No')
    print(f'{"-"*14} Convergence {"-"*14}')
    for i in range(4):
        print(f'{names[i]:<14}: {converged[i]:<3}', end=end[i])


# Common functions


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


# Read functions


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


# Calculate integrals


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


# SCF


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


# Main


def main():
    """
    Main function to perform Hartree-Fock
    """
    f = None
    arg = parser_file()
    timestart = time.time()  # Store time of start
    output = arg.output
    original = sys.stdout
    if not arg.printonly:  # Check if only printing is set
        if output is None:  # If not output defined, use name of input
            output = arg.input + '.out'
        f = open(output, 'w')
        if arg.verbose:
            sys.stdout = PrintFile(sys.stdout, f)
        else:
            header()
            sys.stdout = PrintFile(f)
    header()
    moldata, basisdata, OvInt, KinInt, NucInt, TwoEInt = read_input(arg.input, arg.force)  # Read input
    separator('Input data')
    print(f'{moldata}')
    print(f'\n{basisdata}')
    if any(a is None for a in [OvInt, KinInt, NucInt, TwoEInt]):  # Check if any of the integrals matrices are missing
        if not arg.printonly:
            sys.stdout = original
        print('\nCalculating integrals')
        if any(a is None for a in [OvInt, KinInt, NucInt]):  # Check if any of the one electron matrices is missing
            OvInt, KinInt, NucInt = oeint_calc(moldata, basisdata)
        if TwoEInt is None:  # Check if two electron integrals are missing
            TwoEInt = twoelint_calc(moldata, basisdata)
        if not arg.printonly:
            if arg.verbose:
                sys.stdout = PrintFile(sys.stdout, f)
            else:
                sys.stdout = PrintFile(f)
    print(f'\nOverlap integrals:\n{string_oeint(OvInt)}')
    print(f'\nKinetic integrals:\n{string_oeint(KinInt)}')
    print(f'\nNuclear Attraction integrals:\n{string_oeint(NucInt)}')
    print(f'\nTwo-Electron integrals:\n{string_teint(TwoEInt)}\n')
    F, P, C, eigenvalues, E, TE, iteration = SCF(OvInt, KinInt, NucInt, TwoEInt, moldata, arg.convergence,
                                                 arg.maxiter)  # Do SCF
    print(f'SCF finished in {iteration} iterations\n')
    Fprim = fprime(F, invsqrt_matrix(OvInt))  # Calculate final Fock matrix in the orthogonal basis
    print_output(eigenvalues, C, P, E, TE, F, Fprim, final=True)  # Print final output
    elapsed = time.time() - timestart  # Calculate elapsed time
    s = _timestamp(elapsed, 3)  # Print elapsed time
    separator('Finished')
    print(f'Finished in {s}')
    if not arg.printonly:
        f.close()
        sys.stdout = original
        print(f'\nResults saved in {output}')
        string = get_random_quote()
        print(f'\n{string}')


if __name__ == "__main__":
    main()

