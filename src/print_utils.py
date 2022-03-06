from src.__init__ import __version__,__author__,__package__
import numpy as np

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