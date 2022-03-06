import sys
import time

from src.utils import parser_file,_timestamp,get_random_quote,invsqrt_matrix
from src.read_utils import read_input
from src.one_electron import oeint_calc
from src.two_electron import twoelint_calc
from src.printfile_class import PrintFile
from src.print_utils import separator,header,print_output,string_oeint,string_teint
from src.SCF import SCF,fprime

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