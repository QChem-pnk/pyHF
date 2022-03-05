# pyHF
This is a program to do a simple Hartree-Fock calculation, either by calculating the integrals or not. It uses numpy and
scipy packages. The program is run as:

`python hf.py INPUT [-o OUTPUT -v -f -p -m MAX_ITER -c CONVERGENCE]`

There are several options for this script:

- -o output: Output file.
- -v: Print extra information.
- -m integer: Set max iterations of the SCF (default 50).
- -c float: Set convergence value of the SCF (default 1e-8).
- -f: Force the calculation of the integrals.
- -p: Print the results only on screen, do not make an output file.
- -h: Help for the options of this script.

It is recommended to run the script with verbose option (-v). Also, one letter options with no value can be combined into
one. Example:

`python hf.py INPUT -vfp`

Example input files can be found in the inp folder.
