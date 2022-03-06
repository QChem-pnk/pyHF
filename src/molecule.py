from math import dist

ang_to_bohr = 1.8897259885789

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