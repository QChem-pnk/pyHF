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