import time

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