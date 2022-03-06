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