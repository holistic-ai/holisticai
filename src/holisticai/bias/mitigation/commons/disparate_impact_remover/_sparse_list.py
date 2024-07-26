class SparseList(list):
    """
    A list implementation that efficiently stores sparse data.

    The SparseList class is a subclass of the built-in list class in Python.
    It is designed to efficiently store data where most of the values are the same (default value),
    and only a few values differ from the default.

    Parameters
    ----------
    default : any, optional
        The default value for the list. Default is 0.
    data : list, optional
        The list of values to initialize the SparseList with.

    Attributes
    ----------
    default : any
        The default value for the list.
    vals : dict
        A dictionary that stores the non-default values and their corresponding indices.
    size : int
        The size of the SparseList.

    Methods
    -------
    append(val)
        Appends a value to the SparseList.
    extend(iterator)
        Extends the SparseList with values from an iterator.
    sort()
        Sorts the SparseList in ascending order.
    """

    def __init__(self, default=0, data=None):
        self.default = default
        self.vals = {}
        self.size = 0

        if data:
            self.extend(data)

    def __setitem__(self, index, value):
        """
        Sets the value at the specified index.

        Parameters
        ----------
        index : int
            The index of the value to set.
        value : any
            The value to set at the specified index.
        """
        if self.default != value:
            self.vals[index] = value
        self.size += 1

    def __len__(self):
        """
        Returns the size of the SparseList.

        Returns
        -------
        int
            The size of the SparseList.
        """
        return self.size

    def __getitem__(self, index):
        """
        Gets the value at the specified index.

        Parameters
        ----------
        index : int
            The index of the value to get.

        Returns
        -------
        any
            The value at the specified index.
        """
        if index in self.vals:
            return self.vals[index]
        return self.default

    def __repr__(self):
        """
        Returns a string representation of the SparseList.

        Returns
        -------
        str
            A string representation of the SparseList.
        """
        return f"<SparseList {self.vals}>"

    def append(self, val):
        """
        Appends a value to the SparseList.

        Parameters
        ----------
        val : any
            The value to append to the SparseList.
        """
        if self.default != val:
            self.vals[self.size] = val
        self.size += 1

    def extend(self, iterator):
        """
        Extends the SparseList with values from an iterator.

        Parameters
        ----------
        iterator : iterable
            An iterable containing the values to extend the SparseList with.
        """
        for val in iterator:
            if self.default != val:
                self.vals[self.size] = val
            self.size += 1

    def sort(self):
        """
        Sorts the SparseList in ascending order.
        """
        values = sorted(self.vals.values())
        self.vals = {}
        old_size = self.size
        self.size = 0

        need_to_add_default = True
        for value in values:
            if need_to_add_default and self.default < value:
                self.size += old_size - len(values)
                need_to_add_default = False

            if self.default != value:
                self.vals[self.size] = value
            self.size += 1
