class SparseList(list):
    def __init__(self, default=0, data=None):
        self.default = default
        self.vals = {}
        self.size = 0

        if data:
            self.extend(data)

    def __setitem__(self, index, value):
        if self.default != value:
            self.vals[index] = value
        self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index in self.vals:
            return self.vals[index]
        else:
            return self.default

    def __repr__(self):
        return "<SparseList {}>".format(self.vals)

    def append(self, val):
        if self.default != val:
            self.vals[self.size] = val
        self.size += 1

    def extend(self, iterator):
        for val in iterator:
            if self.default != val:
                self.vals[self.size] = val
            self.size += 1

    def sort(self):
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
