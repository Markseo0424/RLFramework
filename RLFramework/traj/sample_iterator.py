import numpy as np


class SampleIterator(object):
    def __init__(self, *args, epoch=1, batch_size=None, random=False, cyclic=True):
        self.epoch = epoch
        self.batch_size = batch_size
        self.random = random
        self.cyclic = cyclic

        self.args = args
        self.sample_idx = None

        self.n = self.args[0].shape[0]
        self.batches = int(np.ceil(self.n / self.batch_size)) if batch_size is not None else 1

        self.current_epoch = 0
        self.current_batch = 0

        for arg in self.args:
            assert arg.shape[0] == self.n, "sample length mismatch!"

        if self.cyclic:
            self.sample_idx = self.get_permute()

    def get_permute(self):
        if self.random:
            index = np.random.permutation(self.n)
        else:
            index = np.arange(self.n)

        if self.batch_size is None:
            return [index]

        permute = []

        for i in range(self.batches):
            permute.append(index[i * self.batch_size: (i + 1) * self.batch_size])

        return permute

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_epoch >= self.epoch:
            raise StopIteration

        if not self.cyclic:
            if self.current_batch == 0:
                self.sample_idx = self.get_permute()

        sample = []
        for arg in self.args:
            sample.append(arg[self.sample_idx[self.current_batch]])

        self.current_batch += 1
        if self.current_batch >= self.batches:
            self.current_batch = 0
            self.current_epoch += 1

        return sample
