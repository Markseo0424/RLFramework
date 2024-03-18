import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_len=100000, slot_weights: dict = None):
        """
        :param buffer_len: Max length of each slot. after reaching max length, first input is popped out.
        :param slot_weights: Define names and weights of slots.
        """
        self.buffer_len = 100000

        if slot_weights is None:
            self.buffer = {"default": []}
            self.weights = {"default": 1}
            self.slots = ["default"]
        else:
            self.buffer = {}
            self.weights = {}
            self.slots = []
            for key in slot_weights.keys():
                self.buffer[key] = []
                self.weights[key] = slot_weights[key]
                self.slots.append(key)

    def reset(self, slot=None):
        """
        :param slot: Slot name to reset. Reset all if it is None.
        """
        if slot is None:
            for _slot in self.slots:
                self.buffer[_slot] = []
        else:
            self.buffer[slot] = []

    def get_buffer_len(self):
        """
        Get buffer lengths of each slots.
        """
        slot_len = {}
        for slot in self.slots:
            slot_len[slot] = len(self.buffer[slot])
        return slot_len

    def append(self, *args, slot="default"):
        """
        :param args: Elements to save to buffer.
        :param slot: Slot name of where it should be saved.
        """
        assert slot in self.slots, "no slot name"

        if self.get_buffer_len()[slot] >= self.buffer_len:
            self.buffer[slot] = self.buffer[slot][-self.buffer_len + 1:]

        self.buffer[slot].append(args)

    def sample(self, batch_size):
        """
        Get sample from replay buffer.
        :param batch_size: Size of a sample.
        :return: List of samples.
        """
        mod_weights = []
        slot_len = {}

        for slot in self.slots:
            slot_len[slot] = len(self.buffer[slot])
            if slot_len[slot]:
                mod_weights.append(self.weights[slot])
            else:
                mod_weights.append(0)

        assert sum(mod_weights) > 0
        mod_weights = [w / sum(mod_weights) for w in mod_weights]

        slot_choice = np.random.choice(a=self.slots, size=batch_size, p=mod_weights)

        res = []
        for slot in slot_choice:
            res.append(self.buffer[slot][np.random.randint(slot_len[slot])])

        return res
