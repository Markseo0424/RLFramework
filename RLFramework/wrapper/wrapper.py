class Wrapper(object):
    def __init__(self, data: dict = None):
        self.data = data

    def item(self):
        raise NotImplementedError
