


class BatchQueue:
    def __init__(self):
        self.input_plane = []
        self.path = []
        self.node_index = []

    def push(self, input_plane, path, node_index):
        self.input_plane.append(input_plane)
        self.path.append(path)
        self.node_index.append(node_index)

    def clear(self):
        self.input_plane = []
        self.path = []
        self.node_index = []
