from base import Layer


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_feature = in_features
        self.out_features = out_features
        self.bias = bias
