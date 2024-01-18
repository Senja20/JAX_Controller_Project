import jax.numpy as jnp
from controller.GeneralController import GeneralController
from jax import grad, jit
from jax.experimental import optimizers


class NNController(GeneralController):
    """
    Neural network controller class
    AI - based
    Using a neural network to control the plant and update the parameters
    This implementation is based on the paper:
    "Neural Network Control of Nonlinear Dynamic Systems with Unknown Dead Zones":
    https://pdf.sciencedirectassets.com/272229/1-s2.0-S1568494614X00098/1-s2.0-S1568494614003640/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEC0aCXVzLWVhc3QtMSJHMEUCIQCCFpEZXEwb8rVNLg5K9IOrVifJxI9AZB8Om2IQ%2F9CjtQIgStEke%2BuNmiyT%2BoDipaTLkHAwSdsNxE0kDcKUZKhfeZgquwUI1v%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDD4pNermqVQSJ196NiqPBbVMVGoJtTwfjYEPqOWXae2LntbrGExqPwdiIxTjAMb3UHu85Ln%2FL5TfCp0m3cGLsShWmy8xDSuUND0Wua1dpLnwK%2F7%2BKpKoEJY1J%2Bb1TLDnUnWmZTMaf%2BPIdKHnFrrOAyMALGbznAKnDOENb1fAgqEgmZ%2BpAzDKOcMe5LhJ2UCtPfkhcG62%2B%2F%2BKZnJkNpOIrQ3GNNflvkqUcKBtKO3xemHrmUfz5eU%2Frfkx3ks56c%2BvNS2mvlOMqI6RPUH3gLzNfOnlmLl2mjUHGYYLKvKta46f91rMrKJul57Ni%2B%2BaMQkHQXfjhULThQVzBTaw9iC%2BXwcaVJT2A0fMaC6uTG2%2B1rN8UFVfOR%2BX71Gb9u%2F5w%2F2fHYnknQ23%2BO9pEOwz2VuLOOtFmd6JVCbHVXIDmJkzcivmaYeTRKUNozTeUIb%2BsqVcZKGhrBYZ8xnRtg6OB1BCsQYS%2BOi8z51y8fSFQO0ZyOguzhxUXKS0ipdR%2FEE7yiIT9Xotdb1CozpmLaEFm5%2BIWfbZvHDNRRno1sf3SRP2iZNU9hFXruAqZaTpFrJqj2jgqDpN0nRa4uyt24t5I%2BzXdBlerLoFAlB2BQacHbv43i1TYsjKe9kvxTwzkP8gZdzY%2BiUGAJKNiytl3Qs4gFK9mB4SBBa3CqqdjdLaTe%2FAt62JinGxwWhR%2FhVChF1EVqsIX0lZiZBEZBuG8r9JES07TDkKqFb1HLxGu5gNbXjyRLamLdcLarofnHBAFfwM1o4lYx1U3%2BOqMqTnG3Ka0fFDzFGNhNqLcHvmoHSo4kWbvDPRo2CNTdWy4b3tX%2BLB3sT223Hw%2FKkfvQdjlcT2Sksx1dLIbGymOHwnOY9cuizfb0xV%2BkjajGz%2BMH32EaG2CVMw87GkrQY6sQGXqbR4MY4tUiC7HKS0Bz6m0RNqm6H8pCQ9v5%2B2IEk%2B6XF5rjhNn3cKMqSXqDk1oSHofkHiaiHGjwfWapqw25fpeUl8ZX%2BSImrScFQOchi8ykDOu%2FbvnxmcnyO9lKu3PpExbDVaIheaLjhZXMakADj7NmqkDmKNeZNf5FiXVl5C8jlQCvphFCgl%2BKhGNr%2ButbZyd%2F1AYL9JSDLhzSghSLTOzQTKHHXqTYDUVDuD3S5UD7M%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240118T125001Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYSVJJ6AM2%2F20240118%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=62ea3894aa090a7f44c7a20c4e01fabf87a6ccbf9e15a55c35b03cfa5b81bede&hash=b2572aa4fd4a6a7097a30317dabc2a9b49a3c2052a10b4c2aa2483553f231f6a&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1568494614003640&tid=spdf-01ca49ff-6f94-4f4f-9dc6-e07782937494&sid=a001a4c0998358465f1847d1dff9f86b5854gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=140c5954505001575651&rr=8476f6139b3656c4&cc=no
    """

    def __init__(self, weights, biases, learning_rate, noise_rate):
        """Initialize the neural network controller"""
        super().__init__(learning_rate, noise_rate)
        self.weights = weights
        self.biases = biases
