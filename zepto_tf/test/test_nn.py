from nn import nn
import numpy as np
import unittest
import uuid
import pytest


def test_testing_itself():
    assert nn.testable()


@pytest.mark.parametrize("node, expected_bias", [
    (nn.Node('a', nn.TanhActivationFunction, False), 0.1),
    (nn.Node('b', nn.TanhActivationFunction, True), 0)
])
def test_node_constructor_1(node, expected_bias):
    assert node is not None
    np.testing.assert_allclose(node.bias, expected_bias)


def test_regularization_function():
    reg = nn.L2RegularizationFunction()
    np.testing.assert_allclose(2.0, reg.output(2.0))
    assert reg is not None


class ATestCase(unittest.TestCase):

    def test_testing_class(self):
        thing1 = nn.testable()
        self.assertTrue(thing1)