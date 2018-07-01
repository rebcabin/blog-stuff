from nn import nn
import numpy as np
import unittest
import pytest
import matplotlib.pyplot as plt
from typing import List, Any, Callable

#   ___             _            _
#  / __|___ _ _  __| |_ __ _ _ _| |_ ___
# | (__/ _ \ ' \(_-<  _/ _` | ' \  _(_-<
#  \___\___/_||_/__/\__\__,_|_||_\__/__/


NUM_EXAMPLES      =  1000
TRAIN_SPLIT       =  0.80
PLOT_PAUSE        =  0.10
DRAWNOW_INTERVAL  =   100


#  ___       _          ___      _
# |   \ __ _| |_ __ _  / __| ___| |_ ___
# | |) / _` |  _/ _` | \__ \/ -_)  _(_-<
# |___/\__,_|\__\__,_| |___/\___|\__/__/


class Example2D(object):

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.label = z  # Haha! Physics meets ML!


class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y


def shuffle(xs: List[Any]):
    np.random.shuffle(xs)


@pytest.fixture
def num_examples():
    return NUM_EXAMPLES


def uniform_scrambled_col_vec(left, right, dim):
    return np.random.uniform(left, right, (1, dim)).T


def make_data_sets(xs, func, num_examples, train_split):
    train_size = int(num_examples * train_split)
    trainx = xs[:train_size]
    validx = xs[train_size:]
    return trainx, func(trainx), validx, func(validx)


#  ___ _     _   _   _
# | _ \ |___| |_| |_(_)_ _  __ _
# |  _/ / _ \  _|  _| | ' \/ _` |
# |_| |_\___/\__|\__|_|_||_\__, |
#                          |___/


@pytest.mark.skip('preserve this sample')
def test_draw_anything_at_all(num_examples):
    all_x = uniform_scrambled_col_vec(
        -2 * np.pi,
        2 * np.pi,
        num_examples)
    plt.figure(1)
    plt.plot(all_x.T[0])
    plt.pause(0.5)
    # plt.show(block=True)
    # plt.pause(PLOT_PAUSE)
    # plt.show(block=False)


#    _        _            _   _
#   /_\  _ _ (_)_ __  __ _| |_(_)___ _ _
#  / _ \| ' \| | '  \/ _` |  _| / _ \ ' \
# /_/ \_\_||_|_|_|_|_\__,_|\__|_\___/_||_|


# Names of classes must begin with the characters "Test"


# Must assign the result of FuncAnimation to some variable.


@pytest.mark.skip("animation sample")
class TestAnimation(object):

    def init(self):
        self.ax.set_xlim(0, 2 * np.pi)
        self.ax.set_ylim(-1, 1)
        return self.line,  # singleton tuple

    def update(self, frame):
        self.xdata.append(frame)
        self.ydata.append(np.sin(frame))
        self.line.set_data(self.xdata, self.ydata)
        return self.line,  # singleton tuple

    def test_animation(self):
        self.fig, self.ax = plt.subplots()
        self.xdata, self.ydata = [], []
        # singleton tuple
        self.line, = plt.plot([], [], 'ro', animated=True)
        from matplotlib.animation import FuncAnimation

        must_have_a_variable_here = FuncAnimation(
            self.fig,
            self.update,
            frames=np.linspace(0, 2 * np.pi, 128),
            init_func=self.init,
            blit=True)

        plt.pause(3.0)


@pytest.mark.skip("another animation sample")
class TestAnotherAnimation(object):

    @staticmethod
    def data_gen(t=0):
        cnt = 0
        while cnt < 1000:
            cnt += 1
            t += 0.1
            yield t, np.sin(2 * np.pi * t) * np.exp(-t / 10.)

    def init(self):
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlim(0, 10)
        del self.xdata[:]
        del self.ydata[:]
        self.line.set_data(self.xdata, self.ydata)
        return self.line,  # singleton tuple

    def run(self, data):
        # update the data
        t, y = data
        self.xdata.append(t)
        self.ydata.append(y)
        xmin, xmax = self.ax.get_xlim()

        if t >= xmax:
            self.ax.set_xlim(xmin, 2 * xmax)
            self.ax.figure.canvas.draw()
        self.line.set_data(self.xdata, self.ydata)

        return self.line,

    def test_another_animation(self):
        import matplotlib.animation as animation
        self.fig, self.ax = plt.subplots()
        self.xdata, self.ydata = [], []
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.grid()

        # Must assign result to a variable lest nothing shows
        ani = animation.FuncAnimation(
            self.fig,
            self.run,
            self.data_gen,
            blit=False,
            interval=10,
            repeat=False,
            init_func=self.init
        )
        plt.pause(3.0)


#    _      _   _          _   _
#   /_\  __| |_(_)_ ____ _| |_(_)___ _ _  ___
#  / _ \/ _|  _| \ V / _` |  _| / _ \ ' \(_-<
# /_/ \_\__|\__|_|\_/\__,_|\__|_\___/_||_/__/


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


#  _  _     _                  _
# | \| |___| |___ __ _____ _ _| |__
# | .` / -_)  _\ V  V / _ \ '_| / /
# |_|\_\___|\__|\_/\_/\___/_| |_\_\


def test_build_network():
    nodes = nn.build_network(
        network_shape=[2, 3, 2],
        activation=nn.ReluActivationFunction,
        output_activation=nn.ReluActivationFunction,
        regularization=nn.L2RegularizationFunction,
        input_ids=['x1', 'x2'],
        init_zero_q=True,
    )
    assert nodes is not None


#  ___     _____       _     ___                  _
# | _ \_  |_   _|__ __| |_  / __| __ _ _ __  _ __| |___ ___
# |  _/ || || |/ -_|_-<  _| \__ \/ _` | '  \| '_ \ / -_|_-<
# |_|  \_, ||_|\___/__/\__| |___/\__,_|_|_|_| .__/_\___/__/
#      |__/                                 |_|


@pytest.mark.skip('a sample')
class ATestCase(unittest.TestCase):

    def test_testing_class(self):
        thing1 = nn.testable()
        self.assertTrue(thing1)


@pytest.mark.skip('another sample')
class TestACertainClass(object):
    def test_one(self):
        x = "this"
        assert 'h' in x

    def test_two(self):
        x = "hello"
        assert x == 'hello'


@pytest.fixture
def smtp():
    import smtplib
    return smtplib.SMTP("smtp.gmail.com", 587, timeout=5)


@pytest.mark.skip('networking sample')
def test_ehlo(smtp):
    response, msg = smtp.ehlo()
    assert response == 250
