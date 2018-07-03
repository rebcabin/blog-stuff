from nn import nn
import numpy as np
import unittest
import pytest
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Callable


#  __  __   _   ___ ___ _____   _____   _____ ___  ___   ___
# |  \/  | /_\ / __/ __|_ _\ \ / / __| |_   _/ _ \|   \ / _ \
# | |\/| |/ _ \\__ \__ \| | \ V /| _|    | || (_) | |) | (_) |
# |_|  |_/_/ \_\___/___/___| \_/ |___|   |_| \___/|___/ \___/


# TODO: optimize with numpy. Almost everything in here is wicked slow.
# TODO: it's a straight port from TypeScript to ensure we get it right.


#   ___             _            _
#  / __|___ _ _  __| |_ __ _ _ _| |_ ___
# | (__/ _ \ ' \(_-<  _/ _` | ' \  _(_-<
#  \___\___/_||_/__/\__\__,_|_||_\__/__/


NUM_EXAMPLES        =   300
TRAIN_SPLIT         =  0.50
PLOT_PAUSE          =  0.10
DRAWNOW_INTERVAL    =   100
BATCH_SIZE          =    10
LEARNING_RATE       = 0.001
REGULARIZATION_RATE =  0.01


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


np.random.seed(42)  # for reproducibility


@pytest.fixture
def num_examples():
    return NUM_EXAMPLES


def classify_two_gauss_data(
        num_examples: int,
        noise_variance: float) -> Example2D:

    variance_scale = 1.0  # a d3 thing from TypeScript; not needed here
    variance = noise_variance * variance_scale
    std = np.sqrt(variance)
    points = []  # eventually return [Example2D]

    def gen_gauss(mu_x: float, mu_y: float, label: float) -> None:
        x = np.random.randn(1)[0] * std  +  mu_x
        y = np.random.randn(1)[0] * std  +  mu_y
        points.append(Example2D(x, y, label))

    gen_gauss(2, 2, 1)
    gen_gauss(-2, -2, 1)

    return points


def regress_plane(
        num_examples: int,
        noise_variance: float) -> List[Example2D]:
    """TODO: noise model is wrong."""
    radius = 6
    # label_scale from original is a d3 thing; not needed here
    points = []  # eventually a List[Example2D]
    std = np.sqrt(noise_variance)

    def lr():
        return np.random.random() * 2 * radius - radius

    for _ in range(num_examples):
        x = lr()
        y = lr()
        noise_x = lr() * std
        noise_y = lr() * std
        label = x + noise_x + y + noise_y
        points.append(Example2D(x, y, label))

    return points


def regress_gaussian(
        num_examples: int,
        noise_variance: float) -> List[Example2D]:
    pass  # TODO


def classify_circle_data(
        num_examples: int,
        noise: float) -> List[Example2D]:

    radius = 5
    # label_scale from original is a d3 thing; not needed here
    points = []  # eventually a List[Example2D]

    def euclidean_distance(p1: Point, p2: Point) -> float:
        x2 = (p2.x - p1.x) ** 2
        y2 = (p2.y - p1.y) ** 2
        result = np.sqrt(x2 + y2)
        return result

    def get_circle_label(p: Point, center: Point) -> int:
        d = euclidean_distance(p, center)
        result = 1 if d < radius * 0.5 else -1
        return result

    def lr():
        return np.random.random() * 2 * radius - radius

    origin = Point(0, 0)

    def make_points(inner_radial_fraction, outer_radial_fraction):
        for _ in range(int(num_examples / 2)):
            # consider fuzzing the radius | angle instead of x and y
            d = outer_radial_fraction - inner_radial_fraction
            r = radius * (np.random.random() * d + inner_radial_fraction)
            a = np.random.random() * 2 * np.pi
            x = r * np.sin(a) # Huh? usually cos
            y = r * np.cos(a)
            noise_x = lr() * noise
            noise_y = lr() * noise
            label = get_circle_label(
                Point(x + noise_x, y + noise_y),
                origin)
            points.append(Example2D(x, y, label))

    # Positive labels inside the circle
    make_points(0.0, 0.5)
    # Negative labels outside the circle
    make_points(0.7, 1.0)

    return points


def classify_circle_train_test_split(
        num_examples: int,
        data: List[Example2D]) -> Dict:
    split = int(num_examples * TRAIN_SPLIT)
    result = {'training data': data[:split],
              'testing data': data[split:]}
    return result


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


# PyCharm unnecessarily greys-out the following import.
# This import is necessary lest "projection='3d'" be unknown below.
from mpl_toolkits.mplot3d import Axes3D


def draw_2D_points(points: List[Example2D]) -> None:
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.label for p in points]
    cs = ['r' if p.label > 0 else 'b' for p in points]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # the 'c=' is necessary lest 'cs' be ignored.
    ax.scatter(xs, ys, zs, c=cs)
    plt.pause(PLOT_PAUSE)
    # plt.show()


@pytest.mark.parametrize("noise", [0.10, 0.25, 0.50, 0.75, 1.0])
def test_draw_classify_circle_data(num_examples, noise):
    # See the following line in 'playground.ts': secret magic scaling of state.noise
    #   let data = generator(numSamples, state.noise / 100);
    # We won't do that; we'll just let the noise be a number in units of the radius
    draw_2D_points(classify_circle_data(num_examples, noise))


@pytest.mark.skip('at present, not needed')
def test_draw_regress_plane(num_examples):
    """TODO: noise model is wrong."""
    draw_2D_points(regress_plane(num_examples, noise_variance=1.0))


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
    """TODO: This is slow. Why?"""

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
            fig=self.fig,
            func=self.update,
            frames=np.linspace(0, 2 * np.pi, 128),
            init_func=self.init,
            blit=True)

        plt.pause(3.0)


@pytest.mark.skip("sine wave animation sample")
class TestSineWaveAnimation(object):

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
        # plt.show()


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


@pytest.fixture
def a_network():
    nodes = nn.build_network(
        network_shape=[2, 3, 2],
        activation=nn.ReluActivationFunction(),
        output_activation=nn.ReluActivationFunction(),
        regularization=nn.L2RegularizationFunction(),
        input_ids=['x1', 'x2'],
        init_zero_q=False,
    )
    return nodes


@pytest.fixture
def some_classify_circle_data(num_examples):
    """TODO: Parametrize noise."""
    data = classify_circle_data(num_examples, noise=0.25)
    result = classify_circle_train_test_split(num_examples, data)
    return result


def test_build_network(a_network):
    assert a_network is not None


def get_loss(network: List[List[nn.Node]], data: List[Example2D]) -> float:
    temp = 0
    ef = nn.SquareErrorFunction()
    for d in data:
        output = nn.forward_prop(network, [d.x, d.y])
        temp += ef.error(output, d.label)
    result = temp / len(data)
    return result


def classify_circle_train_one_step(
        network: List[List[nn.Node]],
        some_classify_circle_data) -> Dict:
    training_data = some_classify_circle_data['training data']
    testing_data = some_classify_circle_data['testing data']
    l = len(training_data)
    i = 0
    for d in training_data:
        nn.forward_prop(network, [d.x, d.y])
        nn.back_prop(network, d.label, nn.SquareErrorFunction())
        i += 1
        if i % BATCH_SIZE == 0 or i == l:
            nn.update_weights(network, LEARNING_RATE, REGULARIZATION_RATE)
    result = {'training loss': get_loss(network, training_data),
              'testing lost': get_loss(network, testing_data)}
    return result


from functools import partial


# @pytest.mark.skip("network loss animation")
class TestNetworkLossAnimation(object):

    def data_gen(selfk, network, some_classify_circle_data, t=0):
        cnt = 0
        while cnt < 1000:
            cnt += 1
            t += 0.1
            losses = \
                classify_circle_train_one_step(
                    network,
                    some_classify_circle_data)
            # yield t, np.sin(2 * np.pi * t) * np.exp(-t / 10.)
            yield t, losses['training loss']

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

    def test_another_animation(self, a_network, some_classify_circle_data):
        import matplotlib.animation as animation
        self.fig, self.ax = plt.subplots()
        self.xdata, self.ydata = [], []
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.grid()

        # Must assign result to a variable lest nothing shows
        ani = animation.FuncAnimation(
            fig=self.fig,
            func=self.run,
            frames=partial(self.data_gen, a_network, some_classify_circle_data),
            blit=False,
            interval=10,
            repeat=False,
            init_func=self.init
        )
        plt.pause(3.0)
        # plt.show()


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
