"""
This file is a Python port of tf_ufp.ts from Google. Here is the original
copyright.

/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
"""


import numpy as np
from typing import List, Callable, Any
import uuid


def testable() -> bool:
    """Bootstrap unit tests; remove eventually."""
    return True


# Error


def square_error(output: float, target: float) -> float:
    e = output - target
    result = 0.5 * e * e
    return result


def square_error_der(output: float, target: float) -> float:
    result = output - target
    return result


class ErrorFunction(object):

    def __init__(self,
                 error: Callable[[float, float], float],
                 der: Callable[[float, float], float]):
        self.error = error
        self.der = der


class SquareErrorFunction(ErrorFunction):

    def __init__(self):
        super().__init__(square_error, square_error_der)


# Activation


def tanh(x: float) -> float:
    result = np.tanh(x)
    return result


def d_tanh(x: float) -> float:
    t = np.tanh(x)
    result = 1 - t * t
    return result


def relu(x: float) -> float:
    result = np.max([0, x])
    return result


def d_relu(x: float) -> float:
    result = 0 if x <= 0 else 1
    return result


class ActivationFunction(object):

    def __init__(self,
                 output: Callable[[float], float],
                 der: Callable[[float], float]):
        self.output = output
        self.der = der


class TanhActivationFunction(ActivationFunction):

    def __init__(self):
        super().__init__(tanh, d_tanh)


class ReluActivationFunction(ActivationFunction):

    def __init__(self):
        super().__init__(relu, d_relu)


def sigmoid(x):
    return 1.0 / (1.0 - np.exp(-x))


def d_sigmoid(x):
    output = sigmoid(x)
    result = output * (1 - output)
    return result


class SigmoidActivationFunction(ActivationFunction):

    def __init__(self):
        super().__init__(sigmoid, d_sigmoid)


class LinearActivationnFunction(ActivationFunction):

    def __init__(self):
        super().__init__(
            output = lambda x: x,
            der = lambda _: 1
        )


# Regularization


def l1(x: float) -> float:
    result = np.abs(x)
    return result


def d_l1(x: float) -> float:
    if x < 0:
        result = -1
    elif x > 0:
        result = 1
    else:
        result = 0
    return result


def l2(x: float) -> float:
    result = 0.5 * x * x
    return result


def d_l2(x: float) -> float:
    return x


class RegularizationFunction(object):

    def __init__(self,
                 output: Callable[[float], float],
                 der: Callable[[float], float]):
        self.output = output
        self.der = der


class L1RegularizationFunction(RegularizationFunction):

    def __init__(self):
        super().__init__(l1, d_l1)


class L2RegularizationFunction(RegularizationFunction):

    def __init__(self):
        super().__init__(l2, d_l2)


# Node


class Node(object):
    """ A node in a neural network. Each node has a state
    (total input, output, and their respective derivatives),
    which changes after every forward and back propagation run.
    """

    def __init__(self,
                 id: str,
                 activation: ActivationFunction,
                 init_zero_q: bool,
                 ):
        self.id = id
        self.input_links = []
        self.outputs = []
        self.total_input = 0.0
        self.output = 0.0
        self.output_der = 0
        self.input_der = 0
        # Accumulated error derivative with respect to this node's total
        # input since the last update. This derivative equals dE/db where b
        # is the node's bias term.
        self.acc_input_der = 0
        # Number of accumulated err. derivatives with respect to the total
        # input since the last update.
        self.num_accumulated_ders = 0
        # Activation function that takes total input and returns output
        self.activation = activation
        self.bias = 0 if init_zero_q else 0.1

    def update_output(self) -> float:
        """Recompute and return node's output."""
        self.total_input = self.bias
        # inner product / dot product:
        for link in self.input_links:
            self.total_input += link.weight * link.source.output
        result = self.activation.output(self.total_input)
        self.output = result
        return result


# Link


class Link(object):
    """A link in a neural network. Each link has a weight and a source and
    destination node. Also it has an internal state (error derivative
    with respect to a particular input) which gets updated after
    a run of back propagation (TODO and update_weights?)."""

    def __init__(self,
                 source: Node,
                 dest: Node,
                 regularization: RegularizationFunction,
                 init_zero_q: bool,
                 ):
        self.id = source.id + '-' + dest.id
        self.source = source
        self.dest = dest
        self.weight = 0 if init_zero_q else np.random.random() - 0.5
        self.is_dead = False
        #   Error derivative with respect to this weight. */
        self.error_der = 0.0
        #   Accumulated error derivative since the last update. */
        self.acc_error_der = 0
        #   Number of accumulated derivatives since the last update. */
        self.num_accumulated_ders = 0
        self.regularization = regularization


# Build and Operate the Network


def build_network(network_shape: List[int],
                  activation: ActivationFunction,
                  output_activation: ActivationFunction,
                  regularization: RegularizationFunction,
                  input_ids: List[str],
                  init_zero_q: bool,  # in original; doesn't seem necessary
                  ) -> List[List[Node]]:
    """ Builds a neural network.
    @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
      the network will have one input node, 2 nodes in first hidden layer,
      3 nodes in second hidden layer and 1 output node.
    @param activation The activation function of every hidden node.
    @param outputActivation The activation function for the output nodes.
    @param regularization The regularization function that computes a penalty
        for a given weight (parameter) in the network. If None,
        there will be no regularization.
    @param inputIds List of ids for the input nodes. """
    num_layers = len(network_shape)
    id = 1
    # List of layers, with each layer a list of nodes. */
    network = []
    for layer_idx in range(num_layers):
        is_output_layer = layer_idx == num_layers - 1
        is_input_layer = layer_idx == 0
        current_layer = []
        network.append(current_layer)
        num_nodes = network_shape[layer_idx]
        for i in range(num_nodes):
            node_id = str(id)
            if is_input_layer:
                node_id = input_ids[i]
            else:
                id += 1
            node = Node(
                id=node_id,
                activation=(
                    output_activation if is_output_layer else activation),
                init_zero_q=init_zero_q)
            current_layer.append(node)
            if layer_idx >= 1:
                for j in range(len(network[layer_idx - 1])):
                    prev_node = network[layer_idx - 1][j]
                    link = Link(source=prev_node,
                                dest=node,
                                regularization=regularization,
                                init_zero_q=init_zero_q)
                    prev_node.outputs.append(link)
                    node.input_links.append(link)
    return network


def forward_prop(network: List[List[Node]], inputs: List[float]) -> float:
    """ Runs a forward propagation of the input through the
    network. Modifies the internal state of the network - the
    total input and output of each node in the network.

    @param network The neural network.
    @param inputs The input array. Its length must match the
        number of input nodes in the network.
    @return The final output of the network."""
    input_layer = network[0]
    if len(inputs) != len(input_layer):
        raise TypeError(f"""Len inputs {len(inputs)} must match Len first 
        layer of the network {len(input())}""")
    # Let the outputs of the input layer of nodes be the inputs given to this
    #  forward-prop function.
    for i in range(len(input_layer)):
        node = input_layer[i]
        node.output = inputs[i]
    for layer_idx in range(1, len(network)):
        current_layer = network[layer_idx]
        for i in range(len(current_layer)):
            node = current_layer[i]
            node.update_output()  # memoize in the output
    result = network[len(network) - 1][0].output
    return result


def back_prop(network: List[List[Node]],
              target: float,
              error_func: ErrorFunction):
    """ Runs a backward propagation using the provided target and the
    computed output of the previous call to forward propagation.
    This method modifies the internal state of the network - the error
    derivatives with respect to each node, and each weight
    in the network."""
    output_node = network[len(network) - 1][0]
    # The output node is a special case. Use the user-defined error
    # function for the numerical value of the derivative.
    output_node.output_der = error_func.der(output_node.output, target)
    # Go through the layers backwards:
    for layer_idx in range(len(network) - 1, 0, -1):  # skips layer 0
        current_layer = network[layer_idx]
        # TODO: Refactor as for layer in network[backward_slice_op]:
        # Compute error derivative of each node with respect to:
        # 1) its total input
        # 2) each input weight
        for i in range(len(current_layer)):  # vertically
            node = current_layer[i]
            # where did output_der get set up? Ten lines above.
            node.input_der = node.output_der * node.activation.der(
                node.total_input)  # chain rule?
            node.acc_input_der += node.input_der  # linear combination of inputs
            node.num_accumulated_ders += 1
        # Error derivative with respect to each weight coming into the node.
        for i in range(len(current_layer)):
            node = current_layer[i]
            for j in range(len(node.input_links)):
                # TODO: Refactor as for link in node.input_links
                link = node.input_links[j]
                if link.is_dead:
                    continue
                link.error_der = node.input_der * link.source.output
                link.acc_error_der += link.error_der
                link.num_accumulated_ders += 1
        if layer_idx == 1:
            # Don't backprop into the input layer; it has no incoming weights
            continue
        prev_layer = network[layer_idx - 1]
        for i in range(len(prev_layer)):
            # TODO: Refactor as for node in prev_layer
            node = prev_layer[i]
            node.output_der = 0
            for j in range(len(node.outputs)):
                # TODO: Refactor as for output in node.outputs
                output = node.outputs[j]
                node.output_der += output.weight * output.dest.input_der


def update_weights(network: List[List[Node]],
                   learning_rate: float,
                   regularization_rate: float):
    """Updates the weights of the network using the previously accumulated
    error derivatives."""
    for layer_idx in range(len(network)):
        current_layer = network[layer_idx]
        for i in range(len(current_layer)):
            node = current_layer[i]
            if node.num_accumulated_ders > 0:
                node.bias -= \
                    learning_rate * node.acc_input_der / node.acc_input_der
                node.acc_input_der = 0
                node.num_accumulated_ders = 0
            for j in range(len(node.input_links)):
                link = node.input_links[j]
                if link.is_dead:
                    continue
                regul_der = link.regularization.der(link.weight) \
                    if link.regularization is not None else 0  # TODO: right?
                # Update the weight based on dE/dw.
                if link.num_accumulated_ders > 0:
                    link.weight -= \
                        (learning_rate / link.num_accumulated_ders) * \
                        link.acc_error_der  # updated in 'backprop'
                new_link_weight = link_weight - \
                    (learning_rate * regularization_rate) * regul_der
                if link.regularization.output is l1 and \
                        link.weight * new_link_weight < 0:
                    # The weight crossed 0 due to the regularization term. Set
                    #  it to 0 and kill the link.
                    link.weight = 0
                    link.is_dead = True
                else:
                    link.weight = new_link_weight
                link.acc_error_der = 0
                link.num_accumulated_ders = 0


def for_each_node(network: List[List[Node]], ignore_inputs: bool,
                  visitor: Callable[[Node], Any]):
    rg = range(1 if ignore_inputs else 0, len(network))
    # TODO: Refactor as a slice, for layer in slice of network
    for layer_idx in rg:
        current_layer = network[layer_idx]
        for i in range(len(current_layer)):
            # TODO: Refactor as for node in current_layer
            node = current_layer[i]
            visitor(node)


def get_output_node(network: List[List[Node]], which=0):
    return network[-1][which]
