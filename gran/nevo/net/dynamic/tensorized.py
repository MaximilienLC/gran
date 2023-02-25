# Copyright 2023 The Gran Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


class Net:
    def __init__(self, d_input_output):
        self.d_input, self.d_output = d_input_output

        self.nodes = {
            "all": [],
            "input": [],
            "hidden": [],
            "output": [],
            "receiving": [],
            "emitting": [],
            "being pruned": [],
        }

        self.architectural_mutations = [self.grow_node, self.prune_node]

        self.weights, self.biases = SparseWeightMatrix(), BiasVector()

    def initialize_architecture(self):
        for _ in range(self.d_input):
            self.grow_node("input")

        for _ in range(self.d_output):
            self.grow_node("output")

    def grow_node(self, type="hidden"):
        new_node = Node()

        self.weights.add_node(new_node)
        self.biases.add_node(new_node, type)

        self.nodes["all"].append(new_node)
        self.nodes[type].append(new_node)

        if type == "input":
            self.nodes["receiving"].append(new_node)

        elif type == "hidden":
            potential_in_nodes = []

            for input_node in self.nodes["input"]:
                if input_node not in self.nodes["emitting"]:
                    potential_in_nodes.append(input_node)

            if potential_in_nodes == []:
                potential_in_nodes = list(dict.fromkeys(self.nodes["receiving"]))

            potential_out_nodes = []

            for output_node in self.nodes["output"]:
                if output_node not in self.nodes["receiving"]:
                    potential_out_nodes.append(output_node)

            if potential_out_nodes == []:
                potential_out_nodes = self.nodes["hidden"] + self.nodes["output"]

            in_node_1 = np.random.choice(potential_in_nodes)
            self.grow_connection(in_node_1, new_node)

            potential_in_nodes.remove(in_node_1)

            if potential_in_nodes == []:
                potential_in_nodes = list(dict.fromkeys(self.nodes["receiving"]))

            in_node_2 = np.random.choice(potential_in_nodes)
            self.grow_connection(in_node_2, new_node)

            out_node = np.random.choice(potential_out_nodes)
            self.grow_connection(new_node, out_node)

    def grow_connection(self, in_node, out_node):
        in_node.out_nodes.append(out_node)
        out_node.in_nodes.append(in_node)

        self.weights.add_connection(in_node, out_node)

        self.nodes["emitting"].append(in_node)
        self.nodes["receiving"].append(out_node)

    def prune_node(self, node=None):
        if node == None:
            if len(self.nodes["hidden"]) == 0:
                return

            node = np.random.choice(self.nodes["hidden"])

        if node in self.nodes["being pruned"]:
            return

        self.weights.remove_node(node)
        self.biases.remove_node(node)

        self.nodes["being pruned"].append(node)

        for out_node in node.out_nodes.copy():
            self.prune_connection(node, out_node, node)

        for in_node in node.in_nodes.copy():
            self.prune_connection(in_node, node, node)

        for key in self.nodes:
            while node in self.nodes[key]:
                self.nodes[key].remove(node)

    def prune_connection(self, in_node, out_node, calling_node):
        if in_node not in out_node.in_nodes:
            return

        in_node.out_nodes.remove(out_node)
        out_node.in_nodes.remove(in_node)

        self.nodes["receiving"].remove(out_node)
        self.nodes["emitting"].remove(in_node)

        if in_node == calling_node:
            if out_node not in self.nodes["receiving"]:
                if out_node in self.nodes["hidden"]:
                    self.prune_node(out_node)

        if out_node == calling_node:
            if in_node not in self.nodes["emitting"]:
                if in_node in self.nodes["hidden"]:
                    self.prune_node(in_node)


class Node:
    def __init__(self):
        self.in_nodes, self.out_nodes = [], []


class SparseWeightMatrix:
    def __init__(self):
        self.row = np.array([])
        self.col = np.array([])
        self.data = np.array([])

        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_connection(self, in_node, out_node):
        in_node_idx = self.nodes.idx(in_node)
        out_node_idx = self.nodes.idx(out_node)

        self.row = np.append(self.row, in_node_idx)
        self.col = np.append(self.col, out_node_idx)
        self.data = np.append(self.data, np.random.randn())

    def remove_node(self, node):
        node_idx = self.nodes.idx(node)

        self.nodes.remove(node)

        self.data = np.delete(self.data, np.where(self.row == node_idx))
        self.col = np.delete(self.col, np.where(self.row == node_idx))
        self.row = np.delete(self.row, np.where(self.row == node_idx))

        self.data = np.delete(self.data, np.where(self.col == node_idx))
        self.row = np.delete(self.row, np.where(self.col == node_idx))
        self.col = np.delete(self.col, np.where(self.col == node_idx))

        self.row -= self.row > node_idx
        self.col -= self.col > node_idx


class BiasVector:
    def __init__(self):
        self.data = np.array([])

        self.nodes = []

    def add_node(self, node, type):
        self.nodes.append(node)

        self.data = np.append(self.data, np.random.randn() if type == "hidden" else 0)

    def remove_node(self, node):
        node_idx = self.nodes.idx(node)

        self.nodes.remove(node)

        self.data = np.delete(self.data, node_idx)
