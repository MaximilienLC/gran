import numpy as np
import random

from nets.dynamic.base import DynamicNetBase
from utils.functions.misc import one_percent_change
from utils.functions.spatial import BiasVector, WeightMatrix, compute_input_space_dimensions, compute_nb_input_spaces
from utils.functions.spatial import generate_pos, node_dict, reshape_inputs, reshape_outputs

class Net(DynamicNetBase):

    def __init__(self, d_input, d_output=None):

        self.d_input = d_input
        self.d_output = d_input if d_output == None else d_output

        self.nb_input_spaces = compute_nb_input_spaces(self.d_input)

        self.height, self.width = d_input

        self.nodes = node_dict(self.nb_input_spaces, self.d_output)

        self.nb_passes = 1

        self.weights = WeightMatrix()
        self.biases = BiasVector()
        
        self.h = np.empty((0,1))

        self.architectural_mutations = [self.grow_node, self.prune_node]

    def initialize_architecture(self):

        self.grow_input_nodes()
        self.grow_node()

    def mutate_parameters(self):

        self.weights.mutate()
        self.biases.mutate()

        self.nb_passes = one_percent_change(self.nb_passes)

    def grow_input_nodes(self):

        for input_space, (height_size, width_size) in enumerate( compute_input_space_dimensions(self.d_input) ):

            for i in range(height_size):
                for j in range(width_size):

                    n_2_x = (j + 0.5) * (self.width / width_size)
                    n_2_y = self.height - (i + 0.5) * (self.height / height_size)
                    n_2_pos = (n_2_x, n_2_y, 0)

                    self.nodes['pos']['all'].append(n_2_pos)
                    self.nodes['pos']['input'].append(n_2_pos)

                    self.nodes['pos']['in'][n_2_pos] = []
                    self.nodes['pos']['out'][n_2_pos] = []
                    self.nodes['pos']['connected with to'][n_2_pos] = []

                    self.nodes['input space'][n_2_pos] = input_space

                    self.weights.append_input(n_2_pos)
                    self.biases.append_input(n_2_pos)

                    if input_space >= self.nb_input_spaces - 2:

                        self.nodes['pos']['visible'].append(n_2_pos)
                        self.nodes['pos']['visible inputs'].append(n_2_pos)
                        self.nodes['pos']['visible spaced inputs'][input_space].append(n_2_pos)

                    if self.height != height_size or self.width != width_size:

                        if height_size != width_size:
                            n_0_index = len(self.nodes['pos']['all']) - (height_size*width_size*2)+i*width_size-1
                            n_1_index = len(self.nodes['pos']['all']) - (height_size*width_size*2)+(i+1)*width_size-1
                        else: # height_size == width_size:
                            n_0_index = len(self.nodes['pos']['all']) - (height_size*width_size*2)+(i*height_size+j)-1
                            n_1_index = len(self.nodes['pos']['all']) - (height_size*width_size*2)+(i*height_size+j)

                        n_0_pos, n_1_pos = self.nodes['pos']['all'][n_0_index], self.nodes['pos']['all'][n_1_index]

                        self.nodes['pos']['in'][n_2_pos].extend([n_0_pos, n_1_pos])
                        self.nodes['pos']['out'][n_0_pos].append(n_2_pos)
                        self.nodes['pos']['out'][n_1_pos].append(n_2_pos)

    def grow_node(self):

        n_0_pos = random.choice(self.nodes['pos']['visible'])

        n_1_pos = self.find_node_closest_to(n_0_pos, 'connect with')

        if n_1_pos == None:
            return

        n_2_pos = generate_pos(n_0_pos, n_1_pos)

        self.nodes['pos']['all'].append(n_2_pos)
        self.nodes['pos']['hidden'].append(n_2_pos)
        self.nodes['pos']['hidden & output'].append(n_2_pos)
        self.nodes['pos']['visible'].append(n_2_pos)

        self.nodes['pos']['in'][n_2_pos] = [n_0_pos, n_1_pos]
        self.nodes['pos']['out'][n_0_pos].append(n_2_pos)
        self.nodes['pos']['out'][n_1_pos].append(n_2_pos)
        self.nodes['pos']['out'][n_2_pos] = []
        self.nodes['pos']['connected with to'][n_0_pos].append([n_1_pos, n_2_pos])
        self.nodes['pos']['connected with to'][n_1_pos].append([n_0_pos, n_2_pos])
        self.nodes['pos']['connected with to'][n_2_pos] = []

        self.h = np.vstack((self.h, 0))

        if None in self.nodes['pos']['output']:
            self.swap_hidden_and_output_node(n_2_pos)

        elif n_0_pos in self.nodes['pos']['output']:
            self.swap_hidden_and_output_node(n_2_pos, n_0_pos)

        elif n_1_pos in self.nodes['pos']['output']:
            self.swap_hidden_and_output_node(n_2_pos, n_1_pos)

        self.weights.append(n_0_pos, n_2_pos)
        self.weights.append(n_1_pos, n_2_pos)

        n_3_pos = self.find_node_closest_to(n_2_pos, 'connect to')

        if n_3_pos != None:

            self.weights.append(n_2_pos, n_3_pos)

            self.nodes['pos']['out'][n_2_pos].append(n_3_pos)
            self.nodes['pos']['in'][n_3_pos].append(n_2_pos)

        for in_node_pos in [n_0_pos, n_1_pos]:

            if in_node_pos in self.nodes['pos']['input']:

                for in_node_in_node_pos in self.nodes['pos']['in'][in_node_pos]:

                    if in_node_in_node_pos not in self.nodes['pos']['visible']:

                        self.nodes['pos']['visible'].append(in_node_in_node_pos)
                        self.nodes['pos']['visible inputs'].append(in_node_in_node_pos)
                        input_space = self.nodes['input space'][in_node_in_node_pos]
                        self.nodes['pos']['visible spaced inputs'][input_space].append(in_node_in_node_pos)

        self.biases.append(n_2_pos)

    def prune_node(self, node_pos=None):

        if node_pos == None:

            if len(self.nodes['pos']['hidden & output']) == 0:
                return

            node_pos = random.choice(self.nodes['pos']['hidden & output'])

        if node_pos in self.nodes['pos']['being pruned']:
            return

        self.nodes['pos']['being pruned'].append(node_pos)

        if node_pos in self.nodes['pos']['output']:
            self.swap_hidden_and_output_node(original_output_node_pos=node_pos)

        for i in range( len(self.nodes['pos']['out'][node_pos]) - 1, -1, -1 ):
            self.prune_connection(node_pos, self.nodes['pos']['out'][node_pos][i], node_pos)

        for i in range( len(self.nodes['pos']['in'][node_pos]) - 1, -1, -1 ):
            self.prune_connection(self.nodes['pos']['in'][node_pos][i], node_pos, node_pos)

        self.weights.remove(node_pos)
        self.biases.remove(node_pos)

        self.h = np.delete(self.h, self.nodes['pos']['hidden & output'].index(node_pos), axis=0)

        self.nodes['pos']['all'].remove(node_pos)
        self.nodes['pos']['hidden'].remove(node_pos)
        self.nodes['pos']['hidden & output'].remove(node_pos)
        self.nodes['pos']['visible'].remove(node_pos)
        self.nodes['pos']['being pruned'].remove(node_pos)
        
        del self.nodes['pos']['in'][node_pos]
        del self.nodes['pos']['out'][node_pos]
        del self.nodes['pos']['connected with to'][node_pos]

        if len(self.nodes['pos']['hidden & output']) == 0:
            self.grow_node()

    def prune_connection(self, in_node_pos, out_node_pos, calling_node_pos):

        if not out_node_pos in self.nodes['pos']['out'][in_node_pos]:
            return

        self.nodes['pos']['out'][in_node_pos].remove(out_node_pos)
        self.nodes['pos']['in'][out_node_pos].remove(in_node_pos)

        for in_node_connected_with_pos,in_node_connected_to_pos in self.nodes['pos']['connected with to'][in_node_pos]:
            if in_node_connected_to_pos == out_node_pos:
                nodes_connected_with_to_pos = [in_node_connected_with_pos, in_node_connected_to_pos]
                self.nodes['pos']['connected with to'][in_node_pos].remove(nodes_connected_with_to_pos)

        if calling_node_pos == in_node_pos:

            if out_node_pos in self.nodes['pos']['hidden & output']:

                if len(self.nodes['pos']['in'][out_node_pos]) == 0 :

                    self.prune_node(out_node_pos)

        else: # calling_node_pos == out_node_pos:

            if in_node_pos in self.nodes['pos']['hidden']:

                if len(self.nodes['pos']['out'][in_node_pos]) == 0:

                    self.prune_node(in_node_pos)

            elif in_node_pos in self.nodes['pos']['input']:

                for in_node_out_node_pos in self.nodes['pos']['out'][in_node_pos]:

                    if in_node_out_node_pos in self.nodes['pos']['hidden & output']:

                        return

                for in_node_in_node_pos in self.nodes['pos']['in'][in_node_pos]:

                    if in_node_in_node_pos in self.nodes['pos']['input']:
                    
                        if len(self.nodes['pos']['out'][in_node_in_node_pos]) == 1:
                                
                            self.nodes['pos']['visible'].remove(in_node_in_node_pos)
                            self.nodes['pos']['visible inputs'].remove(in_node_in_node_pos)
                            input_space = self.nodes['input space'][in_node_in_node_pos]
                            self.nodes['pos']['visible spaced inputs'][input_space].remove(in_node_in_node_pos)

    def find_node_closest_to(self, node_pos, action):

        if node_pos in self.nodes['pos']['input']:
            input_space = self.nodes['input space'][node_pos]
            potential_nodes_pos = self.nodes['pos']['visible spaced inputs'][input_space].copy()

        else: # node_pos in self.nodes['pos']['hidden & output']:
            potential_nodes_pos = self.nodes['pos']['hidden & output'].copy()

        if self.d_output == self.d_input and action == 'connect to':
            potential_nodes_pos.extend(self.nodes['pos']['visible inputs'])

        equal_node_pos = np.equal(potential_nodes_pos, node_pos).all(axis=1)
        potential_nodes_pos = np.delete( potential_nodes_pos, np.argwhere(equal_node_pos), axis=0 )

        if action == 'connect with':
            
            for node_connected_with_pos, _ in self.nodes['pos']['connected with to'][node_pos]:
                equal_connected_with = (potential_nodes_pos == node_connected_with_pos).all(axis=1)
                potential_nodes_pos = np.delete( potential_nodes_pos, np.argwhere(equal_connected_with), axis=0 )

        else: # action == 'connect to':

            for out_node_pos in self.nodes['pos']['out'][node_pos]:
                equal_out_node_pos = (potential_nodes_pos == out_node_pos).all(axis=1)
                potential_nodes_pos = np.delete( potential_nodes_pos, np.argwhere(equal_out_node_pos), axis=0 )

        if potential_nodes_pos.size == 0:
            return

        broadcasted_node_pos = np.ones([ len(potential_nodes_pos), 3 ]) * node_pos
        squared_distances = np.sum( (broadcasted_node_pos  - potential_nodes_pos) ** 2, axis=1 )
        closest_nodes_indices = np.argwhere( np.equal( squared_distances, squared_distances.min() ) ).squeeze(1)

        random_closest_node_index = random.choice(closest_nodes_indices)

        return tuple(potential_nodes_pos[random_closest_node_index])

    def swap_hidden_and_output_node(self, original_hidden_node_pos=None, original_output_node_pos=None):

        if original_hidden_node_pos == None:

            original_output_node_in_hidden_nodes_pos = []

            for original_output_node_in_node_pos in self.nodes['pos']['in'][original_output_node_pos]:
                if original_output_node_in_node_pos in self.nodes['pos']['hidden']:
                    original_output_node_in_hidden_nodes_pos.append(original_output_node_in_node_pos)

            if len(original_output_node_in_hidden_nodes_pos) > 0:
                original_hidden_node_pos = random.choice(original_output_node_in_hidden_nodes_pos)

        if original_hidden_node_pos != None:

            self.nodes['pos']['hidden'].remove(original_hidden_node_pos)

        if original_output_node_pos != None:

            self.nodes['pos']['hidden'].append(original_output_node_pos)

            output_index = self.nodes['pos']['output'].index(original_output_node_pos)

        else:

            output_index = random.choice( np.where( np.array(self.nodes['pos']['output']) == None )[0] )

        self.nodes['pos']['output'][output_index] = original_hidden_node_pos

    def setup_to_run(self):

        self.w = self.weights.to_sparse_matrix()
        self.b = self.biases.to_vector()

    def setup_to_save(self):
        
        self.w = None
        self.b = None

    def reset(self):

        self.h = np.zeros(( len(self.nodes['pos']['hidden & output']), 1 ))

    def __call__(self, x):

        X = reshape_inputs(x)

        for _ in range(self.nb_passes):

            x = np.concatenate((X, self.h))

            x = self.w @ x + self.b

            x = np.clip(x, 0, 2**31-1)

            self.h = x[ len(self.nodes['pos']['input']): ]

        if self.d_output == self.d_input:

            out = reshape_outputs( x[ :len(self.nodes['pos']['input']) ] )

        else:

            out = np.zeros(self.d_output)

            for i in range(self.d_output):
                if self.nodes['pos']['output'][i] != None:
                    out[i] = x[ self.nodes['pos']['all'].index(self.nodes['pos']['output'][i]) ]

        return out