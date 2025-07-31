# Reference: GDT.py in the original paper repo
import numpy as np
import random
from copy import deepcopy
import graphviz
from IPython.display import Image
from IPython.display import display, clear_output

class DecisionTree:
    def __init__(self, split_thresholds, one_hot_encodings, class_probabilities):
        self.split_thresholds = split_thresholds
        self.one_hot_encodings = one_hot_encodings
        self.class_probabilities = class_probabilities
        self.max_depth = int(np.log2(len(class_probabilities)))
        self.root = self.build_tree()

    def build_tree(self, node_index=0):
        # If node index is greater than the number of internal nodes, we have reached a leaf node
        if node_index >= len(self.split_thresholds):
            return {'class': self.class_probabilities[node_index - len(self.split_thresholds)]}

        # Otherwise, create a new node with the corresponding threshold value and recursively build its children
        threshold_values = self.split_thresholds[node_index]
        one_hot_encoding = self.one_hot_encodings[node_index]
        node = {'threshold_values': threshold_values, 'one_hot_encoding': one_hot_encoding}
        node['left'] = self.build_tree(2 * node_index + 1)
        node['right'] = self.build_tree(2 * node_index + 2)

        return node

    def prune_tree(self, data, min_samples):
        if min_samples < 1: #int(min_samples) - min_samples != 0: #if float number
            min_samples = max(1, data.shape[0] * min_samples)
        
        data_complete = deepcopy(data)
        
        self._pass_node(self.root, data)
        
        self.root_unpruned = deepcopy(self.root)
        self._prune_node(self.root, data, data_complete, min_samples)

    def _pass_node(self, node, data):
        
        node['num_samples_passed'] = data.shape[0]
        #print(node)
        if 'class' in node:
            return

        left_indices = np.where(data[:, np.argmax(node['one_hot_encoding'])] <= node['threshold_values'][np.argmax(node['one_hot_encoding'])])[0]
        right_indices = np.where(data[:, np.argmax(node['one_hot_encoding'])] > node['threshold_values'][np.argmax(node['one_hot_encoding'])])[0]
            
        self._pass_node(node['left'], data[left_indices])
        self._pass_node(node['right'], data[right_indices])        
        
        
    def _prune_node(self, node, data, data_complete, min_samples):
        
        # If the node is a leaf, return
        if 'class' in node:
            return

        # Recursively prune the children of the node
        self._prune_node(node['left'], data, data_complete, min_samples)
        self._prune_node(node['right'], data, data_complete, min_samples)

        # Prune the node if the number of samples passing through it is less than the minimum
        
        samples_left = node['left']['num_samples_passed']
        samples_right = node['right']['num_samples_passed']
        
        #print('node', node)
        #print('node[left]', node['left'])
        
        if samples_left < min_samples and samples_right < min_samples:
            #print(node)
            if 'class' in node['left'] and 'class' in node['right']:
                #print(node['left']['class'], node['right']['class'])
                node['class'] = np.mean([node['left']['class'], node['right']['class']], axis=0)
                #print('node 1', node['class'])
                node['num_samples_passed'] = np.sum([node['left']['num_samples_passed'], node['right']['num_samples_passed']])
                node.pop('left', None)
                node.pop('right', None) 
            else:
                print('SHOULD NOT HAPPEN, CHECK PLEASE')
                return       

        else:
            if samples_left < min_samples:
                #print('node[left]', node['left'])
                #print('node[right]', node['right'])   
                if 'class' in node['right']:
                    node['class'] = node['right']['class']
                    #print('node 2', node['class'])
                    node['num_samples_passed'] = node['right']['num_samples_passed']
                    node.pop('left', None)
                    node.pop('right', None) 
                else:
                    new_node = deepcopy(node['right'])
                    node['left'] = new_node['left']
                    node['right'] = new_node['right']
                    node['one_hot_encoding'] = new_node['one_hot_encoding']
                    node['threshold_values'] = new_node['threshold_values']                
            elif samples_right < min_samples:
                #print('node[left]', node['left'])
                #print('node[right]', node['right'])
                if 'class' in node['left']:
                    node['class'] = node['left']['class']
                    #print('node 3', node['class'])
                    node['num_samples_passed'] = node['left']['num_samples_passed']
                    node.pop('left', None)
                    node.pop('right', None)
                else:
                    new_node = deepcopy(node['left'])
                    node['left'] = new_node['left']
                    node['right'] = new_node['right']
                    node['one_hot_encoding'] = new_node['one_hot_encoding']
                    node['threshold_values'] = new_node['threshold_values']
            else:
                return
        self._pass_node(self.root, data_complete)
                
    def predict(self, instance, node=None):
        # If no starting node is specified, start at the root of the tree
        if node is None:
            node = self.root

        # If we have reached a leaf node, return the corresponding class probability
        if 'class' in node:
            return node['class']

        # Otherwise, compare the instance's feature values to the node's threshold values and traverse the appropriate child
        threshold_values = node['threshold_values']
        one_hot_encoding = node['one_hot_encoding']
        feature_values = instance[one_hot_encoding == 1]
        if np.all(feature_values <= threshold_values):
            return self.predict(instance, node['left'])
        else:
            return self.predict(instance, node['right'])

    def evaluate(self, test_data, true_labels):
        num_correct = 0
        for i in range(len(test_data)):
            prediction = self.predict(test_data[i])
            if prediction == true_labels[i]:
                num_correct += 1
        accuracy = num_correct / len(test_data)
        return accuracy
    
    def extend_to_fully_grown(self):
        self.split_thresholds_unpruned = deepcopy(self.split_thresholds)
        self.one_hot_encodings_unpruned = deepcopy(self.one_hot_encodings)
        self.class_probabilities_unpruned = deepcopy(self.class_probabilities)
        self.root_pruned_extended = deepcopy(self.root)
        
        current_node_list = [self.root_pruned_extended]
        for current_depth in range(self.max_depth):
            current_node_list_new = []
            for current_node in current_node_list:
                if 'class' in current_node:
                    current_node_copy = deepcopy(current_node)
                    current_node.pop('class', None)
                    current_node['threshold_values'] = np.zeros_like(self.split_thresholds_unpruned[0])
                    current_node['one_hot_encoding'] = np.zeros_like(self.one_hot_encodings_unpruned[0])
                    current_node['left'] = current_node_copy
                    current_node['right'] = current_node_copy
                    
                current_node_list_new.append(current_node['left'])
                current_node_list_new.append(current_node['right'])
            current_node_list = current_node_list_new
            
        self.to_array_representation(root_type='pruned')
            

    def plot_tree_from_array(self, filename='./tree_tmp.png', plot_format='png'):
        dot = graphviz.Digraph()
        dot.node('0', 'Root')
        self._plot_subtree_from_array(dot, 0)
        dot.render(filename, format=plot_format, view=True)

    def _plot_subtree_from_array(self, dot, node_index):
        if node_index >= len(self.split_thresholds):
            node_label = f'Class: {self.class_probabilities[node_index - len(self.split_thresholds)]}'
        else:
            node_label = f'Feature {self.one_hot_encodings[node_index].argmax()}: <= {self.split_thresholds[node_index]}'
            left_child_index = 2 * node_index + 1
            right_child_index = 2 * node_index + 2
            dot.node(str(left_child_index), '')
            dot.node(str(right_child_index), '')
            dot.edge(str(node_index), str(left_child_index), 'True')
            dot.edge(str(node_index), str(right_child_index), 'False')
            self._plot_subtree_from_array(dot, left_child_index)
            self._plot_subtree_from_array(dot, right_child_index)
        dot.node(str(node_index), node_label)
        
    def plot_tree(self, filename='./tree_tmp', plot_format='png', root_type='current',
                  feature_names=None,scales=None,means=None): #initial, pruned_extended
        dot = graphviz.Digraph()
        if root_type == 'current':
            self._plot_subtree(dot, self.root,feature_names,scales=scales,means=means)
        elif root_type == 'initial':
            self._plot_subtree(dot, self.root_unpruned,feature_names,scales=scales,means=means)
        elif root_type == 'pruned_extended':
            self._plot_subtree(dot, self.root_pruned_extended,feature_names,scales=scales,means=means)
        else:
            print('Root type ' + root_type + ' not existing, taking current root')
            self._plot_subtree(dot, self.root,feature_names,scales=scales,means=means)
            
        dot.render(filename, format=plot_format, view=False)
        display(dot)
        #dot.render(filename, view=True)

    def _plot_subtree(self, dot, node, feature_names=None,scales=None,means=None):
        if 'class' in node:
            class_value = node["class"]
            class_label = 1 if class_value > 0 else 0
            num_samples_passed = node["num_samples_passed"]
            # node_label = f'Class: {class_value:.3f} Num Samples: {num_samples_passed:.0f}'
            node_label = f'Class: {class_label}; Num Samples: {num_samples_passed:.0f}'
        else:
            scales = np.ones_like(node['threshold_values']) if scales is None else scales
            means = np.zeros_like(node['threshold_values']) if means is None else means

            feature_index = node['one_hot_encoding'].argmax()
            threshold_value = node["threshold_values"][feature_index]
            threshold_value_unscaled = threshold_value * scales[feature_index] + means[feature_index]
            num_samples_passed = node["num_samples_passed"]
            # node_label = f'Feature {feature_index}: <= {threshold_value:.3f} Num Samples: {num_samples_passed:.0f}'
            feature_name = feature_names[feature_index] if feature_names is not None else feature_index
            # node_label = f'{feature_name}: <= {threshold_value:.3f} Num Samples: {num_samples_passed:.0f}'
            node_label = f'{feature_name}: <= {threshold_value_unscaled:.3f} Num Samples: {num_samples_passed:.0f}'
            left_child = node['left']
            right_child = node['right']
            dot.node(str(id(left_child)), '')
            dot.node(str(id(right_child)), '')
            dot.edge(str(id(node)), str(id(left_child)), 'True')
            dot.edge(str(id(node)), str(id(right_child)), 'False')
            self._plot_subtree(dot, left_child,feature_names,scales,means)
            self._plot_subtree(dot, right_child,feature_names,scales,means)
        dot.node(str(id(node)), node_label)
        
    def to_array_representation(self, root_type='pruned'): #, 'pruned'
        split_thresholds = []
        one_hot_encodings = []
        class_probabilities = []
        if root_type == 'initial':
            node_queue = [self.root_unpruned]
        elif root_type == 'pruned':
            try:
                node_queue = [self.root_pruned_extended]
            except:
                self.extend_to_fully_grown()
                node_queue = [self.root_pruned_extended]
        while node_queue:
            node = node_queue.pop(0)

            if 'class' in node:
                class_probabilities.append(node['class'])
            else:
                split_thresholds.append(node['threshold_values'])
                one_hot_encoding = np.zeros(len(node['threshold_values']), dtype=np.int)
                one_hot_encoding[node['one_hot_encoding'].argmax()] = 1
                one_hot_encodings.append(one_hot_encoding)

                node_queue.append(node['left'])
                node_queue.append(node['right'])

        self.split_thresholds = np.array(split_thresholds)
        self.one_hot_encodings = np.array(one_hot_encodings)
        self.class_probabilities = np.array(class_probabilities)

        return self.split_thresholds, self.one_hot_encodings, self.class_probabilities
    
    def count_nodes(self, node=None):
        if node is None:
            node = self.root
            if 'class' in node:
                leaf = 1
            else:
                internal = 1
        
        if 'class' in node:
            return 0, 1#1, 0

        left_internal, left_leaf = self.count_nodes(node['left'])
        right_internal, right_leaf = self.count_nodes(node['right'])
        internal = left_internal + right_internal
        leaf = left_leaf + right_leaf
        if 'left' in node or 'right' in node:
            internal += 1

        return internal, leaf