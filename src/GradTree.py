# This code is implemented based on the tensorflow version of the public code of the paper:
# Marton, Sascha, et al. "GradTree: Learning axis-aligned decision trees with gradient descent." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 13. 2024.

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15


class GradTreeBlock(nn.Module):
    def __init__(self, depth, n_estimators, n_features, random_seed, objective, device, number_of_classes=None, from_logits=False):
        super(GradTreeBlock, self).__init__()

        self.depth = depth
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.random_seed = random_seed
        self.objective = objective
        self.number_of_classes = number_of_classes
        self.from_logits = from_logits

        torch.manual_seed(self.random_seed)

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        # Path and internal node index computation
        self.path_identifier_list = []
        self.internal_node_index_list = []
        for leaf_index in range(self.leaf_node_num_):
            for current_depth in range(1, self.depth + 1):
                path_identifier = (leaf_index // (2 ** (self.depth - current_depth))) % 2
                internal_node_index = (
                    2 ** (current_depth - 1) + leaf_index // (2 ** (self.depth - (current_depth - 1))) - 1
                )
                self.path_identifier_list.append(path_identifier)
                self.internal_node_index_list.append(internal_node_index)

        self.path_identifier_list = torch.tensor(self.path_identifier_list).view(-1, self.depth).float().to(device)
        self.internal_node_index_list = torch.tensor(self.internal_node_index_list).view(-1, self.depth).long().to(device)

        # Initialize weights
        self.split_values = nn.Parameter(torch.empty((self.n_estimators, self.internal_node_num_, self.n_features)))
        self.split_index_array = nn.Parameter(torch.empty((self.n_estimators, self.internal_node_num_, self.n_features)))

        if self.objective in ['binary', 'regression']:
            leaf_classes_array_shape = (self.n_estimators, self.leaf_node_num_)
        else:  # classification
            leaf_classes_array_shape = (self.n_estimators, self.leaf_node_num_, self.number_of_classes)

        self.leaf_classes_array = nn.Parameter(torch.empty(leaf_classes_array_shape))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.split_values, mean=0.0, std=0.1)
        nn.init.normal_(self.split_index_array, mean=0.0, std=0.1)
        nn.init.normal_(self.leaf_classes_array, mean=0.0, std=0.1)

    def forward(self, inputs):
        # Expand dimensions for estimators
        X_estimator = inputs.unsqueeze(1)

        # Softmax transformation for split index array
        split_index_array = entmax15(self.split_index_array, dim=-1)

        hardmax = torch.nn.functional.one_hot(split_index_array.argmax(dim=-1), num_classes=split_index_array.size(-1))
        adjust_constant = (split_index_array - hardmax).detach()
        split_index_array = split_index_array - adjust_constant

        # Generate selected values for split and path traversal
        split_index_array_selected = split_index_array[:, self.internal_node_index_list, :]
        split_values_selected = self.split_values[:, self.internal_node_index_list, :]

        # Compute node results
        s1_sum = torch.einsum("eldn,eldn->eld", split_values_selected, split_index_array_selected)
        s2_sum = torch.einsum("ben,eldn->beld", X_estimator, split_index_array_selected)
        node_result = (torch.tanh(s1_sum - s2_sum) + 1) / 2

        # Hard decision with straight-through operator
        node_result_corrected = node_result - (node_result - torch.round(node_result)).detach()

        # Path reduction
        p = torch.prod(
            ((1 - self.path_identifier_list) * node_result_corrected + self.path_identifier_list * (1 - node_result_corrected)),
            dim=3,
        )

        # Raw prediction
        if self.objective == 'regression':
            layer_output = torch.einsum('el,bel->be', self.leaf_classes_array, p)
        elif self.objective == 'binary':
            layer_output = torch.einsum('el,bel->be', self.leaf_classes_array, p)
            if not self.from_logits:
                layer_output = torch.sigmoid(layer_output)
        elif self.objective == 'classification':
            layer_output = torch.einsum('elc,bel->bec', self.leaf_classes_array, p)
            if not self.from_logits:
                layer_output = F.softmax(layer_output, dim=-1)
        else:
            raise NotImplementedError

        if self.objective in ['regression']:
            result = torch.sum(layer_output, dim=1)
        elif self.objective in ['binary']:
            result = torch.mean(layer_output, dim=1)
        else:
            result = torch.sum(layer_output, dim=2)

        if self.objective in ['regression', 'binary']:
            result = result.unsqueeze(1)

        return result


if __name__ == "__main__":
    batch_size = 64
    input_size = 20
    input = torch.randn(batch_size, input_size)
    grad_tree = GradTreeBlock(depth = 3, n_estimators = 1, n_features = input_size, objective = "binary", random_seed=0)

    output = grad_tree.forward(input)

