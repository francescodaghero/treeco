from bigtree import (
    BinaryNode,
    preorder_iter,
    levelorder_iter,
    levelordergroup_iter,
)
from typing import Optional, Callable
from typing import Literal, Tuple
from typing import Union
from bigtree import shift_nodes
from typing import Mapping
import numpy as np
from treeco.utils import quantize


def _get_max_value(dtype):
    if "int" in dtype.name:
        return dtype.type(np.iinfo(dtype).max)
    elif "float" in dtype.name:
        return dtype.type(np.finfo(dtype).max)


def _get_min_value(dtype):
    if "int" in dtype.name:
        return dtype.type(np.iinfo(dtype).min)
    elif "float" in dtype.name:
        return dtype.type(np.finfo(dtype).min)


def _get_fake_threshold(mode, value):
    if mode == "BRANCH_LEQ":
        val = _get_max_value(value.dtype)
    elif mode == "BRANCH_GEQ":
        val = _get_min_value(value.dtype)
    else:
        raise NotImplementedError("Only LEQ and GEQ are supported")
    return val


class Node(BinaryNode):
    """
    Base class for a tree and a single node. All methods should be called from the root
    element.
    Extends the bigtree BinaryNode class.

    """

    def __init__(
        self,
        name: str,
        targets_ids: np.ndarray,
        targets_weights: np.ndarray,
        nodes_featureids: np.number,
        nodes_values: np.number,
        nodes_hitrates: np.number,
        nodes_modes: Union[str, np.object_],
        **kwargs,
    ):

        super().__init__(name, **kwargs)
        # One or more
        self.targets_ids = targets_ids
        self.targets_weights = targets_weights

        # A single value per node
        self.nodes_featureids = nodes_featureids
        self.nodes_values = nodes_values
        self.nodes_hitrates = nodes_hitrates
        self.nodes_modes = nodes_modes

        # Indexes are re-computed dynamically, they should be
        # exported from the method .node_idx_to_dict

        self._additional_info: Mapping = kwargs

    def is_perfect(self) -> bool:
        depths = list()
        for leaf in self.leaves:
            depths.append(leaf.depth)
        return len(set(depths)) == 1

    def is_oblivious(self) -> bool:
        feature_at_depth = {}
        for node in levelorder_iter(self):
            if not node.is_leaf:
                if node.depth not in feature_at_depth:
                    feature_at_depth[node.depth] = node.nodes_featureids
                elif feature_at_depth[node.depth] != node.nodes_featureids:
                    return False
        return True

    @property
    def n_nodes(self) -> int:
        return len([*preorder_iter(self)])

    @property
    def n_leaves(self) -> int:
        return len([*self.leaves])

    @property
    def leaf_shape(self) -> int:
        """
        The shape of the leaf weights (i.e. the number of classes or a single value)

        Returns
        -------
        int
            The leaf shape
        """
        return next(self.leaves).targets_weights.shape[-1]

    @property
    def n_targets(self) -> int:
        """
        Returns the number of targets in the tree, i.e. the number of classes.

        Returns
        -------
        int
            The number of targets
        """
        tgts = set()
        for leaf in self.leaves:
            tgts = tgts.union(tgts, set(leaf.targets_ids))
        return tgts

    def predict(self, x) -> np.ndarray:
        """
        Iterative prediction function for a tree.
        It works with a batch_size = 1.

        Parameters
        ----------
        X : np.ndarray
            The input data
        Returns
        -------
        np.ndarray
            The predictions with shape (1, output_length)
        """
        if self.is_leaf:
            return self.targets_weights
        else:
            if x[self.nodes_featureids] <= self.nodes_values:
                return self.left.predict(x)
            else:
                return self.right.predict(x)

    @classmethod
    def _from_node(cls, name: str, node: "Node", dummy: bool = False) -> "Node":
        # Return an identical node, but without parents/children
        if dummy:
            nodes_modes = "DUMMY_" + node.nodes_modes
        else:
            nodes_modes = node.nodes_modes
        is_dummy_node = dummy and "BRANCH" in nodes_modes

        nodes_values = (
            node.nodes_values
            if not is_dummy_node
            else _get_fake_threshold(node.nodes_modes, node.nodes_values)
        )
        return cls(
            name=name,
            targets_ids=node.targets_ids,
            targets_weights=node.targets_weights,
            nodes_featureids=node.nodes_featureids,
            nodes_values=nodes_values,
            nodes_hitrates=node.nodes_hitrates,
            nodes_modes=nodes_modes,
        )

    # Quantization methods
    def quantize_nodes_values(self, precision: int, min_val: float, max_val: float):
        for node in preorder_iter(self):
            if node.is_leaf:
                node.nodes_values = node.nodes_values.astype(
                    np.dtype(f"uint{precision}")
                )
            else:
                node.nodes_values, _, _ = quantize(
                    data=node.nodes_values,
                    min_val=min_val,
                    max_val=max_val,
                    precision=precision,
                )

    def round_nodes_values(self, precision: int) -> None:
        """
        Round the nodes values to the nearest integer.
        Used for pre-training quantization.

        Parameters
        ----------
        precision : int
            The number of bits of the output value

        Raises
        ------
        NotImplementedError
            If the node mode is not LEQ or GEQ
        """
        for node in preorder_iter(self):
            if node.nodes_modes == "BRANCH_LEQ":
                node.nodes_values = np.ceil(node.nodes_values).astype(
                    np.dtype(f"uint{precision}")
                )
            elif node.nodes_modes == "BRANCH_GEQ":
                node.nodes_values = np.floor(node.nodes_values).astype(
                    np.dtype(f"uint{precision}")
                )
            elif node.nodes_modes == "LEAF":
                node.nodes_values = np.floor(
                    node.nodes_values,
                ).astype(np.dtype(f"uint{precision}"))
            else:
                raise NotImplementedError("Only LEQ and GEQ are supported")

    def quantize_target_weights(
        self, precision: Literal[8, 16, 32], min_val: float, max_val: float
    ) -> None:
        """
        Quantize the leaf values of the tree.

        Parameters
        ----------
        precision : Literal[8, 16, 32]
            N.Bits of the output value
        min_val : float
            Original range: minimum value
        max_val : float
            Original range : maximum value
        """
        for node in preorder_iter(self):
            if node.is_leaf:
                node.targets_weights, _, _ = quantize(
                    data=node.targets_weights,
                    min_val=min_val,
                    max_val=max_val,
                    precision=precision,
                )
            else:
                node.targets_weights = node.targets_weights.astype(
                    np.dtype(f"uint{precision}")
                )

    def logits_to_vote(self, weights_dtype: np.dtype = np.uint8) -> None:
        """
        Swaps the logits to the vote representation.
        The dtype is given by the maximum number of votes (i.e. n.trees in the ensemble)

        Parameters
        ----------
        weights_dtype : np.dtype
            Smallest dtype that can contain the maximum number of votes. Default np.uint8
        """
        for node in preorder_iter(self):
            if node.is_leaf:
                node.targets_ids = (
                    np.argmax(node.targets_weights)
                    .reshape(1)
                    .astype(node.targets_ids.dtype)
                )
                node.targets_weights = np.asarray([1], dtype=weights_dtype)
            else:
                node.targets_weights = node.targets_weights.astype(weights_dtype)

    def prune_same_targets(self) -> None:
        """
        Remove branches terminating in nodes with the same output.
        """

        stop_flag = True
        while stop_flag:
            stop_flag = False
            for node in self.leaves:
                if (
                    node.siblings[0].is_leaf
                    and node.targets_weights.argmax()
                    == node.siblings[0].targets_weights.argmax()
                ):
                    parent = node.parent
                    parent.left = None
                    parent.right = None
                    stop_flag = True
                    # Add the node data to the parent
                    parent.targets_weights = node.targets_weights
                    parent.targets_ids = node.targets_ids
                    parent.nodes_featureids = node.nodes_featureids
                    parent.nodes_modes = node.nodes_modes
                    break

    def pad_to_min_depth(self, min_depth: int) -> None:
        """
        Moves leaves to the minimum depth. The tree is not binary anymore.
        """
        idx = self.n_nodes
        for leaf in self.leaves:
            if leaf.depth < min_depth:
                # Create a chain of nodes
                n_nodes = min_depth - leaf.depth
                dummies = list()
                for j in range(n_nodes):
                    nodo = Node._from_node(name=j + idx, node=leaf.parent, dummy=True)
                    dummies.append(nodo)
                idx += n_nodes
                new_leaf = Node._from_node(idx, leaf, dummy=False)
                idx += 1
                # Unlink leaves from parents
                parent = leaf.parent
                shift_nodes(self, [leaf.path_name], [None])
                # Link the new nodes
                current_parent = leaf.parent
                for j in range(n_nodes):
                    current_parent.left = dummies[j]
                    current_parent = dummies[j]
                new_leaf.parent = current_parent

        self._set_same_feature_for_dummies()

    def pad_to_perfect(
        self,
        target_edge_depth: Optional[int] = None,
    ) -> None:
        idx = self.n_nodes
        target_depth = (
            target_edge_depth + 1 if target_edge_depth is not None else self.max_depth
        )
        for leaf in self.leaves:
            if (target_depth - leaf.depth) > 0:
                subtree_size = 2 ** (target_depth - leaf.depth + 1) - 1
                n_leaves_subtree = 2 ** (target_depth - leaf.depth)
                n_nodes_subtree = subtree_size - n_leaves_subtree
                # Create the dummy nodes
                dummies = list()
                for j in range(n_nodes_subtree):
                    nodo = Node._from_node(name=j + idx, node=leaf.parent, dummy=True)
                    dummies.append(nodo)
                idx += n_nodes_subtree
                # Create the new leaves
                new_leaves = [
                    Node._from_node(idx + _, leaf, dummy=_ > 0)
                    for _ in range(n_leaves_subtree)
                ]
                idx += n_leaves_subtree
                nodes_made = dummies + new_leaves

                # Unlink leaves from parents
                parent = leaf.parent
                shift_nodes(self, [leaf.path_name], [None])

                # Link the new elements
                for level in range(0, subtree_size):
                    if (2 * level + 1) < len(nodes_made):
                        nodes_made[level].left = nodes_made[2 * level + 1]
                    if (2 * level + 2) < len(nodes_made):
                        nodes_made[level].right = nodes_made[2 * level + 2]
                if parent.left is None:
                    parent.left = nodes_made[0]
                else:
                    parent.right = nodes_made[0]

        self._set_same_feature_for_dummies()

    def _set_same_feature_for_dummies(
        self,
    ):
        # Optimization : Force the same fake feature on same-level dummy nodes
        history = {}
        for grp_dpth in levelordergroup_iter(self):
            for node in grp_dpth:
                if node.is_leaf:
                    break
                if not node.nodes_modes.startswith("DUMMY"):
                    history[node.depth] = node.nodes_featureids
            for node in grp_dpth:
                if node.is_leaf:
                    break
                if node.nodes_modes.startswith("DUMMY"):
                    node.nodes_featureids = history[node.depth]

    # Class methods
    @classmethod
    def parse_dict(cls, tree_dict: Mapping[str, np.ndarray]) -> "Node":
        nodes = list()
        # Build the nodes
        for idx, mode in enumerate(tree_dict["nodes_modes"]):
            indice_vero = tree_dict["nodes_nodeids"][idx]
            nodes_idx = indice_vero
            targets_idx = tree_dict["targets_nodeids"] == nodes_idx
            nodo = cls(
                name=tree_dict["nodes_nodeids"][idx].item(),
                targets_ids=tree_dict["targets_ids"][targets_idx].astype(
                    tree_dict["targets_ids"].dtype
                ),
                targets_weights=tree_dict["targets_weights"][targets_idx].astype(
                    tree_dict["targets_weights"].dtype
                ),
                nodes_featureids=tree_dict["nodes_featureids"][nodes_idx].astype(
                    tree_dict["nodes_featureids"].dtype
                ),
                nodes_values=tree_dict["nodes_values"][nodes_idx].astype(
                    tree_dict["nodes_values"].dtype
                ),
                nodes_hitrates=tree_dict["nodes_hitrates"][nodes_idx].astype(
                    tree_dict["nodes_hitrates"].dtype
                ),
                nodes_modes=tree_dict["nodes_modes"][nodes_idx].astype(
                    tree_dict["nodes_modes"].dtype
                ),
            )
            nodes.append(nodo)
        # Build the tree
        for idx in range(len(nodes)):
            if "LEAF" not in tree_dict["nodes_modes"][idx]:
                nodes[idx].left = nodes[tree_dict["nodes_truenodeids"][idx]]
                nodes[idx].right = nodes[tree_dict["nodes_falsenodeids"][idx]]
        return nodes[0]

    # Export methods
    # TODO: This is sub-optimal, fields should be auto-collected in some way to
    # avoid the need of the method _node_to_dict
    def _node_to_dict(self) -> Mapping[str, np.ndarray]:
        """
        Collects all the node data in a dictionary.

        Returns
        -------
        Mapping[str, np.ndarray]
            Field name : field value
        """
        return {
            "targets_ids": self.targets_ids,
            "targets_weights": self.targets_weights,
            "nodes_featureids": self.nodes_featureids,
            "nodes_values": self.nodes_values,
            "nodes_hitrates": self.nodes_hitrates,
            "nodes_modes": self.nodes_modes,
        }

    def _node_idx_to_dict(self, idx, nodes) -> Mapping[str, np.ndarray]:
        """
        Regenerates the idx of the onnxml model.

        Parameters
        ----------
        idx : _type_
            Index of the node in the tree. Used by onnxml
        nodes : _type_
            List of nodes of the tree.

        Returns
        -------
        Mapping[str, np.ndarray]
            Field name : field value
        """
        return {
            "nodes_nodeids": np.asarray([idx], dtype=np.int64),
            "targets_nodeids": np.asarray(
                [idx for _ in range(len(self.targets_ids))], dtype=np.int64
            ),
            "nodes_falsenodeids": np.asarray(
                [nodes.index(self.right) if self.right is not None else 0],
                dtype=np.int64,
            ),
            "nodes_truenodeids": np.asarray(
                [nodes.index(self.left) if self.left is not None else 0], dtype=np.int64
            ),
        }

    def to_dict(self, tree_idx: int = 0) -> Mapping[str, np.ndarray]:
        """
        Dumps the tree in a dictionary. Fields are compatible with treeco and
        onnxml

        Parameters
        ----------
        tree_idx : int, optional
            the index of the tree in the ensemble, by default 0

        Returns
        -------
        Mapping[str, np.ndarray]
            Field name : field value
        """
        nodes = [*preorder_iter(self)]
        aggregated_dict = nodes[0]._node_to_dict()
        aggregated_dict.update(nodes[0]._node_idx_to_dict(0, nodes))
        for idx, node in enumerate(nodes[1:], 1):
            node_dict = node._node_to_dict()
            node_dict.update(node._node_idx_to_dict(idx, nodes))
            # Add the ids
            for k, v in node_dict.items():
                aggregated_dict[k] = np.hstack(
                    (aggregated_dict[k], v), dtype=aggregated_dict[k].dtype
                )
        # Add the tree fields
        # TODO : This can be moved directly at the ensemble level.
        aggregated_dict["nodes_treeids"] = np.asarray(
            [tree_idx for _ in range(len(nodes))], dtype=np.int64
        )
        aggregated_dict["targets_treeids"] = np.asarray(
            [tree_idx for _ in range(len(aggregated_dict["targets_ids"]))],
            dtype=np.int64,
        )
        return aggregated_dict

    def to_numpy_arrays(
        self, visit_method: Callable, n_features: int
    ) -> Tuple[np.ndarray, ...]:
        """
        Export the tree following the visit method function passed.
        Children indices are computed as the shift from the current node to the one of
        the child in the numpy array - 1 .
        For instance:
        lchild = [4]
        means that the lchild of the current node is the node at position 5 in the array.
        """
        ordered_nodes = [*visit_method(self)]
        # Some utils
        n_leaves: int = self.n_leaves
        target_weights_dtype = next(self.leaves).targets_weights.dtype

        # Since all are ids, we can init them as int64.
        # Further optimization are performed later
        lchild = np.zeros(len(ordered_nodes), dtype=np.int64)
        rchild = np.zeros(len(ordered_nodes), dtype=np.int64)
        nfeatureids = np.zeros(len(ordered_nodes), dtype=np.int64)
        nvalues = np.zeros(len(ordered_nodes), dtype=self.nodes_values.dtype)
        tids = np.zeros((n_leaves, self.leaf_shape), dtype=np.int64)
        tweights = np.zeros((n_leaves, self.leaf_shape), dtype=target_weights_dtype)

        leaf_idx = 0
        for idx, node in enumerate(ordered_nodes):
            if node.right is not None:
                rchild[idx] = ordered_nodes.index(node.right) - idx - 1
            else:
                rchild[idx] = 0

            if node.left is not None:
                lchild[idx] = ordered_nodes.index(node.left) - idx - 1
            else:
                lchild[idx] = 0

            if node.is_leaf:
                nfeatureids[idx] = n_features
                tids[leaf_idx] = node.targets_ids
                tweights[leaf_idx] = node.targets_weights
                leaf_idx += 1
            else:
                nfeatureids[idx] = node.nodes_featureids

            nvalues[idx] = node.nodes_values
        return lchild, rchild, nfeatureids, nvalues, tids, tweights

    def to_numpy_vectors(
        self, prune_full_padding_paths: bool
    ) -> Tuple[np.ndarray, ...]:
        """
        Export to vectors. One per leaf.
        Note that vectors are actually duplicated for each leaf.
        A
         - B
         - C
        Returns twice A, once for B and once for C.

        Parameters
        ----------
        prune_full_padding_paths : bool
            If true, remove the padding paths that end in a fake leaf

        Returns
        -------
        Tuple[np.ndarray, ...]
            [nodes_featureids, nodes_values, targets_ids, targets_weights]
        """
        all_leaves = self.leaves

        # Some utils
        n_leaves = self.n_leaves
        target_weights_dtype = next(self.leaves).targets_weights.dtype
        leaf_n_ids = len(next(self.leaves).targets_ids)
        depth = self.max_depth - 1

        # Since all are ids, we can init them as int64.
        # Further optimization are performed later
        nfeatureids = np.zeros((n_leaves, depth), dtype=np.int64)
        nvalues = np.zeros((n_leaves, depth), dtype=self.nodes_values.dtype)
        tids = np.zeros((n_leaves, leaf_n_ids), dtype=np.int64)
        tweights = np.zeros((n_leaves, self.leaf_shape), dtype=target_weights_dtype)
        padding_path = np.zeros(n_leaves, dtype=bool)

        # Inefficient, but straightforward
        # Iterate over all leaves, get their path and turn it into a vector
        for idx, leaf in enumerate(all_leaves):
            path = self.go_to(leaf)[:-1]  # Last one is the leaf
            for idx_node, node in enumerate(path):
                nfeatureids[idx, idx_node] = node.nodes_featureids
                nvalues[idx, idx_node] = node.nodes_values

            tids[idx] = leaf.targets_ids
            tweights[idx] = leaf.targets_weights
            padding_path[idx] = "DUMMY" in leaf.nodes_modes
        if prune_full_padding_paths:
            # Remove the padding paths
            mask = ~padding_path
            nfeatureids = nfeatureids[mask]
            nvalues = nvalues[mask]
            tids = tids[mask]
            tweights = tweights[mask]

        return nfeatureids, nvalues, tids, tweights

    def to_numpy_matmul(
        self,
        method: Literal[
            "hummingbird_gemm", "hummingbird_iterative", "hummingbird_perfectiterative"
        ],
    ):
        raise NotImplementedError
        # Returns something similar to Hummingbird
