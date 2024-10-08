from copy import deepcopy
from typing import (
    Any,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
    Callable,
    Set,
)

import numpy as np
from bigtree import levelorder_iter, preorder_iter
from xdsl.dialects.builtin import IntegerAttr, StringAttr, i64

from treeco.dialects import treeco
from treeco.model.node import Node
from treeco.utils import (
    convert_attrdict_to_lists,
    convert_np_to_arrayattr,
    convert_np_to_tensor,
)


def _lambda_from_aggregate_mode(aggregate_mode: str) -> Callable:
    mapping = {
        Ensemble.AGGREGATE_MODE_SUM: lambda x: np.sum(x, axis=1),
        # Given a 3D array of dimension (n_samples, n_trees, 1),
        # 1. Squeeze the last dimension
        # 2. Apply bincount to each row, with min_length = n_classes
        Ensemble.AGGREGATE_MODE_VOTE: lambda x: np.apply_along_axis(
            lambda j: np.bincount(j, minlength=len(np.unique(x.squeeze(-1)))),
            axis=1,
            arr=x.squeeze(-1),
        ),
    }
    return mapping[aggregate_mode]

def _get_binary_classification_threshold(dtype: np.dtype, n_trees : int):
    if "float" in dtype.name:
        # Standard float
        base_thr= 0.5
    elif "uint" in dtype.name:
        # Unsigned int
        base_thr= 2 ** (dtype.itemsize * 8 - 1)
    else:
        # Signed int
        base_thr= 0
    
    return base_thr / n_trees 



class Ensemble:
    """
    Core class for the ensemble of trees.
    It can be used as a standalone, or inside the TreeCo module
    """

    # TODO : Maybe map to integers
    AGGREGATE_MODE_SUM = "SUM"
    AGGREGATE_MODE_VOTE = "VOTE"

    # Class Builders
    def __init__(
        self,
        *,
        trees: Union[List[Node], List[List[Node]]],
        n_features: int,
        post_transform: Optional[str] = None,
        aggregate_mode: Optional[str] = None,
    ) -> None:
        self.trees: List[Node] = trees
        self.post_transform: Optional[str] = post_transform
        self.n_features: int = n_features
        self.aggregate_mode: Optional[str] = aggregate_mode

    # PROPERTIES AND CONTROL VARIABLES
    @property
    def targets(self) -> Set:
        """
        Id of each target class
        """
        tgts = set()
        for tree in self.trees:
            tgts = tgts.union(tgts, tree.targets)
        return tgts

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    @property
    def max_depth(self):
        return max([tree.max_depth for tree in self.trees])

    @property
    def min_depth(self):
        return min([leaf.depth for tree in self.trees for leaf in tree.leaves])

    @property
    def n_nodes(self):
        nodes = 0
        for tree in self.trees:
            nodes += len(list(preorder_iter(tree)))
        return nodes

    @property
    def leaf_shape(self):
        return self.trees[0].leaf_shape

    @property
    def n_leaves(self):
        leaves = 0
        for tree in self.trees:
            leaves += len(list(tree.leaves))
        return leaves

    @property
    def n_trees(self) -> int:
        return len(self.trees)

    def is_perfect(self) -> bool:
        """
        Checks if all trees in the ensemble are perfect.

        Returns
        -------
        bool
            True if all trees are perfect, False otherwise
        """
        is_perfect = True
        for tree in self.trees:
            is_perfect &= tree.is_perfect()
        return is_perfect

    def is_oblivious(self) -> bool:
        is_oblivious = self.is_perfect()
        for tree in self.trees:
            is_oblivious &= tree.is_oblivious()
        return is_oblivious

    def has_constant_depth(self) -> bool:
        return len(set([tree.max_depth for tree in self.trees])) == 1

    # Methods
    def predict_raw(self, X) -> np.ndarray:
        """
        Iterative prediction function for the ensemble.
        Returns the output of each tree, with no aggregation or post-transform.

        Parameters
        ----------
        X : np.ndarray
            The input data
        Returns
        -------
        np.ndarray
            The predictions with shape (n_samples, n_trees, leaf_shape)
        """
        predictions = np.zeros((X.shape[0], self.n_trees, self.leaf_shape))
        is_voting = self.aggregate_mode == Ensemble.AGGREGATE_MODE_VOTE
        for idx, tree in enumerate(self.trees):
            for i_idx, x_in in enumerate(X):
                prediction = tree.predict(x_in, return_target_id=is_voting)
                predictions[i_idx, idx] = prediction
        predictions = predictions.astype(prediction.dtype)
        return predictions

    ## The following methods depend on the aggregation mode
    def predict(self, X) -> np.ndarray:
        """
        Iterative prediction function for the ensemble with aggregation.
        It assumes all leaves in a tree have the same targets.
        -----------
        X : np.ndarray
            The input data
        Returns
        -------
        np.ndarray
            The predictions with shape (n_samples, n_trees, n_targets)
        """
        # TODO : Quite slow, could be parallelized.
        assert self.aggregate_mode in [
            Ensemble.AGGREGATE_MODE_SUM,
            Ensemble.AGGREGATE_MODE_VOTE,
        ]
        predictions = self.predict_raw(X)  # Shape (n_samples, n_trees, leaf_shape)
        lambda_fun: Callable = _lambda_from_aggregate_mode(
            aggregate_mode=self.aggregate_mode
        )
        predictions = lambda_fun(predictions)
        return predictions

    @property
    def output_range(self):
        bounds = np.zeros((self.n_targets, self.n_trees, 2))
        for tree_idx, tree in enumerate(self.trees):
            tree = cast(Node, tree)
            min_val, max_val = tree.output_range
            targets_ids: np.ndarray = np.asarray(list(tree.targets))
            bounds[targets_ids, tree_idx, 0] = min_val
            bounds[targets_ids, tree_idx, 1] = max_val
        lambda_fun: Callable = _lambda_from_aggregate_mode(
            aggregate_mode=self.aggregate_mode
        )
        # Apply the aggregation function on the bounds
        aggregated_bounds = lambda_fun(bounds)
        min_val = np.min(aggregated_bounds)
        max_val = np.max(aggregated_bounds)
        return min_val, max_val

    def quantize_thresholds(
        self,
        method: Literal["round", "quantize"],
        precision: int,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> None:
        """
        Quantizes or rounds the splitting thresholds of all trees in the ensemble.

        Parameters
        ----------
        method : Literal[&quot;round&quot;, &quot;quantize&quot;]
            Round (pre-training quant) or quantize (post-training quant) the values
        precision : int
            N.Bits to be used
        min_val : Optional[float], optional
            Min. value of the quantization range, by default None.
        max_val : Optional[float], optional
            Max. value of the quantization range, by default None

        Raises
        ------
        NotImplementedError
            If the method is not implemented
        """
        for tree in self.trees:
            tree = cast(Node, tree)
            if method == "round":
                tree.round_nodes_values(precision)
            elif method == "quantize":
                tree.quantize_nodes_values(precision, min_val, max_val)
            else:
                raise NotImplementedError(f"Method {method} not implemented")

    def quantize_leaves(self, precision: int, min_val: float, max_val: float) -> None:
        """
        Quantize the leaves of the ensemble.

        Parameters
        ----------
        precision : int
            Output precision
        min_val : float
            Min. value of the quantization range
        max_val : float
            Max. value of the quantization range

        Raises
        ------
        Warning
            If no aggregation mode is specified in the ensemble object
        ValueError
            If the aggregation mode is set to "vote"
        """
        if self.aggregate_mode is None:
            raise Warning("No aggregation mode specified, using sum")
        elif self.aggregate_mode == Ensemble.AGGREGATE_MODE_VOTE:
            raise ValueError("Vote quantization has no sense")
        for tree in self.trees:
            tree = cast(Node, tree)
            tree.quantize_target_weights(precision, min_val, max_val)

    def logits_to_vote(self) -> None:
        """
        Swap logits with the argmax in the leaves.
        """
        assert self.aggregate_mode != Ensemble.AGGREGATE_MODE_VOTE
        bin_class_thr = _get_binary_classification_threshold(next(self.trees[0].leaves).targets_weights.dtype, self.n_trees)
        


        new_dtype = np.min_scalar_type(self.n_trees)
        for idx, tree in enumerate(self.trees):
            tree = cast(Node, tree)
            tree._additional_info["binary_threshold"] = bin_class_thr
            tree.prune_same_targets()
            tree.logits_to_vote(weights_dtype=new_dtype)
        self.aggregate_mode = Ensemble.AGGREGATE_MODE_VOTE

    def pad_to_perfect(
        self, target_depth: Union[int | Literal["auto"]] = "auto"
    ) -> None:
        """
        Pad the tree to perfect.

        Parameters
        ----------
        target_depth : Union[int, Literal[&quot;auto&quot;]], optional
            if auto, all trees are padded to the maximum depth of the ensemble
        """

        # TODO: Can this be smarter? (e.g. pad each tree to perfect depending on their max depth)
        if target_depth == "auto":
            target_depth = self.max_depth

        for tree in self.trees:
            tree = cast(Node, tree)
            tree.pad_to_perfect(target_edge_depth=target_depth)

    def pad_to_min_depth(self, min_depth: Optional[int] = None):
        if not min_depth:
            min_depth = self.min_depth
        for tree in self.trees:
            tree = cast(Node, tree)
            tree.pad_to_min_depth(min_depth=min_depth)

    def prune_trees(self, n_trees: int) -> None:
        """
        Prune the ensemble to a number of trees.

        Parameters
        ----------
        n_trees : int
            Number of trees to keep
        """
        self.trees = self.trees[:n_trees]

    # IMPORT FUNCTIONS
    # TODO : Can this be cached in some way?
    @classmethod
    def parse_attr(cls, attr: treeco.TreeEnsembleAttr) -> "Ensemble":
        ensemble_attr = convert_attrdict_to_lists(attr.to_dict())
        tree_list = list()
        for tree_idx in range(attr.get_n_trees()):
            tree_dict = Ensemble._parse_dict(
                tree_idx, ensemble_attr, remove_single_elements=True
            )
            root: Node = Node.parse_dict(tree_dict)
            tree_list.append(root)

        return cls(
            trees=tree_list,
            post_transform=ensemble_attr.get("post_transform", None),
            n_features=ensemble_attr.get("n_features", None),
            aggregate_mode=ensemble_attr.get("aggregate_mode", None),
        )

    @staticmethod
    def _parse_dict(
        tree_idx: int,
        ensemble_attr_data: Mapping[str, Any],
        remove_single_elements: bool = False,
    ) -> Mapping[str, np.ndarray]:
        """
        Utility method to extract the tree arrays from the ensemble attributes.

        Parameters
        ----------
        tree_idx : int
            Id of the tree to parse
        ensemble_attr_data : Mapping[str, Any]
            A dictionary containing the mapped attributes
        remove_single_elements : bool, optional
            If True, drop all single element attributes from the ensemble

        Returns
        -------
        Mapping[str, np.ndarray]
            A dictionary containing the tree information
        """
        tree_dict = {}
        nodes_tree_idx = ensemble_attr_data["nodes_treeids"] == tree_idx
        targets_tree_idx = ensemble_attr_data["targets_treeids"] == tree_idx

        for k, v in ensemble_attr_data.items():
            if k.startswith("nodes_") and isinstance(v, Iterable):
                tree_val = v[nodes_tree_idx]
                tree_dict[k] = tree_val
            elif k.startswith("targets_") and isinstance(v, Iterable):
                tree_val = v[targets_tree_idx]
                tree_dict[k] = tree_val
            elif not remove_single_elements:
                tree_val = v
                tree_dict[k] = tree_val
        return tree_dict

    # Export functions
    def to_attr(self) -> "treeco.TreeEnsembleAttr":
        trees_dicts = list()
        # Generate the base attributes, with no xdsl types
        for idx, tree in enumerate(self.trees):
            trees_dicts.append(tree.to_dict(tree_idx=idx))
        # Concatenate the trees values
        ensemble_dict = deepcopy(trees_dicts[0])
        for k, v in ensemble_dict.items():
            for tree_dict in trees_dicts[1:]:
                v = np.hstack((v, tree_dict[k]))
            ensemble_dict[k] = v

        # Convert the arrays to the correct types
        for k, v in ensemble_dict.items():
            if "id" in k:
                ensemble_dict[k] = convert_np_to_tensor(v, is_index=True)
            elif "str" in v.dtype.name:
                ensemble_dict[k] = convert_np_to_arrayattr(v)
            else:
                ensemble_dict[k] = convert_np_to_tensor(v)
        ensemble_dict["n_features"] = IntegerAttr(self.n_features, i64)
        if self.post_transform is not None:
            ensemble_dict["post_transform"] = StringAttr(self.post_transform)
        if self.aggregate_mode is not None:
            ensemble_dict["aggregate_mode"] = StringAttr(self.aggregate_mode)
        return ensemble_dict

    # Export functions
    def to_numpy_arrays(
        self,
        node_indices: Literal["perfect", "children", "rchild", "auto"] = "auto",
        leaf_values: Literal["internal", "external", "auto"] = "auto",
        compress_indices : bool = False
    ) -> Tuple[Mapping[str, np.ndarray], str, str]:
        """
        Export the ensemble to numpy arrays, for the C/LLVM backend

        Parameters
        ----------
        node_indices : Literal[&quot;perfect&quot;, &quot;children&quot;, &quot;rchild&quot;, &quot;auto&quot;], optional
            How to store indices to visit nodes, by default "auto"
        leaf_values : Literal[&quot;internal&quot;, &quot;external&quot;, &quot;auto&quot;], optional
            How to store leaves, in the arrays of in a new one, by default "auto"

        Returns
        -------
        Mapping[str, np.ndarray]
            A dictionary containing the arrays. The key is the name of the field.
            Data types are numpy-compatible.
        """

        if node_indices == "auto":
            node_indices = "perfect" if self.is_perfect() else "rchild"

        # Change the visit type depending on the node indices mode:
        # - perfect: the indices are perfect, so by-level
        # - children/rchild: pre-order visit
        if node_indices == "perfect":
            visit_method = levelorder_iter
        else:
            visit_method = preorder_iter

        # Leaves storage: internal or external
        # Internal iff leaf_shape == 1 + constraints on types
        # TODO: Implement this part
        # Currently, we are storing the leaves externally
        leaf_values = "external"

        roots_ids = []
        nodes_truenodeids = []
        nodes_falsenodeids = []
        nodes_featureids = []
        nodes_values = []
        targets_ids = []
        targets_weights = []

        # Iterate over the trees
        n_leaves = 0
        n_nodes = 0
        for tree in self.trees:
            lchild, rchild, nfeatureids, nvalues, tids, tweights = tree.to_numpy_arrays(
                visit_method=visit_method, n_features=self.n_features
            )
            if leaf_values == "external":
                leaves_idxs = nfeatureids == self.n_features
                leaf_range = np.arange(n_leaves, n_leaves + tweights.shape[0])
                # Store in the node_ids array if it is perfect, as we need no stopping condition!
                if node_indices == "perfect":
                    nfeatureids = nfeatureids.astype(
                        max(np.min_scalar_type(leaf_range.max()), nfeatureids.dtype)
                    )
                    nfeatureids[leaves_idxs] = leaf_range
                else:
                    # Store the leaves idxs, default in rchild
                    # feature_idx would be invalid, it denotes a leaf
                    rchild = rchild.astype(
                        max(np.min_scalar_type(leaf_range.max()), rchild.dtype)
                    )
                    rchild[leaves_idxs] = leaf_range

            # Append the arrays
            roots_ids.append(n_nodes)
            nodes_truenodeids.append(lchild)
            nodes_falsenodeids.append(rchild)
            nodes_featureids.append(nfeatureids)
            nodes_values.append(nvalues)
            targets_ids.append(tids)
            targets_weights.append(tweights)
            # Update
            n_leaves += tweights.shape[0]
            n_nodes += len(nvalues)

        # Concatenate the arrays
        nodes_featureids = np.concatenate(nodes_featureids)
        nodes_falsenodeids = np.concatenate(nodes_falsenodeids)
        nodes_truenodeids = np.concatenate(nodes_truenodeids)
        nodes_values = np.concatenate(nodes_values)
        roots_ids = np.array(roots_ids)
        targets_ids = np.concatenate(targets_ids)
        targets_weights = np.concatenate(targets_weights)

        # Create the dictionary
        return_dictionary = {
            "roots_ids": roots_ids,
            "nodes_values": nodes_values,
            "nodes_featureids": nodes_featureids,
        }
        if leaf_values == "external":
            return_dictionary["targets_ids"] = targets_ids
            return_dictionary["targets_weights"] = targets_weights
            if self.aggregate_mode == Ensemble.AGGREGATE_MODE_VOTE:
                leaf_array = "targets_ids"
            else:
                leaf_array = "targets_weights"
        else:
            return_dictionary["targets_ids"] = targets_ids
            if self.aggregate_mode == Ensemble.AGGREGATE_MODE_VOTE:
                leaf_array = ""
            else:
                leaf_array = ""

        if node_indices in ["rchild", "children"]:
            return_dictionary["nodes_falsenodeids"] = nodes_falsenodeids

        if node_indices == "children":
            return_dictionary["nodes_truenodeids"] = nodes_truenodeids
        # No nodes idxs are stored in the perfect mode
        if compress_indices:
            for k,v in return_dictionary.items():
                if "ids" in k:
                    return_dictionary[k] = return_dictionary[k].astype(np.min_scalar_type(v.max()))

        return return_dictionary, node_indices, leaf_array

    def to_numpy_vectors(
        self,
        prune_full_padding_paths: bool = True,
    ) -> Mapping[str, np.ndarray]:
        """
        Exports the ensemble as a set of vectors/matrices.
        It requires a perfect ensemble as input.
        All trees must have the same depth.

        Parameters
        ----------
        prune_full_padding_paths : bool, optional
            Remove paths to leaf that are full padding, by default True

        Returns
        -------
        Mapping[str, np.ndarray]
            Field name : numpy array
        """
        assert self.is_perfect(), "Only perfect trees are supported"
        assert self.has_constant_depth(), "All trees must have the same depth"

        depth: int = self.trees[0].max_depth - 1
        nodes_featureids = list()
        nodes_values = list()
        targets_ids = list()
        targets_weights = list()

        # The start of vectors for each tree in the nodes_* and targets_* structures
        # If prune_full_padding_paths is False, this is a single constant
        roots = list()

        n_vectors = 0
        for tree in self.trees:
            nfeatureids, nvalues, tids, tweights = tree.to_numpy_vectors(
                prune_full_padding_paths=prune_full_padding_paths
            )
            roots.append(n_vectors)
            nodes_featureids.append(nfeatureids)
            nodes_values.append(nvalues)
            targets_ids.append(tids)
            targets_weights.append(tweights)

            n_vectors += len(nvalues)

        # Concatenate the arrays
        roots = np.array(roots)
        nodes_featureids = np.concatenate(nodes_featureids)
        nodes_values = np.concatenate(nodes_values)
        targets_ids = np.concatenate(targets_ids)
        targets_weights = np.concatenate(targets_weights)

        return {
            "roots_ids": roots,
            "nodes_values": nodes_values,
            "nodes_featureids": nodes_featureids,
            "targets_ids": targets_ids,
            "targets_weights": targets_weights,
        }
