""" 
All passes that optimize/simplify the IR are in this module.
Ideally, no conversions are involved in the passes.
"""

from .crown_pad_to_perfect import CrownPadTreesPerfectPass
from .crown_prune import CrownPruneTreesPass
from .crown_quantize import (
    CrownQuantizeInputPass,
    CrownQuantizeLeavesPass,
    CrownRoundInputPass,
)
from .crown_voting import CrownConvertToVotingClassifierPass

from .trunk_pad_to_min_depth import TrunkPadToMinDepthPass

from .func_legalize import UpdateSignatureFuncOp

from .memref_merge_subview import FoldMemRefSubViewChainPass
from .memref_quantize_global_index import MemrefQuantizeGlobalIndexPass
from .ml_global_quantize_index import MlGlobalQuantizeIndexPass

from .prepare_llvm_lowering import PrepareLLVMLoweringPass
