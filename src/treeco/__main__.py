# TODO : Work in progress
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Treeco: Compile a tree")
parser.add_argument("model", type=Path, help="Onnx model to compile")
# Nothing is performed at this lvl currently
# onnxml_group = parser.add_argument_group("onnxml", "Onnxml options")

# Optimization passes
crown_group = parser.add_argument_group("crown", "Crown options")
precision = parser.add_argument_group("Precision", "Precision options")
input_precision = precision.add_argument_group("Input Precision", "Precision options")
input_precision.add_argument(
    "--bits",
    type=int,
    default=None,
)
input_precision.add_argument(
    "--scale",
    type=float,
    default=None,
)


parser.add_argument(
    "--quantize-outputs",
    action="store_true",
)

trunk_group = parser.add_argument_group("trunk", "Trunk options")
root_group = parser.add_argument_group("root", "Root options")


parser.add_argument(
    "--emit",
    dest="emit",
    choices=[
        "onnxml",
        "crown",
        "trunk",
        "root",
        "scf",
        "llvm",
        "emitc",
        "emitc-pulp",
    ],
    default="scf",
    help="Compilation target (default: scf)",
)
