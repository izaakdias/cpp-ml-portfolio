import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

x = helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", 2])
y = helper.make_tensor_value_info("prediction", TensorProto.FLOAT, ["batch", 1])
graph = helper.make_graph(
    [
        helper.make_node("MatMul", ["features", "weights"], ["product"]),
        helper.make_node("Add", ["product", "bias"], ["prediction"]),
    ],
    "linear",
    [x],
    [y],
    [
        numpy_helper.from_array(np.array([[2], [-1]], dtype=np.float32), "weights"),
        numpy_helper.from_array(np.array([0.5], dtype=np.float32), "bias"),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, sys.argv[1])
