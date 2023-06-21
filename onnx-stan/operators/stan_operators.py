import onnx
from onnx import helper, AttributeProto, TensorProto, GraphProto


# Creating operator registry
domain = ""  # empty string indicates the default ONNX domain
registry = onnx.checker.check_model.OperatorRegistry()

# Add custom operators
opset_version = 1
registry.register_operator(
    'HMC',
    opset_version,
    AttributeProto.STRING,
    [('input', TensorProto.FLOAT)],
    [('output', TensorProto.FLOAT)]
)

registry.register_operator(
    'PathFinder',
    opset_version,
    AttributeProto.STRING,
    [('input', TensorProto.FLOAT)],
    [('output', TensorProto.FLOAT)]
)

# You would continue this pattern for additional operators
# registry.register_operator(...
# registry.register_operator(...

# Use your custom registry for further tasks
onnx.checker.check_model(model, registry)
