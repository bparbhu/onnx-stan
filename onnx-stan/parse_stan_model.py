from onnx import helper, TensorProto
import onnxruntime
from onnxruntime_extensions import PyOp
import onnxruntime_custom_ops  # ensure the module is imported

stan_model = {
    'data': {
        'N': 'int64',
        'x': 'float32',
        'y': 'float32',
    },
    'parameters': {
        'alpha': 'float32',
        'beta': 'float32',
        'sigma': 'float32',
    },
    'model': 'y ~ normal(alpha + beta * x, sigma)',
}



# Create nodes for the operations in the model
mul_node = helper.make_node(
    'Mul',
    inputs=['beta', 'x'],
    outputs=['mul_result'],
    name='mul_node'
)
add_node = helper.make_node(
    'Add',
    inputs=['alpha', 'mul_result'],
    outputs=['add_result'],
    name='add_node'
)
normal_node = helper.make_node(
    'CustomNormal',  # This would be a custom operator
    inputs=['add_result', 'sigma'],
    outputs=['y'],
    name='normal_node'
)

# Create the graph
graph = helper.make_graph(
    [mul_node, add_node, normal_node],
    'simple_stan_model',
    [
        helper.make_tensor_value_info('alpha', TensorProto.FLOAT, []),
        helper.make_tensor_value_info('beta', TensorProto.FLOAT, []),
        helper.make_tensor_value_info('sigma', TensorProto.FLOAT, []),
        helper.make_tensor_value_info('x', TensorProto.FLOAT, ['N']),
    ],
    [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['N'])],
)

# Create the ONNX model
model = helper.make_model(graph)


onnxruntime.get_library_path()
so = onnxruntime.SessionOptions()
so.register_custom_ops_library(PyOp.get_library_path())

sess = onnxruntime.InferenceSession('model.onnx', so)


