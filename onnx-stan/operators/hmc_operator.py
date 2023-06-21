import onnx
from onnx.helper import make_node, make_operatorsetid
from onnx.onnx_pb import FunctionProto

# Define operator set
operator_set = make_operatorsetid(domain='ai.onnx.extensions', version=1)

# OpSchema().Name("CustomMCMCSampler")
#     .Input(0, "distribution", "The distribution to sample from.", "T")
#     .Input(1, "params", "The parameters of the distribution.", "T")
#     .Output(0, "samples", "The generated samples.", "T")
#     .Attr("num_samples", "The number of samples to draw.", AttributeProto.INT)
#     .Attr("algorithm", "The MCMC algorithm to use.", AttributeProto.STRING)
#     .TypeConstraint("T", ["tensor(float)", "tensor(double)"], "Allow float and double tensors")
#


# Define HMC operator
hmc_operator = onnx.helper.make_operator_schema('HMC',
    domain='ai.onnx.extensions',
    since_version=1,
    inputs=[
        onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, None]),
        onnx.helper.make_tensor_value_info('target', onnx.TensorProto.FLOAT, [1, 1]),
        onnx.helper.make_tensor_value_info('step_size', onnx.TensorProto.FLOAT, [1, 1]),
        onnx.helper.make_tensor_value_info('num_steps', onnx.TensorProto.INT32, [1, 1])
    ],
    outputs=[
        onnx.helper.make_tensor_value_info('samples', onnx.TensorProto.FLOAT, [None, None]),
        onnx.helper.make_tensor_value_info('acceptance_rate', onnx.TensorProto.FLOAT, [1, 1])
    ],
    type_constraints=[
        onnx.helper.make_type_constraint('T', 'tensor(float16)', 'tensor(float)', 'tensor(double)'),
    ],
    doc_string='Perform Hamiltonian Monte Carlo sampling.'
)
