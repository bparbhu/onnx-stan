import onnx
from onnx import helper, TensorProto


class OnnxGenerator:
    def __init__(self):
        pass

    def generate(self, stan_model):
        # Create an empty ONNX model
        onnx_model = helper.make_model()
        # Depending on the structure of your stan_model, you may have to create the nodes in a loop
        # For now, let's assume we're creating one node for an example
        node_def = helper.make_node(
            'NameOfOperator',  # replace this with the name of the operator
            ['input'],  # inputs
            ['output'],  # outputs
            # attributes if any
        )
        onnx_model.graph.node.extend([node_def])

        # Use the information in the stan_model to populate the ONNX model
        # This is the part that you would have to implement
        # ...

        return onnx_model

    def save(self, onnx_model, path):
        # Save the ONNX model to a file
        onnx.save_model(onnx_model, path)
