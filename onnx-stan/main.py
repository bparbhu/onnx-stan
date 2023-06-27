# Import necessary libraries
import os
from parse_stan_model import StanParser  # This is an example, replace with your actual stan parser module
from onnx_generator import OnnxGenerator  # This is an example, replace with your actual onnx generator module

def main():
    # Assuming the user inputs the path to a .stan file
    stan_file_path = input("Please enter the path to your .stan file: ")

    # Check if the file exists
    if not os.path.isfile(stan_file_path):
        print(f"File {stan_file_path} does not exist.")
        return

    # Create the StanParser and parse the .stan file
    stan_parser = StanParser()
    stan_model = stan_parser.parse(stan_file_path)

    # Create the OnnxGenerator and generate an ONNX model
    onnx_generator = OnnxGenerator()
    onnx_model = onnx_generator.generate(stan_model)

    # Save the ONNX model to a file
    onnx_model_path = stan_file_path.replace('.stan', '.onnx')
    onnx_generator.save(onnx_model, onnx_model_path)

    print(f"ONNX model saved to {onnx_model_path}.")

if __name__ == "__main__":
    main()
