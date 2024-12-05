# ModelOptimizer

A tool for optimizing machine learning models through dynamic quantization to enhance efficiency and reduce resource consumption.
This project demonstrates how to make the `facebook/m2m100_418M` multilingual translation model compatible with TorchScript using dynamic quantization. The resulting model and tokenizer are saved in a format that can be used in other environments, such as Rust.

---

## Features

- **Dynamic Quantization**: Reduces model size and improves inference speed by applying 8-bit quantization.
- **TorchScript Compatibility**: Prepares the model for deployment in TorchScript environments.
- **Exportable Artifacts**: Saves both the model and tokenizer in a deployable format.

---

## Requirements

- **Python 3.8+**
- **PyTorch**: Ensure PyTorch is installed. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for installation instructions.
- **Transformers Library**: Install the `transformers` library for pre-trained model and tokenizer.

### Installation

1. Install required dependencies:
   ```bash
   pip install torch transformers
   ```

2. Clone this repository or copy the script.

## Usage

1. **Load the Pre-Trained Model**: The script uses the facebook/m2m100_418M model from the Transformers library.

2. **Disable** `use_cache`: Ensures TorchScript compatibility.

3. **Apply Dynamic Quantization**: Reduces the size of `torch.nn.Linear` layers to 8-bit integers.

4. **Wrap the Model for TorchScript**: Encapsulates the model in a wrapper class for tracing.

5. **Trace and Save**: Generates a TorchScript-compatible quantized model and saves it along with the tokenizer.

## Output Artifacts

- M**odel Configuration and Tokenizer**: Saved in the `./m2m100_quantized` directory for use in deployment environments.

    - `config.json`: Model configuration.
    - `vocab.json` and `merges.txt`: Tokenizer vocabulary files.

- **Quantized Model**: The traced model is saved as `m2m100_quantized.pt` in the same directory.

## Steps in Detail

1. **Dynamic Quantization:**
    - Applies 8-bit integer quantization to all `torch.nn.Linear` layers in the model for optimized performance.

2. **TorchScript Wrapper:**
    - Defines a custom wrapper class to handle TorchScript's requirements.

3. **Tracing:** 
    - Uses example inputs to create a traced model.

4. **Saving Artifacts:**
    - Saves the traced model and tokenizer for deployment.

## Benefits

- **Reduced Size:** The model is dynamically quantized, reducing its memory footprint.
- **Faster Inference:** Using 8-bit operations speeds up inference on supported hardware.
- **Compatibility:** The TorchScript model can be deployed in various environments, including Rust-based frameworks.


## Contributing

Contributions are welcome! Please submit issues or pull requests with any improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.