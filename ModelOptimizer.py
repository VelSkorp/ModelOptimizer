import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Specify any Seq2Seq model name from the Hugging Face Hub
model_name = "facebook/m2m100_418M"

# Load the pre-trained model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Disable `use_cache` to make the model TorchScript-compatible
model.config.use_cache = False

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Prepare example inputs
vocab_size = model.config.vocab_size
example_input_ids = torch.randint(0, vocab_size, (1, 10), dtype=torch.long)
example_attention_mask = torch.ones((1, 10), dtype=torch.long)
example_decoder_input_ids = torch.randint(0, vocab_size, (1, 10), dtype=torch.long)

# Define a wrapper class for TorchScript compatibility
class ScriptWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        return outputs.logits

# Create an instance of the wrapper
script_wrapper = ScriptWrapper(quantized_model)

# Trace the model
traced_model = torch.jit.trace(
    script_wrapper,
    (example_input_ids, example_attention_mask, example_decoder_input_ids),
    strict=False  # Allow non-traceable code paths
)

# Save the model configuration and tokenizer
output_directory = "./quantized_model"
model.config.save_pretrained(output_directory)
tokenizer.save_pretrained(output_directory)

# Save the traced model
traced_model_path = f"{output_directory}/model_quantized.pt"
traced_model.save(traced_model_path)
print(f"Quantized TorchScript model saved to {traced_model_path}")
