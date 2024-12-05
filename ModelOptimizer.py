import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load the pre-trained model
model_name = "facebook/m2m100_418M"
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
model.eval()

# Disable `use_cache` to make the model TorchScript-compatible
model.config.use_cache = False

# Apply dynamic quantization
model_int8 = torch.quantization.quantize_dynamic(
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
        super(ScriptWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        return outputs.logits

# Create an instance of the wrapper
script_wrapper = ScriptWrapper(model_int8)

# Trace the model
traced_model = torch.jit.trace(
    script_wrapper,
    (example_input_ids, example_attention_mask, example_decoder_input_ids),
    strict=False  # Allow non-traceable code paths
)

# Save the model configuration and tokenizer for use in Rust
model.config.save_pretrained("./m2m100_quantized")
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("./m2m100_quantized")

# Save the traced model
traced_model.save("./m2m100_quantized/m2m100_quantized.pt")