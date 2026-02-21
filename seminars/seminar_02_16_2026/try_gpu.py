"""
Inference acceleration with PyTorch (CPU/GPU)
"""

# pylint: disable=too-few-public-methods

import os
import time
from typing import cast

try:
    from transformers import AutoModelForCausalLM, GPT2Tokenizer
except ImportError:
    print('Library "Transformers" not installed. Failed to import.')

try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')


try:
    import psutil  # type: ignore
except ImportError:
    print('Library "psutil" not installed. Failed to import.')


class GPT2Wrapper(torch.nn.Module):
    """
    Wraps a HuggingFace GPT-2 model to expose only its Transformer core.
    """

    def __init__(self, lm_model: AutoModelForCausalLM) -> None:  # type: ignore
        """
        Initialize an instance of GPT2Wrapper.
        """
        super().__init__()
        # Extract the pure Transformer module from the HuggingFace model.
        self.transformer = lm_model.transformer

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Wraps a HuggingFace GPT-2 model to expose only its Transformer core.
        """
        hidden_states, _ = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=False,
        )
        return cast(torch.Tensor, hidden_states)


def trace_model(hf_model: AutoModelForCausalLM) -> None:  # type: ignore
    """
    Generates dummy inputs, traces the GPT2Wrapper with TorchScript, and saves the traced module.
    """
    wrapper = GPT2Wrapper(hf_model).eval()
    ex_input_ids = torch.randint(
        low=0, high=hf_model.config.vocab_size, size=(1, 16), dtype=torch.long
    )
    ex_attention_mask = torch.ones_like(ex_input_ids, dtype=torch.long)
    ex_position_ids = torch.arange(16, dtype=torch.long).unsqueeze(0)

    # Step 6: Trace the wrapper with TorchScript
    traced = torch.jit.trace(
        wrapper, (ex_input_ids, ex_attention_mask, ex_position_ids), strict=False
    )
    traced.save("gpt2_three_inputs.pt")
    print("Saved traced model → gpt2_three_inputs.pt")


def main() -> None:
    """
    Entrypoint for the listing.
    """

    # Disable tokenizer parallelism to suppress warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
    hf_model.eval()

    # Step 1: Instantiate and prepare the wrapper for tracing

    trace_model(hf_model)

    device = torch.device("cpu")

    # Step 2: Load tokenizer and model on CPU
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

    # Step 3: Tokenize input text
    tokens = tokenizer("When will humans go to Mars?", return_tensors="pt").to(device)

    # Step 4: Measure memory usage using psutil
    process = psutil.Process()
    start_mem = process.memory_info().rss

    # Step 5: Measure inference time
    start_time = time.time()
    output_ids = model.generate(**tokens, max_new_tokens=20)
    end_mem = process.memory_info().rss
    inference_time = time.time() - start_time

    # Step 6: Decode output tokens to text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Step 7: Print CPU inference results
    print(
        f"""
    [ CPU ]
    Inference time: {inference_time:.3f} sec
    Memory usage: start {start_mem / (1024*1024):.1f} MB → end {end_mem / (1024*1024):.1f} MB
    Generation: {output_text}
    """
    )

    ####################################
    # GPU Inference with PyTorch (CUDA)
    ####################################

    # Сheck that the GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")

        # Step 8: Reload model to GPU
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        tokens = tokenizer("When will humans go to Mars?", return_tensors="pt").to(device)

        # Step 9: Reset GPU memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Step 10: Measure memory before inference
        start_mem = torch.cuda.memory_allocated(device)
        start_time = time.time()
        output_ids = model.generate(**tokens, max_new_tokens=20)
        torch.cuda.synchronize()

        # Step 11: Measure memory after inference
        end_mem = torch.cuda.memory_allocated(device)
        torch.cuda.max_memory_allocated(device)
        inference_time = time.time() - start_time
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Step 12: Print GPU inference results
        print(
            f"""
        [ CUDA ]
        Inference time: {inference_time:.3f} sec
        Memory usage: start {start_mem / (1024*1024):.1f} MB → end {end_mem / (1024*1024):.1f} MB
        Generation: {output_text}
        """
        )


if __name__ == "__main__":
    main()
