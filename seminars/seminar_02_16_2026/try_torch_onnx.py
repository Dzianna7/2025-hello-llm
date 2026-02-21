"""
Wraps a HuggingFace GPT-2 model to expose only its Transformer core.
"""

try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')

try:
    from transformers import (
        AutoModelForCausalLM,
    )
except ImportError:
    print('Library "transformers" not installed. Failed to import.')

from seminars.seminar_02_16_2026.try_gpu import GPT2Wrapper


def main() -> None:
    """
    Entrypoint for the listing.
    """
    hf_model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    wrapper = GPT2Wrapper(hf_model).eval()

    # Step 13: Prepare sample inputs
    seq_len = 16
    ex_ids = torch.randint(0, hf_model.config.vocab_size, (1, seq_len))
    ex_mask = torch.ones_like(ex_ids)
    ex_pos = torch.arange(seq_len).unsqueeze(0)

    # Step 14: Export to ONNX
    torch.onnx.export(
        wrapper,
        (ex_ids, ex_mask, ex_pos),
        "gpt2_transformer.onnx",
        opset_version=14,
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "attention_mask": {1: "seq_len"},
            "position_ids": {1: "seq_len"},
            "last_hidden_state": {1: "seq_len"},
        },
    )
    print("Saved ONNX model → gpt2_transformer.onnx")


if __name__ == "__main__":
    main()
