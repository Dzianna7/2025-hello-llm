"""
Inference acceleration with ONNX Runtime
"""

import subprocess
import time
from typing import Dict, List, Tuple

try:
    import numpy as np
except ImportError:
    print('Library "numpy" not installed. Failed to import.')


try:
    import psutil  # type: ignore
except ImportError:
    print('Library "psutil" not installed. Failed to import.')


try:
    from transformers import GPT2Config, GPT2Tokenizer
except ImportError:
    print('Library "transformers" not installed. Failed to import.')


try:
    import onnxruntime as ort  # type: ignore[import]
except ImportError:
    print('Libraries "onnx onnxruntime optimum[onnxruntime]" not installed. Failed to import.')


def export_gpt2_to_onnx() -> None:
    """
    export GPT-2 model to ONNX
    """
    cmd = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        "gpt2",
        "--task",
        "causal-lm-with-past",
        "gpt2_onnx",
    ]

    subprocess.run(cmd, check=True)
    print("Export is finished, gpt2_onnx is created.")


def load_model(  # type: ignore[no-any-unimported]
    model_dir: str = "gpt2_onnx",
) -> Tuple[GPT2Tokenizer, GPT2Config, ort.InferenceSession, List[str]]:
    """
    Load then initialize and return
    the tokenizer, model config, inference session, and past_key_values names.
    """
    export_gpt2_to_onnx()
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    config = GPT2Config.from_pretrained(model_dir)
    session = ort.InferenceSession(f"{model_dir}/model.onnx", providers=["CPUExecutionProvider"])
    past_names = [
        inp.name for inp in session.get_inputs() if inp.name.startswith("past_key_values")
    ]
    return tokenizer, config, session, past_names  # type: ignore[return-value]


def init_inputs(  # type: ignore
    prompt: str, tokenizer: GPT2Tokenizer, config: GPT2Config, past_names: List[str]
) -> Dict[str, np.ndarray]:
    """
    Tokenize the prompt and build the initial input dictionary
    with input_ids, attention_mask, position_ids, and empty past_key_values.
    """
    enc = tokenizer(prompt, return_tensors="np")
    input_ids = enc["input_ids"].astype(np.int64)
    base = {
        "input_ids": input_ids,
        "attention_mask": np.ones_like(input_ids, np.int64),
        "position_ids": np.arange(input_ids.shape[1], dtype=np.int64)[None, :],
    }

    past = {
        name: np.zeros((1, config.n_head, 0, config.n_embd // config.n_head), dtype=np.float32)
        for name in past_names
    }

    base.update(past)
    return base


def run_step(  # type: ignore[no-any-unimported]
    session: ort.InferenceSession,
    ort_inputs: Dict[str, np.ndarray],
    output_names: List[str],
    past_names: List[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Execute one inference pass through the ONNX session,
    update past_key_values cache in ort_inputs, and return logits.
    """
    outs = session.run(output_names, ort_inputs)
    logits, presents = outs[0], outs[1:]
    for name, arr in zip(past_names, presents):
        ort_inputs[name] = arr
    return logits, ort_inputs  # ort_inputs now holds updated past


def main() -> None:
    """
    Entrypoint for the listing.
    """
    # Step 15: Load ONNX model for CPU inference
    tokenizer, config, session, past_names = load_model()

    ort_inputs = init_inputs("When will humans go to Mars?", tokenizer, config, past_names)
    output_names = [o.name for o in session.get_outputs()]

    start_mem = psutil.Process().memory_info().rss
    start_time = time.time()

    # Step 16: First forward pass with full prompt
    logits, ort_inputs = run_step(session, ort_inputs, output_names, past_names)
    tokens = ort_inputs["input_ids"][0].tolist()

    # Step 17: Select first generated token using argmax
    tokens.append(int(np.argmax(logits[0, -1])))
    cur_len = len(tokens)

    # Step 18: Iterative token-by-token generation
    for _ in range(20):
        step_inputs = {
            "input_ids": np.array([[tokens[-1]]], np.int64),
            "attention_mask": np.ones((1, 1), np.int64),
            "position_ids": np.array([[cur_len - 1]], np.int64),
        }
        step_inputs.update({n: ort_inputs[n] for n in past_names})
        logits, ort_inputs = run_step(session, step_inputs, output_names, past_names)
        next_id = int(np.argmax(logits[0, -1]))
        tokens.append(next_id)
        cur_len += 1

    end_mem = psutil.Process().memory_info().rss
    inference_time = time.time() - start_time

    print(
        f"""
    [ CPU with ONNX ]
        Inference time: {inference_time:.3f} sec
        Memory usage: start {start_mem / (1024*1024):.1f} MB → end {end_mem / (1024*1024):.1f}
        Generation: {tokenizer.decode(tokens, skip_special_tokens=True)}
        """
    )


if __name__ == "__main__":
    main()
