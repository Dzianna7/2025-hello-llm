"""
HuggingFace model listing.
"""

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        BertForSequenceClassification,
    )
except ImportError:
    print('Library "transformers" not installed. Failed to import.')

try:
    from torchinfo import summary  # type: ignore
except ImportError:
    print('Library "torchinfo" not installed. Failed to import.')

try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')


def main() -> None:
    """
    Entrypoint for the listing.
    """

    #########################
    # Classification scenario
    #########################

    # 1.1 Import model - exact class name from config
    model = BertForSequenceClassification.from_pretrained("s-nlp/russian_toxicity_classifier")
    print(type(model))

    # 1.2 Import model - auto class for particular task
    model = AutoModelForSequenceClassification.from_pretrained("s-nlp/russian_toxicity_classifier")
    print(type(model))

    # 2. Get model config
    config = model.config

    # 3. Model configuration (reflects config.json on HuggingFace website)
    print(config)

    # 4. Model's max context size
    # Pay attention to where it is defined for your model.
    # For some models it is called d_model, for some (generative ones) in decoder section
    embeddings_length = config.max_position_embeddings

    # 5. Imitating input data - fill full input of the model
    ids = torch.ones(1, embeddings_length, dtype=torch.long)

    # 6. Prepare data based on args of forward method of the corresponding model class
    tokens = {"input_ids": ids, "attention_mask": ids}

    # 7. Call summary method from torchinfo library
    result = summary(model, input_data=tokens, device="cpu", verbose=0)

    # 8. Resulting summary
    print(result)

    # 9. Get output shape
    shape = result.summary_list[-1].output_size
    print(shape)

    #########################
    # Generation scenario
    #########################

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    print(type(model))
    result = summary(
        model,
        input_data={
            "input_ids": torch.ones(1, model.config.max_position_embeddings, dtype=torch.long)
        },
    )
    shape = result.summary_list[-1].output_size
    print(shape)


if __name__ == "__main__":
    main()
