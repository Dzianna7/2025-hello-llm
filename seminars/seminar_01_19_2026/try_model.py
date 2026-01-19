"""
HuggingFace model listing.
"""

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        GenerationConfig,
    )
except ImportError:
    print('Library "transformers" not installed. Failed to import.')

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

    # 1. Classification
    tokenizer = AutoTokenizer.from_pretrained("s-nlp/russian_toxicity_classifier")

    # 2. Convert text to tokens
    text = "KFC заработал в Нижнем под новым брендом"
    tokens = tokenizer(text, return_tensors="pt")

    # 3. Print tokens keys
    print(tokens.keys())

    raw_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"].tolist()[0])
    print(raw_tokens)

    # line numbers with these IDs in vocab.txt (-1 because of zero indexing)
    print(tokens["input_ids"].tolist()[0])

    # 4. Import model
    model = AutoModelForSequenceClassification.from_pretrained("s-nlp/russian_toxicity_classifier")

    # 5 Print model
    print(model)

    model.eval()

    # 6. Classify text
    with torch.no_grad():
        output = model(**tokens)

    # 7. Print prediction
    print(output.logits)
    print(output.logits.shape)

    # 8. Print label
    predictions = torch.argmax(output.logits).item()

    # 9. Print predictions
    print(predictions)

    # 10. Map with labels
    labels = model.config.id2label
    print(labels[predictions])

    #########################
    # Generation scenario
    #########################

    # 11. Import tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 12. Convert text to tokens
    text = (
        "Ron DeSantis’ fraught presidential campaign ended Sunday following a months-long downward"
    )
    tokens = tokenizer(text, return_tensors="pt")

    raw_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"].tolist()[0])
    print(raw_tokens)

    # 13. Load model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()

    # Predict next token
    with torch.no_grad():
        output = model(**tokens).logits[0]

    # 14. next token is stored in last row
    last_token_predictions = output[-1]
    next_token_id = torch.argmax(last_token_predictions).item()

    # Shock content: GPT-2 from 2018 predicts continuation from 2024!
    print(next_token_id)
    print(tokenizer.decode((next_token_id,)))

    # 14. Generate text of given length
    with torch.no_grad():
        output = model.generate(**tokens)
    results = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(results[0])

    # 15. Configure tokenizer
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=120)
    with torch.no_grad():
        output = model.generate(
            **tokens,
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=0.7,
                max_new_tokens=10,
            ),
            pad_token_id=tokenizer.eos_token_id,
        )
    results = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(results[0])


if __name__ == "__main__":
    main()
