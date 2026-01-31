"""
Starter for demonstration of laboratory work.
"""
import json
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline
from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
# pylint: disable=too-many-locals, undefined-variable, unused-import


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    importer = RawDataImporter(hf_name="dair-ai/emotion")
    importer.obtain()
    preprocessor = RawDataPreprocessor(raw_data=importer.raw_data)
    preprocessor.transform()


    # # 2. Convert text to tokens
    # text = "i feel like a faithful servant"
    # tokens = tokenizer(text, return_tensors="pt")

    # # 3. Print tokens keys
    # print(tokens.keys())

    # raw_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"].tolist()[0])
    # print(raw_tokens)

    # # line numbers with these IDs in vocab.txt (-1 because of zero indexing)
    # print(tokens["input_ids"].tolist()[0])

    
    
    # print(model)

    #     # 6. Classify text
    # with torch.no_grad():
    #     output = model(**tokens)

    # # 7. Print prediction
    # print(output.logits)
    # print(output.logits.shape)

    # # 8. Print label
    # predictions = torch.argmax(output.logits).item()

    # # 9. Print predictions
    # print(predictions)

    # # 10. Map with labels
    # labels = model.config.id2label
    # print(labels[predictions])

    dataset = TaskDataset(preprocessor.data.head(100))
    df = dataset.data
    print(df.head(5))

    with open('settings.json', 'r') as f:
        settings = json.load(f)
    
    batch_size = 1
    max_length = 120
    device = 'cpu'

    pipeline = LLMPipeline(settings["parameters"]["model"], settings["parameters"]["dataset"], max_length, batch_size, device)
    
    analysis_result = pipeline.analyze_model()
    print(analysis_result)

    test_samples = [
        ("im feeling quite sad and sorry for myself but ill snap out of it soon",),
        ("i'm feeling this weird mix of anxious and excited about tomorrow's meeting, like my stomach is doing flips",),
        ("i stopped feeling cold and began feeling hot",),
    ]

    for sample in test_samples:
        print(f"\n{'=' * 50}")
        print(f"Sample: {sample[0]}")
        prediction = pipeline.infer_sample(sample)
        print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()

