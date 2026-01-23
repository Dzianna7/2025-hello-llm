"""
Starter for demonstration of laboratory work.
"""
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor


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


if __name__ == "__main__":
    main()

