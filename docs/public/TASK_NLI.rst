.. _nli-label:

NLI
===

Models
------

+-------------------------------------------------------------------+------+
| Model                                                             | Lang |
+===================================================================+======+
| `cointegrated/rubert-base-cased-nli-threeway <https://            | RU   |
| huggingface.co/cointegrated/rubert-base-cased-nli-threeway>`__    |      |
+-------------------------------------------------------------------+------+
| `cointegrated/rubert-tiny-bilingual-nli <https://                 | RU   |
| huggingface.co/cointegrated/rubert-tiny-bilingual-nli>`__         |      |
+-------------------------------------------------------------------+------+
| `cross-encoder/qnli-distilroberta-base                            | EN   |
| <https://huggingface.co/cross-encoder/qnli-distilroberta-base>`__ |      |
+-------------------------------------------------------------------+------+
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli <https:             | EN   |
| //huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli>`__ |      |
+-------------------------------------------------------------------+------+


Datasets
--------

1. `cointegrated/nli-rus-translated-v2021 <https://huggingface.co/datasets/cointegrated/nli-rus-translated-v2021>`__

   1. **Lang**: RU
   2. **Rows**: 19647
   3. **Preprocess**:

      1. Select ``dev`` split.
      2. Filter the dataset by the column ``source`` with the value ``mnli``.

         1. This step you should implement in :py:meth:`lab_7_llm.main.RawDataImporter.obtain`.

      3. Leave only columns ``premise_ru``, ``hypothesis_ru`` and ``label``.
      4. Rename column ``premise_ru`` to ``premise``.
      5. Rename column ``hypothesis_ru`` to ``hypothesis``.
      6. Rename column ``label`` to  ``target``.
      7. Delete empty rows in dataset.
      8. Delete duplicates in dataset.
      9. Map ``target`` with class labels.
      10. Reset indexes.

2. `XNLI <https://huggingface.co/datasets/facebook/xnli>`__

   1. **Lang**: RU
   2. **Rows**: 2490
   3. **Preprocess**:

      1. Select ``ru`` subset.
      2. Select ``validation`` split.
      3. Rename column ``label`` to  ``target``.
      4. Delete duplicates in dataset.
      5. Delete empty rows in dataset.
      6. Reset indexes.

3. `GLUE QNLI <https://huggingface.co/datasets/nyu-mll/glue>`__

   1. **Lang**: EN
   2. **Rows**: 5463
   3. **Preprocess**:

      1. Select ``qnli`` subset.
      2. Select ``validation`` split.
      3. Rename column ``question`` to  ``premise``.
      4. Rename column ``sentence`` to  ``hypothesis``.
      5. Rename column ``label`` to  ``target``.
      6. Delete duplicates in dataset.
      7. Delete empty rows in dataset.
      8. Map ``target`` with class labels.
      9. Reset indexes.

4. `GLUE MNLI <https://huggingface.co/datasets/nyu-mll/glue>`__

   1. **Lang**: EN
   2. **Rows**: 9815
   3. **Preprocess**:

      1. Select ``mnli`` subset.
      2. Select ``validation_matched`` split.
      3. Rename column ``label`` to  ``target``.
      4. Delete duplicates in dataset.
      5. Delete empty rows in dataset.
      6. Reset indexes.

Supervised Fine-Tuning (SFT) Parameters
---------------------------------------

.. note:: Set the parameter ``target_modules=["key"]`` for the
          `cointegrated/rubert-base-cased-nli-threeway
          <https://huggingface.co/cointegrated/rubert-base-cased-nli-threeway>`__ model.

Metrics
-------

-  Accuracy
