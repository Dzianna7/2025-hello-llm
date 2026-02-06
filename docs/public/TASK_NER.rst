.. _ner-label:

NER
==============

Models
------

+--------------------------------------------------------------------------+------+-----------+
| Model                                                                    | Lang | Task      |
+==========================================================================+======+===========+
| `dslim/distilbert-NER                                                    | EN   | NER       |
| <https://huggingface.co/dslim/distilbert-NER>`__                         |      |           |
+--------------------------------------------------------------------------+------+-----------+
| `Babelscape/wikineural-multilingual-ner                                  | EN   | NER       |
| <https://huggingface.co/Babelscape/wikineural-multilingual-ner>`__       |      |           |
+--------------------------------------------------------------------------+------+-----------+

Datasets
--------

1. `Babelscape/wikineural <https://huggingface.co/Babelscape/wikineural-multilingual-ner>`__

   1. **Lang**: EN
   2. **Rows**: 11590
   3. **Preprocess**:

      1. Select ``val_en`` split.
      2. Rename column ``ner_tags`` to ``target``.
      3. Rename column ``tokens`` to ``source``.
      4. Reset indexes.

2. `eriktks/conll2003 <https://huggingface.co/datasets/eriktks/conll2003>`__

   1. **Lang**: EN
   2. **Rows**: 3250
   3. **Preprocess**:

      1. Select ``validation`` split.
      2. Rename column ``ner_tags`` to ``target``.
      3. Rename column ``tokens`` to ``source``.
      4. Reset indexes.

.. note::

   When obtaining this dataset, pass the following parameters to the call of
   ``load_dataset``:

   - ``revision="refs/convert/parquet"``

Inferring batch
---------------

Process of implementing method
:py:meth:`lab_7_llm.main.LLMPipeline._infer_batch`
for named entity recognition task has its specifics:

   1. You need to set the ``is_split_into_words=True`` parameter during the tokenization.
   2. The prediction of the model will contain a tensor with labels for each token
      obtained during tokenization of ``sample_batch``.
   3. The number of labels corresponds to the number of tokens.
   4. To assess the quality of the model, it is necessary that the number of labels
      coincides with the length of the original sequence.
   5. You need to process model prediction result so that the prediction contains only
      the labels of the first tokens of each word. Use the ``word_ids`` method of the
      tokenizer to determine the word boundaries.

.. note:: For example, there is a sample ``['CRICKET', '-', 'LEICESTERSHIRE', 'TAKE', 'OVER', 'AT', 'TOP', '.']``
          which is tokenized to ``['[CLS]', 'CR', '##IC', '##KE', '##T', '-', 'L', '##EI', '##CE',
          '##ST', '##ER', '##S', '##H', '##IR', '##E', 'T', '##A', '##KE', 'O', '##VE', '##R', 'AT', 'TO', '##P',
          '[SEP]']``. In this case, each token corresponds to the following predictions
          ``[[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]``.
          Only the labels for the first token of each word need to be included in the final result,
          namely ``[[0, 0, 0, 1, 0, 0, 0, 0, 0]]``. Thus, if the model predicted label ``1`` for the first token
          of the word ``LEICESTERSHIRE``, then the final result for this word will include ``1``.

Supervised Fine-Tuning (SFT) Parameters
---------------------------------------

.. note:: Set the parameter ``target_modules`` as
          ``["q_lin", "k_lin", "v_lin", "out_lin"]`` for the
          `dslim/distilbert-NER <https://huggingface.co/dslim/distilbert-NER>`__
          model.

Metrics
-------

-  Accuracy
