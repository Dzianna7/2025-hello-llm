.. _summarization-label:

Summarization
=============

Models
------

+-----------------------------------------------------------------------+-------+
| Model                                                                 | Lang  |
+=======================================================================+=======+
| `mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization   | EN    |
| <https://huggingface.co/mrm8488/bert-mini2bert-mini-                  |       |
| finetuned-cnn_daily_mail-summarization>`__                            |       |
+-----------------------------------------------------------------------+-------+
| `nandakishormpai/t5-small-machine-articles-tag-generation             | EN    |
| <https://huggingface.co/nandakishormpai                               |       |
| /t5-small-machine-articles-tag-generation>`__                         |       |
+-----------------------------------------------------------------------+-------+
| `mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization | EN    |
| <https://huggingface.co/mrm8488/bert-small2bert-                      |       |
| small-finetuned-cnn_daily_mail-summarization>`__                      |       |
+-----------------------------------------------------------------------+-------+
| `UrukHan/t5-russian-summarization                                     | RU    |
| <https://huggingface.co/UrukHan/t5-russian-summarization>`__          |       |
+-----------------------------------------------------------------------+-------+
| `dmitry-vorobiev/rubert_ria_headlines                                 | RU    |
| <https://huggingface.co/dmitry-vorobiev/rubert_ria_headlines>`__      |       |
+-----------------------------------------------------------------------+-------+


Datasets
--------

1. `ccdv/govreport-summarization <https://huggingface.co/datasets/ccdv/govreport-summarization>`__

   1. **Lang**: EN
   2. **Rows**: 973
   3. **Preprocess**:

      1. Select ``test`` split.
      2. Rename column ``report`` to ``source``.
      3. Rename column ``summary`` to ``target``.
      4. Reset indexes.

2. `cnn_dailymail <https://huggingface.co/datasets/abisee/cnn_dailymail>`__

   1. **Lang**: EN
   2. **Rows**: 11490
   3. **Preprocess**:

      1. Select ``1.0.0`` subset.
      2. Select ``test`` split.
      3. Drop columns ``id``.
      4. Rename column ``article`` to ``source``.
      5. Rename column ``highlights`` to ``target``.
      6. Delete duplicates in dataset.
      7. Remove substring ``(CNN)`` for each ``source`` row.
      8. Reset indexes.

3. `ccdv/pubmed-summarization <https://huggingface.co/datasets/ccdv/pubmed-summarization>`__

   1. **Lang**: EN
   2. **Rows**: 6658
   3. **Preprocess**:

      1. Select ``test`` split.
      2. Rename column ``article`` to ``source``.
      3. Rename column ``abstract`` to ``target``.
      4. Reset indexes.

4. `IlyaGusev/gazeta <https://huggingface.co/datasets/IlyaGusev/gazeta>`__

   1. **Lang**: RU
   2. **Rows**: 6793
   3. **Preprocess**:

      1. Select ``test`` split.
      2. Drop columns ``title``, ``date``, ``url``.
      3. Rename column ``text`` to ``source``.
      4. Rename column ``summary`` to ``target``.
      5. Reset indexes.

5. `d0rj/curation-corpus-ru <https://huggingface.co/datasets/d0rj/curation-corpus-ru>`__

   1. **Lang**: RU
   2. **Rows**: 30454
   3. **Preprocess**:

      1. Select ``train`` split.
      2. Drop columns ``title``, ``date``, ``url``.
      3. Rename column ``article_content`` to ``source``.
      4. Rename column ``summary`` to ``target``.
      5. Reset indexes.

6. `CarlBrendt/Summ_Dialog_News <https://huggingface.co/datasets/CarlBrendt/Summ_Dialog_News>`__

   1. **Lang**: RU
   2. **Rows**: 7609
   3. **Preprocess**:

      1. Select ``test`` split.
      2. Rename column ``info`` to ``source``.
      3. Rename column ``summary`` to ``target``.
      4. Reset indexes.

7. `trixdade/reviews_russian <https://huggingface.co/datasets/trixdade/reviews_russian>`__

   1. **Lang**: RU
   2. **Rows**: 95
   3. **Preprocess**:

      1. Select ``train`` split.
      2. Rename column ``Reviews`` to ``source``.
      3. Rename column ``Summary`` to ``target``.
      4. Reset indexes.

Supervised Fine-Tuning (SFT) Parameters
---------------------------------------

.. note:: Set the parameter ``target_modules`` as
          ``["query", "key", "value", "dense"]`` for the
          `mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization
          <https://huggingface.co/mrm8488/bert-small2bert-small-finetuned-
          cnn_daily_mail-summarization>`__,
          `mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization
          <https://huggingface.co/mrm8488/bert-mini2bert-mini-finetuned-
          cnn_daily_mail-summarization>`__ and
          `dmitry-vorobiev/rubert_ria_headlines <https://huggingface.co/dmitry-
          vorobiev/rubert_ria_headlines>`__ models, and as
          ``["q", "k", "wi", "wo"]`` for the
          `UrukHan/t5-russian-summarization
          <https://huggingface.co/UrukHan/t5-russian-summarization>`__ model.


.. note:: Set the parameters ``fine_tuning_steps=150, rank=24, alpha=36`` for the
          `mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization
          <https://huggingface.co/mrm8488/bert-small2bert-
          small-finetuned-cnn_daily_mail-summarization>`__
          as SFT parameters.


Metrics
-------

-  BLEU
-  ROUGE

.. note:: Use the ``rougeL`` metric and set ``seed=77`` parameter
          when loading the rouge metric.
