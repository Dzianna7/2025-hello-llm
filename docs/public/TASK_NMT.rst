.. _nmt-label:

Neural Machine Translation
==========================

Models
------

+-------------------------------------------------------------------------------------+------+
| Model                                                                               | Lang |
+=====================================================================================+======+
| `Helsinki-NLP/opus-mt-en-fr                                                         | EN   |
| <https://huggingface.co/Helsinki-NLP/opus-mt-en-fr>`__                              |      |
+-------------------------------------------------------------------------------------+------+
| `google-t5/t5-small <https://huggingface.co/t5-small>`__                            | EN   |
+-------------------------------------------------------------------------------------+------+
| `Helsinki-NLP/opus-mt-ru-en                                                         | RU   |
| <https://huggingface.co/Helsinki-NLP/opus-mt-ru-en>`__                              |      |
+-------------------------------------------------------------------------------------+------+
| `Helsinki-NLP/opus-mt-ru-es                                                         | RU   |
| <https://huggingface.co/Helsinki-NLP/opus-mt-ru-es>`__                              |      |
+-------------------------------------------------------------------------------------+------+

Datasets
--------

1. `enimai/MuST-C-fr <https://huggingface.co/datasets/enimai/MuST-C-fr>`__

   1. **Lang**: EN
   2. **Rows**: 2630
   3. **Preprocess**:

      1. Select ``test`` split.
      2. Rename column ``en`` to ``source``.
      3. Rename column ``fr`` to ``target``.
      4. Delete duplicates in dataset.
      5. Reset indexes.

2. `RocioUrquijo/en_de <https://huggingface.co/datasets/RocioUrquijo/en_de>`__

   1. **Lang**: EN
   2. **Rows**: 700
   3. **Preprocess**:

      1. Select ``test`` split.
      2. Rename column ``en`` to ``source``.
      3. Rename column ``de`` to ``target``.
      4. Delete duplicates in dataset.
      5. Add prefix *Translate from English to German:*  for each ``source`` row.
      6. Reset indexes.

3. `shreevigneshs/iwslt-2023-en-ru-train-val-split-0.2 <https://huggingface.co/datasets/shreevigneshs/iwslt-2023-en-ru-train-val-split-0.2>`__

   1. **Lang**: RU
   2. **Rows**: 600
   3. **Preprocess**:

      1. Select ``if_test`` split.
      2. Drop columns ``ru_annotated``, ``styles``.
      3. Rename column ``ru`` to ``source``.
      4. Rename column ``en`` to ``target``.
      5. Reset indexes.

4. `nuvocare/Ted2020_en_es_fr_de_it_ca_pl_ru_nl <https://huggingface.co/datasets/nuvocare/Ted2020_en_es_fr_de_it_ca_pl_ru_nl>`__

   1. **Lang**: RU
   2. **Rows**: 7210
   3. **Preprocess**:

      1. Select ``test`` split.
      2. Drop columns ``de``, ``en``, ``fr``, ``it``, ``nl``, ``pl``.
      3. Rename column ``ru`` to ``source``.
      4. Rename column ``es`` to ``target``.
      5. Delete empty rows in dataset.
      6. Delete duplicates in dataset.
      7. Reset indexes.

Supervised Fine-Tuning (SFT) Parameters
---------------------------------------

.. note:: Set the parameter
          ``target_modules=["q_proj", "k_proj"]`` for the
          `Helsinki-NLP/opus-mt-en-fr <https://huggingface.co/Helsinki-NLP/opus-mt-en-fr>`__,
          `Helsinki-NLP/opus-mt-ru-en <https://huggingface.co/Helsinki-NLP/opus-mt-ru-en>`__,
          `Helsinki-NLP/opus-mt-ru-es <https://huggingface.co/Helsinki-NLP/opus-mt-ru-es>`__
          models.

.. note:: Set the parameters ``target_modules=["q", "k", "v"], rank=24, alpha=36`` for the
          `google-t5/t5-small <https://huggingface.co/t5-small>`__ model as SFT parameter.

.. note:: Set the parameter ``fine_tuning_steps=100`` for the
          `Helsinki-NLP/opus-mt-ru-es <https://huggingface.co/Helsinki-NLP/opus-mt-ru-es>`__
          model as SFT parameter.

.. note:: Set the parameters ``fine_tuning_steps=60, rank=16, alpha=24`` for the
          `Helsinki-NLP/opus-mt-en-fr <https://huggingface.co/Helsinki-NLP/opus-mt-en-fr>`__
          model as SFT parameters.


Metrics
-------

-  BLEU
