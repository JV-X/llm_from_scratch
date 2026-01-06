import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input="ddia_c0.txt",
    model_prefix="sp",
    vocab_size=2914,
    model_type="unigram",
    character_coverage=0.9995
)
