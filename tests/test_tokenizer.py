from src.tokenizer.tokenizer import CharacterTokenizer


def test_character_tokenizer_encode_decode_roundtrip():
    transcripts = [
        "hello world",
        "how are you",
        "good morning",
        "this is a speech recognition system",
    ]

    tokenizer = CharacterTokenizer()
    tokenizer.build_vocabulary(transcripts)

    text = "hello world"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == tokenizer.normalize(text)
    assert tokenizer.vocabulary_size() > 0
