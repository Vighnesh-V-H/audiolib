import json


class CharacterTokenizer:

    def __init__(self):
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.token_to_id = {}
        self.id_to_token = {}

    def normalize(self, text):
        text = text.lower()
        text = " ".join(text.split())
        return text

    def build_vocabulary(self, transcripts):
        characters = set()

        for transcript in transcripts:
            transcript = self.normalize(transcript)
            characters.update(transcript)

        special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        ]

        tokens = special_tokens + sorted(characters)

        self.token_to_id = {
            token: idx
            for idx, token in enumerate(tokens)
        }

        self.id_to_token = {
            idx: token
            for token, idx in self.token_to_id.items()
        }

    def encode(self, text):
        text = self.normalize(text)

        bos_id = self.token_to_id[self.bos_token]
        eos_id = self.token_to_id[self.eos_token]
        unk_id = self.token_to_id[self.unk_token]

        ids = [bos_id]

        for character in text:
            token_id = self.token_to_id.get(character, unk_id)
            ids.append(token_id)

        ids.append(eos_id)

        return ids

    def decode(self, ids):
        tokens = []

        for token_id in ids:
            token = self.id_to_token[token_id]

            if token == self.bos_token:
                continue

            if token == self.eos_token:
                break

            if token == self.pad_token:
                continue

            tokens.append(token)

        return "".join(tokens)

    def save(self, path):
        data = {
            "token_to_id": self.token_to_id
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.token_to_id = {
            token: int(token_id)
            for token, token_id in data["token_to_id"].items()
        }

        self.id_to_token = {
            token_id: token
            for token, token_id in self.token_to_id.items()
        }

    def vocabulary_size(self):
        return len(self.token_to_id)

    def pad_id(self):
        return self.token_to_id[self.pad_token]

    def bos_id(self):
        return self.token_to_id[self.bos_token]

    def eos_id(self):
        return self.token_to_id[self.eos_token]

    def unk_id(self):
        return self.token_to_id[self.unk_token]
