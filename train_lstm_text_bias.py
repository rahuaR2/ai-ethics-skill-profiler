# train_lstm_text_bias.py

import json
import numpy as np
from pathlib import Path

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

DATA_PATH = Path("data/text_corpus.txt")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def load_corpus():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Datei {DATA_PATH} nicht gefunden.")
    text = DATA_PATH.read_text(encoding="utf-8")
    text = text.lower().replace("\n", " ")
    return text

def build_sequences(text, max_len=5, num_words=10000):
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts([text])

    sequences = []
    words = text.split()
    for i in range(max_len, len(words)):
        seq = words[i-max_len:i+1]
        sequences.append(" ".join(seq))

    encoded = tokenizer.texts_to_sequences(sequences)
    encoded = np.array(encoded)

    X, y = encoded[:, :-1], encoded[:, -1]
    vocab_size = len(tokenizer.word_index) + 1
    y = to_categorical(y, num_classes=vocab_size)

    return X, y, tokenizer, vocab_size, max_len

def build_model(vocab_size, max_len, embedding_dim=128, lstm_units=128):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
    model.add(LSTM(lstm_units))
    model.add(Dense(vocab_size, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def main():
    print("📚 Lade Korpus...")
    text = load_corpus()

    print("🧩 Erzeuge Sequenzen...")
    max_len = 5
    X, y, tokenizer, vocab_size, max_len = build_sequences(text, max_len=max_len)

    print(f"Vokabulargröße: {vocab_size}, Trainingsbeispiele: {len(X)}")

    print("🧠 Baue Modell...")
    model = build_model(vocab_size, max_len)

    print("🚀 Starte Training...")
    model.fit(X, y, batch_size=128, epochs=10, validation_split=0.1)

    print("💾 Speichere Modell & Tokenizer...")
    model.save(MODELS_DIR / "lstm_text_bias.h5")

    tokenizer_json = tokenizer.to_json()
    (MODELS_DIR / "lstm_tokenizer.json").write_text(tokenizer_json, encoding="utf-8")

    config = {"max_len": max_len, "vocab_size": vocab_size}
    (MODELS_DIR / "lstm_config.json").write_text(json.dumps(config), encoding="utf-8")

    print("✅ Training abgeschlossen und gespeichert.")

if __name__ == "__main__":
    main()
