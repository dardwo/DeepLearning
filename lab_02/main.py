import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report
import os
import nltk

# nltk.download('punkt_tab')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

train_path = "sport-text-classification-ball-ISI-public/train/train.tsv.gz"
dev_path = "sport-text-classification-ball-ISI-public/dev-0/in.tsv"
dev_labels_path = "sport-text-classification-ball-ISI-public/dev-0/expected.tsv"
test_path = "sport-text-classification-ball-ISI-public/test-A/in.tsv"

train = pd.read_csv(train_path, sep="\t", header=None, names=["label", "text"], on_bad_lines='skip')
dev = pd.read_csv(dev_path, sep="\t", header=None, names=["text"], on_bad_lines='skip')
dev_labels = pd.read_csv(dev_labels_path, sep="\t", header=None, names=["label"], on_bad_lines='skip')
test = pd.read_csv(test_path, sep="\t", header=None, names=["text"], on_bad_lines='skip')

train["tokens"] = train["text"].apply(lambda x: word_tokenize(str(x).lower()))
dev["tokens"] = dev["text"].apply(lambda x: word_tokenize(str(x).lower()))
test["tokens"] = test["text"].apply(lambda x: word_tokenize(str(x).lower()))

tagged_train = [TaggedDocument(words=row["tokens"], tags=[str(i)]) for i, row in train.iterrows()]

doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4)
doc2vec_model.build_vocab(tagged_train)

for epoch in range(3):
    print(f"\nEpoch {epoch+1}/3 - Doc2Vec training")
    doc2vec_model.train(tagged_train, total_examples=doc2vec_model.corpus_count, epochs=1)
    doc2vec_model.alpha -= 0.002
    doc2vec_model.min_alpha = doc2vec_model.alpha

X_train = np.array([doc2vec_model.dv[str(i)] for i in range(len(train))])
y_train = train["label"].astype(int).values

X_dev = np.array([doc2vec_model.infer_vector(tokens, alpha=0.025, epochs=10) for tokens in dev["tokens"]])
y_dev = dev_labels["label"].astype(int).values

X_test = np.array([doc2vec_model.infer_vector(tokens, alpha=0.025, epochs=10) for tokens in test["tokens"]])

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nTrening sieci neuronowej...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_dev, y_dev))

y_pred_prob = model.predict(X_dev).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\n--- Wyniki ewaluacji ---")
print(f"Dokładność: {accuracy_score(y_dev, y_pred)}")
print("\nClassification Report:\n", classification_report(y_dev, y_pred))

y_test_pred_prob = model.predict(X_test).flatten()
y_test_pred = (y_test_pred_prob >= 0.5).astype(int)

with open("sport-text-classification-ball-ISI-public/test-A/predicted.tsv", "w") as f:
    for label in y_test_pred:
        f.write(f"{label}\n")
