import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
# Use standalone Keras optimizer to match HuggingFace's Keras backend
from keras.optimizers import Adam

# 1. Load and split dataset
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
data = fetch_20newsgroups(
    subset='all', categories=categories,
    remove=('headers','footers','quotes')
)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. TF-IDF + Logistic Regression
print("--- TF-IDF + Logistic Regression ---")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
clf_tfidf = LogisticRegression(max_iter=1000)
clf_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred_tfidf))

# 3. Word2Vec + Logistic Regression
print("--- Word2Vec + Logistic Regression ---")
tokenized_train = [doc.split() for doc in X_train]
word2vec_model = Word2Vec(
    sentences=tokenized_train, vector_size=100, window=5,
    min_count=2, workers=4
)

def document_vector(doc):
    words = doc.split()
    vecs = [word2vec_model.wv[w] for w in words if w in word2vec_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

X_train_w2v = np.vstack([document_vector(d) for d in X_train])
X_test_w2v = np.vstack([document_vector(d) for d in X_test])
clf_w2v = LogisticRegression(max_iter=1000)
clf_w2v.fit(X_train_w2v, y_train)
y_pred_w2v = clf_w2v.predict(X_test_w2v)
print(classification_report(y_test, y_pred_w2v))

# 4. RNN (LSTM) classifier
print("--- RNN (LSTM) ---")
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
seq_train = tokenizer.texts_to_sequences(X_train)
seq_test = tokenizer.texts_to_sequences(X_test)
max_len = 200
X_train_seq = pad_sequences(seq_train, maxlen=max_len)
X_test_seq = pad_sequences(seq_test, maxlen=max_len)

model_rnn = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_len),
    LSTM(128),
    Dense(len(categories), activation='softmax')
])
model_rnn.compile(
    optimizer='adam',  # tf.keras optimizer works here
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model_rnn.fit(
    X_train_seq, y_train,
    epochs=3, batch_size=64,
    validation_split=0.1
)
y_pred_rnn = np.argmax(model_rnn.predict(X_test_seq), axis=1)
print(classification_report(y_test, y_pred_rnn))

# 5. Seq2Seq as classification
print("--- Seq2Seq Classification ---")
chars = ['0', '1', '2', '3']
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
max_label_len = 1

y_train_in = [[char_to_idx[str(lbl)]] for lbl in y_train]
y_train_out = [[char_to_idx[str(lbl)]] for lbl in y_train]
y_train_in = pad_sequences(y_train_in, maxlen=max_label_len, padding='post')
y_train_out = pad_sequences(y_train_out, maxlen=max_label_len, padding='post')

# Encoder
encoder_inputs = Input(shape=(max_len,), name='encoder_input')
enc_emb = Embedding(10000, 128, name='enc_embedding')(encoder_inputs)
_, state_h, state_c = LSTM(128, return_state=True, name='enc_lstm')(enc_emb)
encoder_states = [state_h, state_c]

# Decoder for training
decoder_inputs = Input(shape=(max_label_len,), name='decoder_input')
dec_embedding_layer = Embedding(input_dim=len(chars), output_dim=32, name='dec_embedding')
dec_emb = dec_embedding_layer(decoder_inputs)
dec_lstm = LSTM(128, return_sequences=True, return_state=True, name='dec_lstm')
dec_outputs, _, _ = dec_lstm(dec_emb, initial_state=encoder_states)
dec_dense = Dense(len(chars), activation='softmax', name='dec_dense')
decoder_outputs = dec_dense(dec_outputs)

seq2seq = Model([encoder_inputs, decoder_inputs], decoder_outputs)
seq2seq.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
seq2seq.fit([X_train_seq, y_train_in], np.expand_dims(y_train_out, -1), epochs=3, batch_size=64)

# Inference models
encoder_model_inf = Model(encoder_inputs, encoder_states)

dec_state_input_h = Input(shape=(128,), name='dec_inf_input_h')
dec_state_input_c = Input(shape=(128,), name='dec_inf_input_c')
dec_states_inputs = [dec_state_input_h, dec_state_input_c]

dec_input_inf = Input(shape=(max_label_len,), name='decoder_input_inf')
dec_emb_inf = dec_embedding_layer(dec_input_inf)
dec_outputs_inf, state_h_inf, state_c_inf = dec_lstm(dec_emb_inf, initial_state=dec_states_inputs)
dec_outputs_inf = dec_dense(dec_outputs_inf)

decoder_model = Model([dec_input_inf] + dec_states_inputs, [dec_outputs_inf, state_h_inf, state_c_inf], name='decoder_model')

def classify_seq2seq(input_seq):
    states_value = encoder_model_inf.predict(input_seq)
    target_seq = np.zeros((1, max_label_len), dtype='int32')
    output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
    return np.argmax(output_tokens[0, -1, :])

preds_seq2seq = [classify_seq2seq(x.reshape(1, max_len)) for x in X_test_seq]
print(classification_report(y_test, preds_seq2seq))

# 6. Transformer (BERT) classifier
print("--- Transformer (BERT) ---")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_texts(texts, max_len=128):
    ids, masks = [], []
    for txt in texts:
        enc = bert_tokenizer.encode_plus(
            txt, max_length=max_len,
            padding='max_length', truncation=True,
            return_attention_mask=True
        )
        ids.append(enc['input_ids'])
        masks.append(enc['attention_mask'])
    return np.array(ids), np.array(masks)

train_ids, train_masks = encode_texts(X_train)
test_ids, test_masks = encode_texts(X_test)

bert_model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=len(categories)
)
bert_model.compile(
    optimizer=Adam(learning_rate=2e-5),  # use standalone Keras optimizer
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
bert_model.fit([train_ids, train_masks], y_train, epochs=2, batch_size=16)

y_pred_bert = np.argmax(bert_model.predict([test_ids, test_masks])[0], axis=1)
print(classification_report(y_test, y_pred_bert))

# 7. Compare accuracies
results = {
    'TF-IDF': accuracy_score(y_test, y_pred_tfidf),
    'Word2Vec': accuracy_score(y_test, y_pred_w2v),
    'RNN': accuracy_score(y_test, y_pred_rnn),
    'Seq2Seq': accuracy_score(y_test, preds_seq2seq),
    'BERT': accuracy_score(y_test, y_pred_bert)
}
print("--- Accuracy Comparison ---")
for method, acc in results.items():
    print(f"{method}: {acc:.4f}")

