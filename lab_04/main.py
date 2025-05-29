import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from nltk.translate.bleu_score import corpus_bleu
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DATA_PATH = 'fra-eng/fra.txt'
MAX_SAMPLES = 10000
LATENT_DIM = 256
BATCH_SIZE = 64
EPOCHS = 30

def load_data(path, max_samples):
    with open(path, encoding='utf-8') as file:
        lines = file.read().splitlines()
    input_texts, target_texts = [], []
    input_vocab, target_vocab = set(), set()
    for line in lines[:max_samples]:
        if '\t' in line:
            source, target = line.split('\t')[:2]
            target = f'\t{target}\n'
            input_texts.append(source)
            target_texts.append(target)
            input_vocab.update(source)
            target_vocab.update(target)
    return input_texts, target_texts, sorted(input_vocab), sorted(target_vocab)

input_texts, target_texts, input_chars, target_chars = load_data(DATA_PATH, MAX_SAMPLES)
input_token_idx = {char: i for i, char in enumerate(input_chars)}
target_token_idx = {char: i for i, char in enumerate(target_chars)}

reverse_input_idx = {i: char for char, i in input_token_idx.items()}
reverse_target_idx = {i: char for char, i in target_token_idx.items()}

max_encoder_len = max(len(txt) for txt in input_texts)
max_decoder_len = max(len(txt) for txt in target_texts)

num_encoder_tokens = len(input_chars)
num_decoder_tokens = len(target_chars)

encoder_input = np.zeros((MAX_SAMPLES, max_encoder_len, num_encoder_tokens), dtype='float32')
decoder_input = np.zeros((MAX_SAMPLES, max_decoder_len, num_decoder_tokens), dtype='float32')
decoder_target = np.zeros((MAX_SAMPLES, max_decoder_len, num_decoder_tokens), dtype='float32')

for i, (src, tgt) in enumerate(zip(input_texts, target_texts)):
    for t, ch in enumerate(src):
        encoder_input[i, t, input_token_idx[ch]] = 1.
    for t, ch in enumerate(tgt):
        decoder_input[i, t, target_token_idx[ch]] = 1.
        if t > 0:
            decoder_target[i, t - 1, target_token_idx[ch]] = 1.

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(LATENT_DIM, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
training_model.fit([encoder_input, decoder_input], decoder_target,
                   batch_size=BATCH_SIZE,
                   epochs=EPOCHS,
                   validation_split=0.2)

encoder_model_inf = Model(encoder_inputs, encoder_states)

dec_state_input_h = Input(shape=(LATENT_DIM,))
dec_state_input_c = Input(shape=(LATENT_DIM,))
dec_states_inputs = [dec_state_input_h, dec_state_input_c]

dec_outputs, dec_h, dec_c = decoder_lstm(decoder_inputs, initial_state=dec_states_inputs)
dec_states = [dec_h, dec_c]
dec_outputs = decoder_dense(dec_outputs)

decoder_model_inf = Model([decoder_inputs] + dec_states_inputs, [dec_outputs] + dec_states)

def translate_sequence(seq):
    states = encoder_model_inf.predict(seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_idx['\t']] = 1.
    decoded = ''
    while True:
        output, h, c = decoder_model_inf.predict([target_seq] + states)
        sampled_index = np.argmax(output[0, -1, :])
        sampled_char = reverse_target_idx[sampled_index]
        decoded += sampled_char
        if sampled_char == '\n' or len(decoded) > max_decoder_len:
            break
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_index] = 1.
        states = [h, c]
    return decoded

references = []
hypotheses = []

for i in range(100):
    input_seq = encoder_input[i:i+1]
    decoded_sent = translate_sequence(input_seq)
    true_sent = target_texts[i][1:-1]
    references.append([list(true_sent)])
    hypotheses.append(list(decoded_sent.strip()))

bleu = corpus_bleu(references, hypotheses)
print("BLEU score:", bleu)