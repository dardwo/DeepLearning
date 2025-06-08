import pandas as pd
from transformers import pipeline
from seqeval.metrics import classification_report, f1_score
from tqdm import tqdm
import torch

def load_data(dev_x_path, dev_y_path):
    dev_x = pd.read_csv(dev_x_path, sep="\t", names=["text"])
    dev_y = pd.read_csv(dev_y_path, sep="\t", names=["label"])
    
    dev_x["tokens"] = dev_x["text"].apply(lambda x: x.split())
    dev_y["labels"] = dev_y["label"].apply(lambda x: x.split())
    
    return dev_x["tokens"].tolist(), dev_y["labels"].tolist()

def align_predictions_with_tokens(tokens, ner_results, tokenizer):
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        return_offsets_mapping=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    
    token_labels = ["O"] * len(tokens)
    
    for entity in ner_results:
        start_idx = entity['start']
        end_idx = entity['end']
        entity_label = entity['entity']
        
        current_pos = 0
        
        for word_idx, token in enumerate(tokens):
            token_start = current_pos
            token_end = current_pos + len(token)
            
            if (start_idx < token_end and end_idx > token_start):
                token_labels[word_idx] = entity_label
                word_found = True
                break
            
            current_pos = token_end + 1
        
    return token_labels

def predict_labels_batch(tokens_list, ner_pipe):
    predicted_labels = []
    
    batch_size = 16
    
    for i in tqdm(range(0, len(tokens_list), batch_size), desc="Predicting"):
        batch = tokens_list[i:i+batch_size]
        batch_texts = [" ".join(tokens) for tokens in batch]
        
        batch_results = ner_pipe(batch_texts)
        
        if not isinstance(batch_results[0], list):
            batch_results = [batch_results]
        
        for j, (tokens, ner_results) in enumerate(zip(batch, batch_results)):
            aligned_labels = align_predictions_with_tokens(tokens, ner_results, ner_pipe.tokenizer)
            predicted_labels.append(aligned_labels)
    
    return predicted_labels

def predict_labels_token_level(tokens_list, ner_pipe):
    predicted_labels = []
    tokenizer = ner_pipe.tokenizer
    
    for tokens in tqdm(tokens_list, desc="Predicting"):
        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        
        word_ids = encoded.word_ids()
        
        text = " ".join(tokens)
        ner_results = ner_pipe(text)
        
        token_labels = ["O"] * len(tokens)
        
        for result in ner_results:
            entity_label = result.get('entity', result.get('label', ''))
            token_idx = result.get('index', 0) - 1
            
            if token_idx < len(word_ids) and word_ids[token_idx] is not None:
                word_idx = word_ids[token_idx]
                if word_idx < len(tokens):
                    if token_labels[word_idx] == "O" or entity_label.startswith('B-'):
                        token_labels[word_idx] = entity_label
        
        predicted_labels.append(token_labels)
    
    return predicted_labels

def main():
    dev_x_path = "en-ner-conll-2003/dev-0/in.tsv"
    dev_y_path = "en-ner-conll-2003/dev-0/expected.tsv"
    
    print("Loading data...")
    tokens_list, true_labels = load_data(dev_x_path, dev_y_path)
    
    print(f"Loaded {len(tokens_list)} sentences")
    print(f"Sample tokens: {tokens_list[0][:10]}")
    print(f"Sample labels: {true_labels[0][:10]}")
    
    model_options = [
        "dbmdz/bert-large-cased-finetuned-conll03-english",
        "dslim/bert-base-NER",
        "Jean-Baptiste/roberta-large-ner-english"
    ]
    
    model_name = model_options[0]
    print(f"Loading model: {model_name}")
    
    ner_pipe = pipeline(
        "ner", 
        model=model_name, 
        tokenizer=model_name,
        aggregation_strategy=None,
        device=0 if torch.cuda.is_available() else -1
    )
    
    print("Predicting labels...")
    
    print("Debugging model output format...")
    test_text = " ".join(tokens_list[0])
    test_results = ner_pipe(test_text)
    print(f"Sample model output: {test_results[:2] if len(test_results) > 0 else 'No entities found'}")
    
    predicted_labels = predict_labels_token_level(tokens_list, ner_pipe)
    
    for i, (pred, true) in enumerate(zip(predicted_labels, true_labels)):
        if len(pred) != len(true):
            print(f"Length mismatch at index {i}: pred={len(pred)}, true={len(true)}")
            if len(pred) < len(true):
                predicted_labels[i] = pred + ["O"] * (len(true) - len(pred))
            else:
                predicted_labels[i] = pred[:len(true)]
    
    print("\nEvaluation Results:")
    print("F1 score:", f1_score(true_labels, predicted_labels))
    print("\nClassification report:")
    print(classification_report(true_labels, predicted_labels))

if __name__ == "__main__":
    main()
