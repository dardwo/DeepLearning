import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import f1_score
from tqdm import tqdm

class CorpusLoader:
    def __init__(self, path_train, path_dev_x, path_dev_y):
        print("Reading dataset...")
        self.train_df = pd.read_csv(path_train, sep='\t', names=['label', 'text'])
        self.dev_inputs = pd.read_csv(path_dev_x, sep='\t', names=['text'])
        self.dev_targets = pd.read_csv(path_dev_y, sep='\t', names=['label'])

        # Token splitting
        self.train_df['tokens'] = self.train_df['text'].apply(lambda x: x.split())
        self.train_df['tags'] = self.train_df['label'].apply(lambda x: x.split())
        self.dev_inputs['tokens'] = self.dev_inputs['text'].apply(lambda x: x.split())
        self.dev_targets['tags'] = self.dev_targets['label'].apply(lambda x: x.split())

        self._validate_samples()

        # Build dictionaries
        self.token_to_index = self._build_vocab(self.train_df['tokens'], pad='<PAD>', unk='<UNK>')
        self.label_to_index = self._build_vocab(self.train_df['tags'], pad='<PAD>')

        print(f"Tokens: {len(self.token_to_index)}, Labels: {len(self.label_to_index)}")
        print(f"Training examples: {len(self.train_df)}, Dev examples: {len(self.dev_inputs)}")

    def _validate_samples(self):
        inconsistent = sum(len(t) != len(l) for t, l in zip(self.train_df['tokens'], self.train_df['tags']))
        if inconsistent:
            print(f"[!] {inconsistent} mismatched token-label sequences detected.")

    def _build_vocab(self, sequences, pad='<PAD>', unk=None):
        vocab = {pad: 0}
        if unk:
            vocab[unk] = 1
        for seq in sequences:
            for item in seq:
                if item not in vocab:
                    vocab[item] = len(vocab)
        return vocab


class TaggingDataset(Dataset):
    def __init__(self, tokens, tags, vocab, tag_map):
        self.tokens = tokens
        self.tags = tags
        self.vocab = vocab
        self.tag_map = tag_map

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = [self.vocab.get(t, self.vocab['<UNK>']) for t in self.tokens[idx]]
        y = [self.tag_map[tag] for tag in self.tags[idx]]
        return x, y


def pad_batch(seqs):
    max_len = max(len(s) for s in seqs)
    padded, masks = [], []
    for s in seqs:
        mask = [1]*len(s) + [0]*(max_len - len(s))
        s += [0]*(max_len - len(s))
        padded.append(s)
        masks.append(mask)
    return torch.tensor(padded), torch.tensor(masks)


def collate_batch(batch):
    xs, ys = zip(*batch)
    x_pad, mask = pad_batch(list(xs))
    y_pad, _ = pad_batch(list(ys))
    return x_pad, y_pad, mask


class BiLSTMNER(nn.Module):
    def __init__(self, n_tokens, n_labels, emb_dim=100, hid_dim=256, n_layers=1, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hid_dim, n_labels)

    def forward(self, x):
        emb = self.dropout(self.emb(x))
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.dropout(lstm_out)
        return torch.log_softmax(self.out(lstm_out), dim=2)


class SequenceTrainer:
    def __init__(self, model, loaders, tag_map, device):
        self.model = model.to(device)
        self.train_loader, self.dev_loader = loaders
        self.tag_map = tag_map
        self.device = device

        self.rev_map = {i: t for t, i in tag_map.items()}
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.metrics = {'loss': [], 'f1': []}

    def run_epoch(self):
        self.model.train()
        epoch_loss = 0

        for x, y, m in tqdm(self.train_loader, desc='Training'):
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred.view(-1, pred.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        all_pred, all_true = [], []

        with torch.no_grad():
            for x, y, m in tqdm(self.dev_loader, desc='Evaluating'):
                x, y, m = x.to(self.device), y.to(self.device), m.to(self.device)
                logits = self.model(x)
                pred_tags = torch.argmax(logits, dim=2)

                for i in range(x.size(0)):
                    true_seq, pred_seq = [], []
                    for j in range(x.size(1)):
                        if m[i][j] == 0:
                            continue
                        true_seq.append(self.rev_map[y[i][j].item()])
                        pred_seq.append(self.rev_map[pred_tags[i][j].item()])
                    all_true.append(true_seq)
                    all_pred.append(pred_seq)

        score = f1_score(all_true, all_pred)
        self.metrics['f1'].append(score)
        return score

    def train_model(self, epochs=25, validate_every=5):
        print(f"Starting training for {epochs} epochs")
        best = 0.0

        for e in range(epochs):
            loss = self.run_epoch()
            self.metrics['loss'].append(loss)
            print(f"Epoch {e+1}: Loss = {loss:.4f}")

            if (e + 1) % validate_every == 0:
                f1 = self.validate()
                print(f"Validation F1 = {f1:.4f}")
                if f1 > best:
                    best = f1
                    print(f"[+] New best F1 = {best:.4f}")

        print(f"Training complete. Best validation F1 = {best:.4f}")
        return best


def launch():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    data = CorpusLoader(
        'en-ner-conll-2003/train/train.tsv',
        'en-ner-conll-2003/dev-0/in.tsv',
        'en-ner-conll-2003/dev-0/expected.tsv'
    )

    train_data = TaggingDataset(data.train_df['tokens'], data.train_df['tags'], data.token_to_index, data.label_to_index)
    dev_data = TaggingDataset(data.dev_inputs['tokens'], data.dev_targets['tags'], data.token_to_index, data.label_to_index)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_batch)
    dev_loader = DataLoader(dev_data, batch_size=32, shuffle=False, collate_fn=collate_batch)

    ner_model = BiLSTMNER(len(data.token_to_index), len(data.label_to_index), n_layers=2, dropout=0.3)
    print(f"Model has {sum(p.numel() for p in ner_model.parameters()):,} parameters")

    trainer = SequenceTrainer(ner_model, (train_loader, dev_loader), data.label_to_index, device)
    best = trainer.train_model()

    return ner_model, trainer, best


if _name_ == '_main_':
    launch()