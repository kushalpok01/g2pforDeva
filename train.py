# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import pickle
# from tqdm import tqdm

# # ---------------- DEVICE ----------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", DEVICE)

# # ---------------- DATASET ----------------

# class G2PDataset(Dataset):
#     def __init__(self, path):
#         self.data = []
#         with open(path, encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue  # skip empty lines
#                 parts = line.split("\t")
#                 if len(parts) < 2:
#                     continue  # skip malformed lines
#                 w, ipa = parts[0], parts[1]
#                 self.data.append((list(w), list(ipa)))


#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# def collate_fn(batch, src_vocab, tgt_vocab):
#     src_seqs = []
#     tgt_seqs = []

#     for src, tgt in batch:
#         src_ids = [src_vocab["<s>"]] + [src_vocab[c] for c in src] + [src_vocab["</s>"]]
#         tgt_ids = [tgt_vocab["<s>"]] + [tgt_vocab[c] for c in tgt] + [tgt_vocab["</s>"]]
#         src_seqs.append(torch.tensor(src_ids))
#         tgt_seqs.append(torch.tensor(tgt_ids))

#     src_seqs = nn.utils.rnn.pad_sequence(src_seqs, padding_value=0)
#     tgt_seqs = nn.utils.rnn.pad_sequence(tgt_seqs, padding_value=0)
#     return src_seqs.to(DEVICE), tgt_seqs.to(DEVICE)

# # ---------------- LOAD DATA ----------------
# train_data = G2PDataset("data/train.tsv")
# val_data = G2PDataset("data/val.tsv")

# # ---------------- BUILD VOCAB ----------------
# def build_vocab(seqs):
#     vocab = {"<pad>":0, "<s>":1, "</s>":2}
#     for seq in seqs:
#         for s in seq:
#             if s not in vocab:
#                 vocab[s] = len(vocab)
#     return vocab

# src_vocab = build_vocab([x[0] for x in train_data])
# tgt_vocab = build_vocab([x[1] for x in train_data])

# # Save vocab
# with open("data/src_vocab.pkl", "wb") as f:
#     pickle.dump(src_vocab, f)
# with open("data/tgt_vocab.pkl", "wb") as f:
#     pickle.dump(tgt_vocab, f)
# print("Saved src and tgt vocabularies.")

# # ---------------- MODEL ----------------
# class TransformerG2P(nn.Module):
#     def __init__(self, src_vocab_size, tgt_vocab_size):
#         super().__init__()
#         self.src_emb = nn.Embedding(src_vocab_size, 256)
#         self.tgt_emb = nn.Embedding(tgt_vocab_size, 256)
#         self.transformer = nn.Transformer(
#             d_model=256,
#             nhead=4,
#             num_encoder_layers=4,
#             num_decoder_layers=4
#         )
#         self.fc = nn.Linear(256, tgt_vocab_size)

#     def forward(self, src, tgt):
#         src = self.src_emb(src)
#         tgt = self.tgt_emb(tgt)
#         out = self.transformer(src, tgt)
#         return self.fc(out)

# model = TransformerG2P(len(src_vocab), len(tgt_vocab)).to(DEVICE)

# # ---------------- TRAINING ----------------
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# loss_fn = nn.CrossEntropyLoss(ignore_index=0)

# train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
#                           collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab))
# val_loader = DataLoader(val_data, batch_size=32, shuffle=False,
#                         collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab))

# EPOCHS = 50

# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0
#     for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
#         optimizer.zero_grad()
#         out = model(src, tgt[:-1])
#         loss = loss_fn(out.view(-1, out.size(-1)), tgt[1:].view(-1))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1} - Training Loss: {total_loss/len(train_loader):.4f}")

# # ---------------- SAVE MODEL ----------------
# torch.save(model.state_dict(), "nepali_g2p.pt")
# print("Model saved to nepali_g2p.pt")


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm

# ---------------- DEVICE ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ---------------- DATASET ----------------
class G2PDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                w, ipa = parts[0], parts[1]
                self.data.append((list(w), list(ipa)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, src_vocab, tgt_vocab):
    src_seqs = []
    tgt_seqs = []

    for src, tgt in batch:
        src_ids = [src_vocab["<s>"]] + [src_vocab[c] for c in src] + [src_vocab["</s>"]]
        tgt_ids = [tgt_vocab["<s>"]] + [tgt_vocab[c] for c in tgt] + [tgt_vocab["</s>"]]
        src_seqs.append(torch.tensor(src_ids))
        tgt_seqs.append(torch.tensor(tgt_ids))

    # pad_sequence with batch_first=True
    src_seqs = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    tgt_seqs = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=0)
    return src_seqs.to(DEVICE), tgt_seqs.to(DEVICE)

# ---------------- LOAD DATA ----------------
train_data = G2PDataset("hindidata/train.tsv")
val_data = G2PDataset("hindidata/val.tsv")

# ---------------- BUILD VOCAB ----------------
def build_vocab(seqs):
    vocab = {"<pad>":0, "<s>":1, "</s>":2}
    for seq in seqs:
        for s in seq:
            if s not in vocab:
                vocab[s] = len(vocab)
    return vocab

src_vocab = build_vocab([x[0] for x in train_data])
tgt_vocab = build_vocab([x[1] for x in train_data])

# Save vocab
with open("hindidata/src_vocab.pkl", "wb") as f:
    pickle.dump(src_vocab, f)
with open("hindidata/tgt_vocab.pkl", "wb") as f:
    pickle.dump(tgt_vocab, f)
print("Saved src and tgt vocabularies.")

# ---------------- MODEL ----------------
class TransformerG2P(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True  # use batch_first to avoid warnings
        )
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, tgt_mask=None):
        src_emb = self.src_emb(src)
        tgt_emb = self.tgt_emb(tgt)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc(out)

# ---------------- CREATE MODEL ----------------
model = TransformerG2P(len(src_vocab), len(tgt_vocab)).to(DEVICE)

# ---------------- MASK GENERATOR ----------------
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

# ---------------- TRAINING ----------------
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab))
val_loader = DataLoader(val_data, batch_size=32, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab))

EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        tgt_input = tgt[:, :-1]   # exclude </s>
        tgt_output = tgt[:, 1:]   # shifted left

        # generate mask
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(DEVICE)

        out = model(src, tgt_input, tgt_mask=tgt_mask)
        loss = loss_fn(out.view(-1, out.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Training Loss: {total_loss/len(train_loader):.4f}")

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "hindi_g2p.pt")
print("Model saved to hindi_g2p.pt")
