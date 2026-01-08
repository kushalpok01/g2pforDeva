import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pickle

# ---------------- DEVICE ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ---------------- LOAD VOCAB ----------------
with open("hindidata/src_vocab.pkl", "rb") as f:
    src_vocab = pickle.load(f)
with open("hindidata/tgt_vocab.pkl", "rb") as f:
    tgt_vocab = pickle.load(f)

inv_tgt = {v: k for k, v in tgt_vocab.items()}

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
            batch_first=True
        )
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, tgt_mask=None):
        src_emb = self.src_emb(src)
        tgt_emb = self.tgt_emb(tgt)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc(out)

# ---------------- LOAD MODEL ----------------
model = TransformerG2P(len(src_vocab), len(tgt_vocab)).to(DEVICE)
model.load_state_dict(torch.load("hindi_g2p.pt", map_location=DEVICE))
model.eval()

# ---------------- HELPERS ----------------
def encode_word(word):
    return [src_vocab["<s>"]] + [src_vocab[c] for c in word] + [src_vocab["</s>"]]

def preprocess_batch(words):
    seqs = [torch.tensor(encode_word(list(w))) for w in words]
    return pad_sequence(seqs, batch_first=True, padding_value=0).to(DEVICE)

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask.to(DEVICE)

# ---------------- BATCH INFERENCE ----------------
def batch_nepali_to_ipa(words, max_len_extra=10):
    batch_size = len(words)
    src = preprocess_batch(words)

    # Encode through encoder
    with torch.no_grad():
        encoder_out = model.src_emb(src)
    
    # Start decoder with <s>
    tgt_seq = torch.ones(batch_size, 1, dtype=torch.long, device=DEVICE) * tgt_vocab["<s>"]
    finished = [False] * batch_size
    outputs = [[] for _ in range(batch_size)]

    for _ in range(max(len(w) for w in words) + max_len_extra):
        tgt_mask = generate_square_subsequent_mask(tgt_seq.size(1))
        dec_out = model.transformer.encoder(encoder_out)  # encoder output
        dec_out = model.transformer.decoder(model.tgt_emb(tgt_seq), dec_out, tgt_mask=tgt_mask)
        logits = model.fc(dec_out[:, -1, :])  # take last token
        next_ids = logits.argmax(-1)

        for i, nid in enumerate(next_ids.tolist()):
            if not finished[i]:
                if nid == tgt_vocab["</s>"]:
                    finished[i] = True
                else:
                    outputs[i].append(inv_tgt[nid])

        if all(finished):
            break

        tgt_seq = torch.cat([tgt_seq, next_ids.unsqueeze(1)], dim=1)

    return ["".join(o) for o in outputs]

# ---------------- USAGE ----------------
if __name__ == "__main__":
    
    words = [
"बढ़ना",
"बढ़ता",
"बढ़ती",
"बढ़ते",
"बढ़ाया",
"बढ़ाए",
"बढ़ाई",
"बढ़ाव",
"बढ़ोतरी",
"विकास",
"विकसित",
"विकसित करना",
"उन्नति",
"प्रगति",
"प्रगतिशील",
"विस्तार",
"विस्तारित",
"विस्तृत",
"उत्थान",
"उभार",
"वृद्धि",
"वृद्धिशील",
"संवर्धन",
"संवर्धित",
"संवर्धित करना",
"फैलाव",
"फैलाना",
"फैला हुआ",
"तेज़ी",
"तेज़",
"तेज़ करना",
"तेज़ी से",
"उच्च",
"ऊँचा",
"ऊँचाई",
"उठान",
"उत्कर्ष",
"उत्कृष्ट",
"आगे बढ़ना",
"आगे बढ़ाना",
"सुधार",
"सुधारना",
"सुधरा हुआ",
"बढ़चढ़",
"बढ़-चढ़कर",
"आरोह",
"आरोहण",
"चढ़ना",
"चढ़ाव",
"चढ़ती",
"चढ़ता",
"चढ़ते",
"सशक्त",
"सशक्त करना",
"मजबूत",
"मजबूती",
"बल",
"बलवान",
"प्रभाव",
"प्रभावी",
"प्रभाव बढ़ाना",
"समृद्धि",
"समृद्ध",
"उत्पादन",
"उत्पादक",
"उत्पादकता",
"उन्नत",
"उन्नयन",
"विकसन",
"विकसित होना",
"फूलना",
"फलना",
"फलना-फूलना",
"आवर्धन",
"आवर्धित",
"आगे",
"आगे की ओर",
"ऊर्ध्व",
"ऊर्ध्वगामी",
"वर्धन",
"वर्धित",
"वर्धमान",
"अभिवृद्धि"
]
    predictions = batch_nepali_to_ipa(words)
    for w, p in zip(words, predictions):
        print(f"{w} → {p}")
