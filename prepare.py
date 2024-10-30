"""
Prepare the Zulu dataset for language modelling.
Encoding using SentencePiece unigram model.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import pickle
import numpy as np
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file = "insert_unigram_model_name")

with open('insert_training_set_name', 'r', encoding="utf-8") as file:
    lines = file.readlines()
train_ids = []
for line in lines:
    train_ids.extend(sp.tokenize(line))

with open('insert_validation_set_name', 'r', encoding="utf-8") as file:
    lines = file.readlines()
val_ids = []
for line in lines:
    val_ids.extend(sp.tokenize(line))


print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
# train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile('val.bin')

with open('train.bin', 'wb') as f:
    for i in range(0, train_ids.shape[0], 500000):
        # Select the batch
        batch = train_ids[i:i + 500000]
        # Write the batch to the file
        batch.tofile(f)

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': 10000,
}
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)