import sent2vec
import pandas as pd
import numpy as np

model = sent2vec.Sent2vecModel()
model.load_model('model.bin')

path = input("Enter path of input file: \n")
df = pd.read_csv(path, sep='\,\|\,', engine='python', header=None)
prompts = df.iloc[:,0]
labels = df.iloc[:,1]
prompts = prompts.values.tolist()
labels = labels.values.tolist()

embs = []
for prompt in prompts:
    emb = model.embed_sentence(prompt)
    emb = emb.flatten()
    embs.append(emb)
# embs = model.embed_sentences([prompts])
embs = np.array(embs)
embs = pd.DataFrame(embs)
path = input('Enter path of output file: \n')
embs.to_csv('data/s2v_sms_test_0_features.csv',index=False, header=False)
