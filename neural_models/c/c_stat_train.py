import pickle
import matplotlib.pyplot as plt

with open("checkout/c-train.pkl", "rb") as fin:
    train_samples = pickle.load(fin)

sample_len = []
for sample in train_samples:
    sample_len.append(len(sample.token_values))

plt.hist(sample_len, bins=100)
plt.savefig("checkout/c-train.png", dpi=300, bbox_inches="tight")
