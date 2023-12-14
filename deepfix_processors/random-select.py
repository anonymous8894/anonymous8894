
import deepfixutlis
import random
import os

samples = deepfixutlis.load_all_with_error()
r = random.Random(0)
r.shuffle(samples)

TARGET_DIR = "checkout/random"
os.mkdir(TARGET_DIR)

for i, sample in enumerate(samples):
    with open(os.path.join(TARGET_DIR, f"{i}.c"), "w") as f:
        f.write(sample.code)
    if i == 10:
        break
