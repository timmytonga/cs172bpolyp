import random
import torch
import numpy as np

SEED = 35069
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

print("Seed Set :", SEED )

