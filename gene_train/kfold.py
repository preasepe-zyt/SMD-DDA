import random

def kfold(data, test_ratio=0.2, seed=42):
    random.seed(seed)
    
    indices = list(range(len(data)))
    random.shuffle(indices)
    
    split = int(len(data) * (1 - test_ratio))
    
    train_idx = indices[:split]
    test_idx  = indices[split:]
    
    train_data = [data[i] for i in train_idx]
    test_data  = [data[i] for i in test_idx]
    
    return train_data, test_data
