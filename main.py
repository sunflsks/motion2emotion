#!/usr/bin/env -S uv run

import sys
import torch
import platform
import numpy as np
from torch import nn
from time import time
from model import Transformer
from data_process import load_csv, get_X_y, EmotionDataset
from torch.utils.data import DataLoader

device = torch.device("cpu")
if platform.system() == "Darwin":
    device = torch.device("mps")
elif platform.system() == "Linux":
    device = torch.device("cuda")

def main() -> int:
    if len(sys.argv) < 2:
        print("Pass path of data as first arg")
        return 1

    path = sys.argv[1]

    (train_df, test_df) = load_csv(path)

    train_X, train_y = get_X_y(train_df)
    test_X, test_y = get_X_y(test_df)

    assert not np.isnan(train_X).any(), "X contains NaN values"
    assert not np.isnan(test_X).any(), "X contains NaN values"

    (batch_count, seq_len, joint_per) = train_X.shape

    train_ds = EmotionDataset(train_X, train_y)
    test_ds = EmotionDataset(test_X, test_y)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=True)

    model = nn.DataParallel(Transformer(seq_len=seq_len).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print("Starting training...")
    start_train_time = time()
    for epoch in range(100):
        start_epoch_time = time()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_targets = []

        for batch in train_loader:
            batch = [x.to(device) for x in batch]
            inputs, targets = batch

            mask = (inputs == 0).all(dim=2)

            optimizer.zero_grad()
            outputs = model(inputs, mask=mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            ''' testing !!!
            if int(random.uniform(1, 100)) % 20 == 0:
                sequence = int(random.uniform(1, 32))
                idxs = random.sample(range(0, 26), 4)

                for i in idxs:
                    print(f"ACTUAL: {targets[sequence][i]}. PREDICTED: {sigmoid(outputs[sequence][i])}.")
            end testing '''

            total_loss += loss.item()
            num_batches += 1

        print(f"Epoch {epoch+1}/{100}, Loss: {total_loss/num_batches:.4f}. Completed in {time() - start_epoch_time} seconds.")

    print(f"Training complete. Completed in {time() - start_train_time} seconds.")

    print("testing...")

    total_loss = 0
    batch_cnt = 0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = [x.to(device) for x in batch]
            inputs, targets = batch
            assert not torch.isnan(inputs).any()
            assert not torch.isnan(targets).any()

            mask = (inputs == 0).all(dim=2)
            outputs = model(inputs, mask=mask)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            batch_cnt += 1

    print(f"FINAL TESTING LOSS: {total_loss/batch_cnt:.4f}")


    correct = 0

if __name__ == "__main__":
    sys.exit(main())
