#!/usr/bin/env -S uv run

import sys
import torch
import platform
import numpy as np
from torch import nn
from time import time
from model import Transformer
from utils import evaluating_model
from data_process import load_csv, get_X_y, EmotionDataset, get_joints_for_row, process_joints
from consts import JOINT_IDS, PATIENCE, CONFIDENCE_THRESHOLD
from torch.utils.data import DataLoader
from statistics import mAP_and_mROCAUC

def init_torch():
    device = torch.device("cpu")
    if platform.system() == "Linux":
        device = torch.device("cuda")

    torch.set_default_device(device)

def main() -> int:
    if len(sys.argv) < 2:
        print("Pass path of data as first arg")
        return 1

    init_torch()

    path = sys.argv[1]

    (train_df, test_df, validate_df) = load_csv(path)

    train_X, train_y = get_X_y(train_df)
    test_X, test_y = get_X_y(test_df)
    validate_X, validate_y = get_X_y(validate_df)

    assert not torch.isnan(train_X).any(), "X contains NaN values"
    assert not torch.isnan(test_X).any(), "X contains NaN values"
    assert not torch.isnan(validate_X).any(), "X contains NaN values"

    (batch_count, seq_len, joint_per) = train_X.shape

    train_ds = EmotionDataset(train_X, train_y)
    test_ds = EmotionDataset(test_X, test_y)
    validate_ds = EmotionDataset(validate_X, validate_y)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=torch.Generator(device=torch.get_default_device()))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=True, generator=torch.Generator(device=torch.get_default_device()))
    validate_loader = DataLoader(validate_ds, batch_size=64, shuffle=True, generator=torch.Generator(device=torch.get_default_device()))

    model = nn.DataParallel(Transformer(seq_len=seq_len))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_y_classified = (train_y > CONFIDENCE_THRESHOLD)
    pos_weight = (~train_y_classified.bool()).sum(0) / train_y_classified.bool().sum(0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    positive_rates = train_y.mean(0)
    print(f"Positive rates per class: {positive_rates}")
    print("pos_weight values:", pos_weight)

    print("Starting training...")

    start_train_time = time()
    stop = False
    least_validation_loss = np.inf
    patience_counter = 0
    epoch = 0

    for i in range(3000):
        start_epoch_time = time()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            inputs, targets = batch

            mask = (inputs == 0).all(dim=2)

            optimizer.zero_grad()
            outputs = model(inputs, mask=mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # validation
        all_outputs = []
        all_targets = []
        with evaluating_model(model), torch.no_grad():
            for batch in validate_loader:
                inputs, targets = batch
                mask = (inputs == 0).all(dim=2)

                rinput = torch.randn_like(inputs)

                logits = model(inputs, mask=mask)
                rlogits = model(rinput)

                loss = criterion(logits, targets)
                rloss = criterion(rlogits, targets)
                validation_loss = loss.item()

                all_outputs.append(torch.sigmoid(logits))
                all_targets.append(targets)

        if validation_loss < least_validation_loss:
            least_validation_loss = validation_loss
        else:
            patience_counter += 1

        if patience_counter > PATIENCE:
            stop = True

        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        mAP, mROCAUC = mAP_and_mROCAUC(all_outputs.numpy(), all_targets.numpy())

        print(f"Epoch {epoch+1}. Loss: {total_loss/num_batches:.4f}. Completed in {time() - start_epoch_time} seconds, with a validation loss of {validation_loss}. Real/Rand ratio is currently {rloss/validation_loss:.2f}. For validation set -- (mAP: {mAP}, mROCAUC: {mROCAUC}")

        epoch += 1

    print(f"Training complete. Completed in {time() - start_train_time} seconds.")

    print("testing...")

    total_loss = 0
    batch_cnt = 0

    all_outputs = []
    all_targets = []

    with evaluating_model(model), torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            assert not torch.isnan(inputs).any()
            assert not torch.isnan(targets).any()

            mask = (inputs == 0).all(dim=2)
            outputs = model(inputs, mask=mask)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            batch_cnt += 1

            all_outputs.append(torch.sigmoid(outputs))
            all_targets.append(targets)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        mAP, mROCAUC = mAP_and_mROCAUC(all_outputs.numpy(), all_targets.numpy())
        print(f"FINAL TESTING LOSS: {total_loss/batch_cnt:.4f}. mAP is {mAP}")

if __name__ == "__main__":
    sys.exit(main())
