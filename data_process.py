import os
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from consts import JOINT_IDS
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

prefix = None # prefix - initialized my load_csv

def is_valid_array(array: np.ndarray) -> bool:
    return (array is not None) and (not np.isnan(array).any()) and (not np.isinf(array).any()) and (array.size > 0)

def load_csv(path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame): # returns train and test dataframes, (train, test)
    global prefix
    prefix = path
    df = pd.read_csv(prefix + 'annotations/train.csv')
    (train_df, test_df) = train_test_split(df, test_size=0.2, random_state=42)
    (test_df, validate_df) = train_test_split(test_df, test_size=0.5, random_state=42)
    return (train_df, test_df, validate_df)

def get_joints_for_row(row: pd.Series) -> np.ndarray:
    """ get row joints from given dataframe row"""
    path = Path(row[0])
    filename = path.stem
    videoname = path.parent.name
    full_path = (Path(prefix) / Path('joints') / Path('003') / videoname / filename).with_suffix('.npy')
    if not full_path.exists():
        return None

    return np.load(full_path)

def process_joints(raw: np.ndarray) -> np.ndarray:
    joints = raw[:, 2:].reshape(-1, 18, 3) # 54 -> 18x3, where each joint has (x, y, confidence)

    # kill for now # joints[joints[:, :, 2] < CONFIDENCE_THRESHOLD] = 0 # no change to shape, mask set joints with low confidence to 0
    joints = joints[:, :, :2] # 18x3 -> 18x2, kill confidence column

    '''
    # normalize joint positions against the hips
    center = (joints[:, JOINT_IDS["RHip"], :] + joints[:, JOINT_IDS["LHip"], :]) / 2 # "center" of body
    joints -= center[:, np.newaxis, :] # subtract center from all joints

    # scale body parts against the distance between the shoulders
    shoulder_width = joints[:, JOINT_IDS["RShoulder"], :] - joints[:, JOINT_IDS["LShoulder"], :]
    scale = np.linalg.norm(shoulder_width, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    joints /= scale[:, None, :] # divide all joints by the shoulder width

    # interpolate joints to 64 frames
    original_time = np.linspace(0, 1, joints.shape[0])
    new_time = np.linspace(0, 1, TARGET_LEN)
    pose_resampled = scipy.interpolate.interp1d(original_time, joints, axis=0, kind="linear")(new_time)
    '''

    pose_resampled = joints
    pose_resampled = pose_resampled.reshape(-1, 36) # 18x2 -> 36
    return pose_resampled

'''
def process_joints(raw: np.ndarray) -> np.ndarray:
    joints = raw[:, 2:].copy().reshape(-1, 18, 3)
    joints = joints[:, :, :2].reshape(-1, 18, 2)

    x_coords = joints[:, :, 0]
    y_coords = joints[:, :, 1]

    x_min = x_coords[x_coords > 0].min() if (x_coords > 0).any() else 0
    y_min = y_coords[y_coords > 0].min() if (y_coords > 0).any() else 0
    x_max = x_coords[x_coords > 0].max()
    y_max = y_coords[y_coords > 0].max()

    joints[:, :, 0] = (x_coords - x_min) / (x_max - x_min + 1e-6)
    joints[:, :, 1] = (y_coords - y_min) / (y_max - y_min + 1e-6)

    return joints.reshape(-1, 36)
'''


def get_emotions_for_row(row: pd.Series) -> (np.ndarray, np.ndarray):
    """ get row emotions from given dataframe row"""
    emotions = row[4:30].to_numpy() # 4-30 r emotion columns
    emotional_characteristics = row[30:33].to_numpy() # valence, arousal, dominance
    return (emotions, emotional_characteristics)

def get_X_y(df: pd.DataFrame, cache=True, spoof=False) -> (torch.Tensor, torch.Tensor):
    X = torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(process_joints(get_joints_for_row(row))) for _, row in df.iterrows() if is_valid_array(get_joints_for_row(row))],
            batch_first=True,
            padding_value=0.0
    )
<<<<<<< Updated upstream

    if spoof:
        X = torch.rand(X.shape) * 500
        print(X)

    X = X[:, :120, :] # truncate, just for now.
    y = torch.tensor(np.array([get_emotions_for_row(row)[0] for _, row in df.iterrows() if is_valid_array(get_joints_for_row(row))], dtype=np.float32))

    return (X, y)

class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].to(torch.float32).detach().clone(), self.y[idx].to(dtype=torch.float32).detach().clone()
