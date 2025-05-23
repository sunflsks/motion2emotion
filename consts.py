JOINT_IDS = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "RHip": 8,
    "RKnee": 9,
    "RAnkle": 10,
    "LHip": 11,
    "LKnee": 12,
    "LAnkle": 13,
    "REye": 14,
    "LEye": 15,
    "REar": 16,
    "LEar": 17,
}

RELIABLE_JOINTS = {
    "Nose" : 0,
    "Neck" : 1,
    "RShoulder" : 2,
    "LShoulder" : 5,
    "REye" : 14,
    "LEye" : 15
}

CONFIDENCE_THRESHOLD = 0.25
TARGET_LEN = 64 # target length of frames per sequence
