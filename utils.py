import math
import numpy as np

def sigmoid(logit: int):
    return (1 / (1 + math.exp(-logit)))

def frame_to_image(prefix: str, processed: bool, batch: int, frames: np.ndarray):
    global count
    ''' frames must be of wtv x 18 x 2 (18 x 2 are the joints)'''
    if not processed:
        frames = frames[:, 2:].reshape(-1, 18, 3)
        frames = frames[:, :, :2]
    else:
        frames = frames.reshape(-1, 18, 2)

    for i in range(frames.shape[0]):
        frame = np.ceil(frames[i]).astype(int)
        coords = tuple(np.max(frame, axis=0) + 20) # the maximum coords along the x and y axes; give us the biggest coords we need.
        imgbuf = np.zeros(coords, dtype=bool)

        # print count of 1s in array
        unique, counts = np.unique(frame, return_counts=True)

        lis = frame.tolist()
        for x in range(10, coords[0]):
            for y in range(10, coords[1]):
                if [x,y] in lis:
                    imgbuf[x-2:x+2, y-2:y+2] = True

        img = Image.fromarray(imgbuf)
        path = f"/Users/sunchipnacho/Temp/pics/{prefix}/{i}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)
        count += 1
