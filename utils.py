import numpy as np

from itertools import permutations
from scipy.spatial.distance import cdist
from tqdm import tqdm
from numpy.random import randint

from keras.preprocessing.image import load_img, img_to_array

def path_to_array(path):
    img = load_img(path)
    img = img_to_array(img)
    img /= 255.
    return img

def create_jigsaw(img, permutations, jitter=2, sub_size=225, cell_size=75, tile_size=64):
    # 30% chance to render image into grayscale
    # if the image is rendered into grayscale, omit jitter
    if np.random.uniform() < 0.3:
        jitter = 0
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        img = np.dstack((img,)*3)

    sub_size = sub_size + (jitter * 2)
    max_offset = img.shape[0] - sub_size

    y_offset = randint(0, max_offset)
    x_offset = randint(0, max_offset)
    sub = img[y_offset:y_offset+sub_size, x_offset:x_offset+sub_size, :]

    tile_offset = cell_size - tile_size

    tiles = []

    for row in range(3):
        for col in range(3):
            y_offset = cell_size * row + jitter
            x_offset = cell_size * col + jitter
            
            cell = []
            for channel in range(3):
                xj = randint(-jitter, jitter+1)
                yj = randint(-jitter, jitter+1)
                this_y_start = y_offset+yj
                this_x_start = x_offset+xj
                this_y_stop = this_y_start+cell_size
                this_x_stop = this_x_start+cell_size
                cell.append(sub[this_y_start:this_y_stop, this_x_start:this_x_stop, channel])
            cell = np.dstack(cell)
            y_offset = randint(0, tile_offset)
            x_offset = randint(0, tile_offset)

            tile = cell[y_offset:y_offset+tile_size, x_offset:x_offset+tile_size, :].copy()
            
            # normalizing by channel
            # note: if a patch is all the same color, std will be 0, leading to a divide by 0 error
            for k in range(3):
                std = tile[:, :, k].std()
                if std == 0:
                    tile[:, :, k] = tile[:, :, k] - tile[:, :, k].mean()
                else:
                    tile[:, :, k] = (tile[:, :, k] - tile[:, :, k].mean())/(tile[:, :, k].std())
            
            tiles.append(tile)

    tiles = np.array(tiles)
    return tiles[permutations, :, :, :]

def max_permutations(n=100):
    print("Generating set of {} permutations.".format(n))
    P = list(permutations(range(9)))
    P_bar = []
    j = randint(0, len(P))
    P_bar.append(P.pop(j))

    for i in tqdm(range(n-1)):
        D = cdist(np.array(P_bar), np.array(P), "hamming")
        D_bar = np.ones((1, D.shape[0])).dot(D)
        j = np.argmax(D_bar[0])
        P_bar.append(P.pop(j))

    print("Done.")
    D = cdist(np.array(P_bar), np.array(P_bar), "hamming")
    print("Average hamming distance: {}".format(np.round(D[np.where(D!=0)].mean() * 9, 2)))
    print("Minimum hamming distance: {}".format(np.round(D[np.where(D!=0)].min() * 9, 2)))
    return P_bar