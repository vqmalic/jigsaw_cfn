import os
import numpy as np

from keras.utils import Sequence
from utils import path_to_array, create_jigsaw, max_permutations

class JigsawGenerator(Sequence):
	def __init__(self, imagedir_path, batch_size, jitter, permutations):
		self.image_filenames = [os.path.join(imagedir_path, x) for x in os.listdir(imagedir_path)]
		self.permutations=permutations
		self.jitter = jitter
		self.batch_size = batch_size

	def __len__(self):
		return int(np.ceil(len(self.image_filenames) / self.batch_size))

	def __getitem__(self, idx):
		batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
		x_out = [[] for i in range(9)]
		y_out = []

		for path in batch_x:
			img = path_to_array(path)
			p_i = np.random.choice(range(len(self.permutations)))
			y = np.zeros(len(self.permutations))
			y[p_i] = 1
			p = self.permutations[p_i]
			tiles = create_jigsaw(img, p, jitter=self.jitter)

			for i in range(9):
				x_out[i].append(tiles[i, :, :, :])
			y_out.append(y)

		x_out = [np.array(arr) for arr in x_out]

		return x_out, [y_out]
