import numpy as np
import csv
import faiss
import pickle
import pandas as pd
from scipy.spatial import distance


class Search:
	def __init__(self, path):
		#the cvs file's path
		self.path = path

	def search(self, queryFeatures, limit=20):
		results = dict()
		#opening the csv file
		with open(self.path) as f:
			reader = csv.reader(f)
			#for each element in the csv file
			for row in reader:
				#separiting the the image Name from features, and computing the chi-squared distance.

				features = [float(x) for x in row[0:3800]]
				d = self.chi_squared_distance(features, queryFeatures)
				results[row[3800]] = d

				# features = [float(x) for x in row[1:]]
				# d = self.chi_squared_distance(features, queryFeatures)
				# results[row[0]] = d
			f.close()

		# dictionarry sort
		results = sorted(
			[(v,k) for (k,v) in results.items()]
		)

		return results[:limit]


	def chi_squared_distance(self, histA, histB):
		d = distance.euclidean(histA, histB)
		return d

class SearchFaiss:
	def __init__(self, path):
		#the cvs file's path
		self.path = path

	def search(self, queryFeatures):
		df = pd.read_csv(self.path)
		X = df[df.columns[:500]]
		y = df['imgpath']
		des = X.to_numpy()
		imgpath = y.to_numpy()
		l_img_path = list(imgpath)
		des32 = np.float32(des)

		desto = []
		for i in range(len(des)):
			desto.append(np.reshape(des32[i], (-1, 500)))

		distance, indices = self.faiss_EL2(desto, queryFeatures)

		dis = distance[0]
		ind = indices[0]

		img_inds = []
		for file_index in (ind):
			img_inds.append(l_img_path[file_index])
		res = dict(zip(img_inds, dis))
		return res

	def faiss_EL2(self,feuter, query_img):

		index = faiss.IndexFlatL2(500)
		descriptors = np.vstack(feuter)
		index.add(descriptors)
		distance, indices = index.search(query_img, 20)
		return distance,indices