import numpy as np

def softmax(x):
	return np.exp(x)/np.sum(np.exp(x),axis=0)

if __name__ == "__main__":
	# a sample
	scores = [1.0, 2.0, 3.0]
	print(softmax(scores))

	# each column represents a sample
	scores = np.array([[1, 2, 3, 6],
	                   [2, 4, 5, 6],
	                   [3, 8, 7, 6]])
	print(softmax(scores))
