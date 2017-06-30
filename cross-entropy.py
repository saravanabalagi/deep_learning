import numpy as np
from softmax import softmax

def cross_entropy(x,y):
	sum = 0
	for i in range(0, len(x)):
		sum += y[i] * np.log(x[i])
	return -sum

if __name__ == "__main__":
	legit_scores = [3.0, 1.0, 2.0]
	softmax_scores = softmax(legit_scores)
	one_hot_encoding = [1, 0 ,0]

	print(softmax_scores)
	print(cross_entropy(softmax_scores, one_hot_encoding))