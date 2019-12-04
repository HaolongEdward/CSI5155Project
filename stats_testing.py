from scipy.stats import friedmanchisquare
import numpy as np 
import scipy.stats as ss
import Orange
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def main():
	knn = [0.662, 0.662, 0.679, 0.679, 0.662, 0.662, 0.679, 0.679, 0.591, 0.591, 0.627, 0.627, 0.591, 0.591, 0.626, 0.626, 0.657, 0.657, 0.69, 0.691, 0.657, 0.657, 0.682, 0.682, 0.672, 0.672, 0.727, 0.727, 0.672, 0.672, 0.728, 0.728]

	dt = [0.633, 0.633, 0.647, 0.646, 0.633, 0.632, 0.648, 0.647, 0.633, 0.632, 0.668, 0.668, 0.632, 0.633, 0.669, 0.669, 0.63, 0.63, 0.656, 0.654, 0.628, 0.628, 0.652, 0.65, 0.628, 0.629, 0.69, 0.69, 0.629, 0.629, 0.692, 0.691]

	nb = [0.416, 0.424, 0.426, 0.434, 0.416, 0.424, 0.428, 0.435, 0.417, 0.426, 0.433, 0.439, 0.417, 0.426, 0.433, 0.439, 0.676, 0.676, 0.697, 0.698, 0.676, 0.676, 0.694, 0.694, 0.675, 0.675, 0.722, 0.721, 0.675, 0.675, 0.722, 0.721]

	rb = [0.653,0.654,0.671,0.67,0.656,0.653,0.671,0.671,0.654,0.654,0.688,0.688,0.653,0.653,0.688,0.688,0.646,0.646,0.678,0.678,0.647,0.646,0.671,0.67,0.645,0.646,0.687,0.687,0.647,0.647,0.689,0.69]

	mlp = [0.333, 0.133, 0.0, 0.0, 0.133, 0.267, 0.463, 0.066, 0.641, 0.64, 0.646, 0.663, 0.637, 0.639, 0.663, 0.665, 0.066, 0.066, 0.0, 0.0, 0.0, 0.265, 0.0, 0.0, 0.724, 0.72, 0.756, 0.757, 0.722, 0.72, 0.757, 0.76]

	stat, p = friedmanchisquare(knn, dt, nb, rb, mlp)
	print('Statistics=%.3f, p=%.3f' % (stat, p))
	# interpret
	alpha = 0.05
	if p > alpha:
		print('Same distributions (fail to reject H0)')
	else:
		print('Different distributions (reject H0)')

	matrix = np.array([knn, dt, nb, rb, mlp])
	matrix = matrix.transpose()

	matrix_ranking = []
	for row in matrix:
		matrix_ranking.append(list(ss.rankdata(row)))
		# print(ss.rankdata(row))
	matrix_ranking = np.array(matrix_ranking).transpose()


	names = ['KNN','DT','NB','RB','MLP']
	avranks = np.average(matrix_ranking, axis=1)
	cd = Orange.evaluation.compute_CD(avranks, 32) #tested on 30 datasets
	Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
	plt.savefig('Critical difference diagram.pdf')










if __name__ == '__main__':
	main()









