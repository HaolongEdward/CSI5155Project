

# t-test for independent samples
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import sem
from scipy.stats import t
 

def Nemenyi_Test():
	from scipy.stats import friedmanchisquare
	import numpy as np 
	import scipy.stats as ss
	import Orange
	import matplotlib
	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt

	# knn = [0.662, 0.662, 0.679, 0.679, 0.662, 0.662, 0.679, 0.679, 0.591, 0.591, 0.627, 0.627, 0.591, 0.591, 0.626, 0.626, 0.657, 0.657, 0.69, 0.691, 0.657, 0.657, 0.682, 0.682, 0.672, 0.672, 0.727, 0.727, 0.672, 0.672, 0.728, 0.728]

	# dt = [0.633, 0.633, 0.647, 0.646, 0.633, 0.632, 0.648, 0.647, 0.633, 0.632, 0.668, 0.668, 0.632, 0.633, 0.669, 0.669, 0.63, 0.63, 0.656, 0.654, 0.628, 0.628, 0.652, 0.65, 0.628, 0.629, 0.69, 0.69, 0.629, 0.629, 0.692, 0.691]

	# nb = [0.416, 0.424, 0.426, 0.434, 0.416, 0.424, 0.428, 0.435, 0.417, 0.426, 0.433, 0.439, 0.417, 0.426, 0.433, 0.439, 0.676, 0.676, 0.697, 0.698, 0.676, 0.676, 0.694, 0.694, 0.675, 0.675, 0.722, 0.721, 0.675, 0.675, 0.722, 0.721]

	# rb = [0.653,0.654,0.671,0.67,0.656,0.653,0.671,0.671,0.654,0.654,0.688,0.688,0.653,0.653,0.688,0.688,0.646,0.646,0.678,0.678,0.647,0.646,0.671,0.67,0.645,0.646,0.687,0.687,0.647,0.647,0.689,0.69]

	# mlp = [0.333, 0.133, 0.0, 0.0, 0.133, 0.267, 0.463, 0.066, 0.641, 0.64, 0.646, 0.663, 0.637, 0.639, 0.663, 0.665, 0.066, 0.066, 0.0, 0.0, 0.0, 0.265, 0.0, 0.0, 0.724, 0.72, 0.756, 0.757, 0.722, 0.72, 0.757, 0.76]

	# stat, p = friedmanchisquare(knn, dt, nb, rb, mlp)
	# matrix = np.array([knn, dt, nb, rb, mlp])
	# names = ['knn', 'dt', 'nb', 'rb', 'mlp']


	voting = [0.642, 0.644, 0.658, 0.659, 0.641, 0.642, 0.658, 0.661, 0.576, 0.577, 0.605, 0.609, 0.575, 0.577, 0.604, 0.606, 0.685, 0.685, 0.713, 0.714, 0.685, 0.686, 0.708, 0.708, 0.686, 0.687, 0.734, 0.734, 0.686, 0.687, 0.735, 0.735]

	dt_ada_boosting = [0.664, 0.659, 0.675, 0.678, 0.664, 0.659, 0.676, 0.674, 0.664, 0.659, 0.702, 0.697, 0.664, 0.659, 0.701, 0.703, 0.657, 0.652, 0.688, 0.684, 0.657, 0.652, 0.679, 0.681, 0.657, 0.652, 0.714, 0.716, 0.657, 0.652, 0.717, 0.716]

	nb_ada_boosting = [0.508, 0.513, 0.415, 0.512, 0.508, 0.513, 0.469, 0.378, 0.414, 0.41, 0.378, 0.414, 0.414, 0.41, 0.412, 0.384, 0.407, 0.472, 0.219, 0.495, 0.407, 0.472, 0.476, 0.581, 0.37, 0.569, 0.481, 0.504, 0.37, 0.569, 0.506, 0.523]

	stat, p = friedmanchisquare(voting, dt_ada_boosting, nb_ada_boosting)
	matrix = np.array([voting, dt_ada_boosting, nb_ada_boosting])
	names = ['Voting', 'DT-Ada-Boosting', 'NB-Ada-Boosting']


	print('Statistics=%.3f, p=%.3f' % (stat, p))
	# interpret
	alpha = 0.05
	if p > alpha:
		print('Same distributions (fail to reject H0)')
	else:
		print('Different distributions (reject H0)')

	matrix = matrix.transpose()
	matrix_ranking = []
	# >>> a = [1,2,3,4,3,2,3,4]
	# >>> rankdata([-1 * i for i in row]).astype(int)
	for row in matrix:
		matrix_ranking.append(list(ss.rankdata([-1 * i for i in row])))
		# print(ss.rankdata(row))
	matrix_ranking = np.array(matrix_ranking).transpose()


	avranks = np.average(matrix_ranking, axis=1)
	print(avranks)
	cd = Orange.evaluation.compute_CD(avranks, 32) #tested on 30 datasets
	Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
	plt.savefig('Ensumble ML Critical difference diagram.pdf')


# function for calculating the t-test for two independent samples
def dependent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = mean(data1), mean(data2)
	# number of paired samples
	n = len(data1)
	# sum squared difference between observations
	d1 = sum([(data1[i]-data2[i])**2 for i in range(n)])
	# sum difference between observations
	d2 = sum([data1[i]-data2[i] for i in range(n)])
	# standard deviation of the difference between means
	sd = sqrt((d1 - (d2**2 / n)) / (n - 1))
	# standard error of the difference between the means
	sed = sd / sqrt(n)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = n - 1
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p
 

def paired_t():
	# seed the random number generator
	# seed(1)
	# generate two independent samples
	gender_single_label = [0.842, 0.851, 0.858, 0.853, 0.857, 0.849, 0.856, 0.859, 0.849, 0.856, 0.853]
	gender_multi_label = [0.852, 0.858, 0.847, 0.851, 0.854, 0.852, 0.857, 0.853, 0.848, 0.855, 0.752]

	# calculate the t test
	alpha = 0.05
	t_stat, df, cv, p = dependent_ttest(gender_single_label, gender_multi_label, alpha)
	print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
	# interpret via critical value

	print('singel label gender v.s. multi label gender learning result')
	if abs(t_stat) <= cv:
		print('Accept null hypothesis that the means are equal.')
	else:
		print('Reject the null hypothesis that the means are equal.')
	# interpret via p-value
	if p > alpha:
		print('Accept null hypothesis that the means are equal.')
	else:
		print('Reject the null hypothesis that the means are equal.')

	cardio_single_label = [0.754, 0.747, 0.755, 0.758, 0.742, 0.763, 0.773, 0.76, 0.761, 0.739, 0.755]
	cardio_multi_label = [0.743, 0.741, 0.752, 0.747, 0.755, 0.757, 0.746, 0.761, 0.764, 0.759, 0.752]

	t_stat, df, cv, p = dependent_ttest(cardio_single_label, cardio_multi_label, alpha)
	print('singel label cardio v.s. multi label cardio learning result')
	print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
	# interpret via critical value
	if abs(t_stat) <= cv:
		print('Accept null hypothesis that the means are equal.')
	else:
		print('Reject the null hypothesis that the means are equal.')
	# interpret via p-value
	if p > alpha:
		print('Accept null hypothesis that the means are equal.')
	else:
		print('Reject the null hypothesis that the means are equal.')


def main():
	Nemenyi_Test()
	

if __name__ == '__main__':
	main()









