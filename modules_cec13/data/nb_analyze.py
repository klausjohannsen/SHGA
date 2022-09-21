import numpy as np

x = np.loadtxt("nb_10")
xx = np.dot(x, np.transpose(x))
XX = np.fill_diagonal(xx, 0)
m = np.max(xx, axis=1)

m_min = np.min(m)
m_max = np.max(m)
m_mean = np.mean(m)

print("mean: %f" % m_mean)
print("max: %f" % m_max)
print("min: %f" % m_min)
