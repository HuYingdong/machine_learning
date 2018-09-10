
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # 随机森林


# 数据压缩


def mypca(x, n):
    x = x.astype(np.double)
    assert(n < x.shape[1])
    print('Input array shape: ', x.shape)
    print('Number of principal components: {0}'.format(n))
    cov = np.dot(x.transpose(), x)/x.shape[0]
    w, v = np.linalg.eigh(cov)
    start_index = x.shape[1] - n
    sum_of_eigenvalues = np.sum(w)
    wn = w[start_index:]
    sum_of_principal_eigenvalues = np.sum(wn)
    percentage = sum_of_principal_eigenvalues/sum_of_eigenvalues
    print('Percentage of variance retained: {0:.1f}%'.format(100*percentage))
    eig_v = v[:, start_index:]
    output = np.dot(x, eig_v)
    print('Output array shape: ', output.shape)
    return output


avg = np.load('log_avg.npy')
std = np.load('log_std.npy')
num = np.load('log_num.npy')

plt.plot(num, avg, 'b-', num, avg+std, 'g:', num, avg-std, 'g:')  # b r y g k w 颜色，- ： -- 线条
plt.grid(True)  # 网格
plt.title('Random forest classification rate')
plt.xlabel('Number of estimators')
plt.ylabel('Classification rate (%)')
plt.show()
