import numpy as np
import matplotlib.pyplot as plt


class DualPerception:
    def __init__(self, alpha, b, lr, x_train, y_train):
        self.alpha = alpha
        self.b = b
        self.lr = lr
        self.x_train = x_train
        self.y_train = y_train
        self.gram = np.dot(self.x_train, self.x_train.T)

    def loss(self, i):
        return self.y_train[i] * (np.sum(self.alpha * self.y_train * self.gram[:, i]) + self.b)

    def update_para(self, i):
        self.alpha[i] += self.lr
        self.b += self.lr * self.y_train[i]

    def compute_w(self, x, y):
        w = np.dot((self.alpha * y).T, x)
        return w

    def obj_fun(self, x_points, w, b):
        return (x_points * w[1] + b) / (-w[0] + 1e-4)

    def fit(self):
        '''
        首先计算 gram 矩阵
        按顺序选取训练数据和 gram 矩阵中的某一列，若未被正确分类，则更新对应的参数，并不是全部参数
        每次计算错误分类点的个数，如果迭代完全部数据错误分类个数为零，则停止训练
        :return: 训练后的模型和参数 b ，参数 w 需要通过 alpha * y * x 得到
        '''
        flag = False
        while not flag:
            error_points = 0
            for i in range(len(self.x_train)):
                if self.loss(i) <= 0:
                    self.update_para(i)
                    print('Selected error point: xi= {}'.format(self.x_train[i]))
                    error_points += 1
            if error_points == 0:
                flag = True


if __name__ == '__main__':

    x1 = np.array([[3, 3]])
    x2 = np.array([[4, 3]])
    x3 = np.array([[1, 1]])
    x = np.concatenate((x1, x2, x3), axis=0)
    y = np.array([[1], [1], [-1]])
    data = np.append(x, y, axis=1)
    print('data: \n{} \nlabel: \n{}'.format(x, y))

    # w = np.zeros(2)
    alpha = np.zeros(data.shape[0])
    b = np.zeros(1)
    lr = 1
    x_train = data[:, 0:2]
    y_train = data[:, 2]
    perception = DualPerception(alpha, b, lr, x_train, y_train)
    perception.fit()
    w = perception.compute_w(x_train, y_train)

    print('Optimized parameters: \nw: {} \nb: {}'.format(w, b))

    xp = np.linspace(0, max(x[:, 0]), 100)
    for i in data:
        x1, x2, yi = i
        if yi == 1:
            plt.plot(x1, x2, 'ro', label='positive')
        else:
            plt.plot(x1, x2, 'bo', label='negative')
    plt.plot(xp, perception.obj_fun(xp, w, b), label='Boundary')
    plt.legend()
    plt.show()
