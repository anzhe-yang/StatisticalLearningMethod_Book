import numpy as np
import matplotlib.pyplot as plt


class Perception:
    def __init__(self, w, b, lr, num_epoch, train_data):
        self.w = w
        self.b = b
        self.lr = lr
        self.num_epoch = num_epoch
        self.data = train_data
        self.x_train = train_data[:, 0:2]
        self.y_train = train_data[:, 2]

    def sign(self, hx):
        if hx > 0:
            return 1
        else:
            return -1

    def loss(self, j):
        x = self.x_train[j]
        y = self.y_train[j]
        return y * self.sign(np.dot(self.w, x.T) + self.b)

    def sum_loss(self):
        y_pre_val = (np.dot(self.w.T, self.x_train.T) + self.b).T
        y_pre_act = np.where(y_pre_val > 0, 1, -1)
        return np.sum(self.y_train - y_pre_act)

    def update_para(self, j):
        x = self.x_train[j]
        y = self.y_train[j]
        self.w += self.lr * y * x
        self.b += self.lr * y

    def obj_fun(self, x_points):
        return (x_points * self.w[1] + b) / (-w[0] + 1e-4)

    def fit(self):
        '''
        每次按顺序从训练数据中选取某个数据，如果未被正确分类，则更新参数
        每次迭代完全部数据后，计算总体损失值，如果为零则结束训练
        否则训练次数为迭代次数
        :return: 训练后模型的参数 w 和 b
        '''
        for i in range(self.num_epoch):
            for j in range(self.data.shape[0]):
                if perception.loss(j) <= 0:
                    perception.update_para(j)
                    print('Selected error point: xi= {}'.format(self.x_train[j]))
            if perception.sum_loss() == 0:
                break


if __name__ == '__main__':

    x1 = np.array([[3, 3]])
    x2 = np.array([[4, 3]])
    x3 = np.array([[1, 1]])
    x = np.concatenate((x1, x2, x3), axis=0)
    y = np.array([[1], [1], [-1]])
    data = np.append(x, y, axis=1)
    print('data: \n{} \nlabel: \n{}'.format(x, y))

    w = np.zeros(2)
    b = np.zeros(1)
    lr = 1
    perception = Perception(w, b, lr, num_epoch=100, train_data=data)
    perception.fit()

    print('Optimized parameters: \nw: {} \nb: {}'.format(w, b))

    xp = np.linspace(0, max(x[:, 0]), 100)
    for i in data:
        x1, x2, yi = i
        if yi == 1:
            plt.plot(x1, x2, 'ro', label='positive')
        else:
            plt.plot(x1, x2, 'bo', label='negative')
    plt.plot(xp, perception.obj_fun(xp), label='Boundary')
    plt.legend()
    plt.show()
