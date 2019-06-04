import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


class LeastSquare:
    def __init__(self):
        self._lambda = 0.0001

    def object_func(self, _x):
        f = np.sin(2 * np.pi * _x)
        return f

    def poly_func(self, _p, _x):
        # np.poly1d返回一个多项式函数类，其中参数代表多项式的最高次数
        # 之后调用本身的函数，参数为多项式中x的值
        f = np.poly1d(_p)
        func = f(_x)
        return func

    def loss(self, _p, _x, _y):
        return 1/2 * np.square(self.poly_func(_p, _x) - _y)

    def loss_with_l1_regularizer(self, _p, _x, _y):
        return np.append(self.loss(_p, _x, _y), self._lambda * abs(_p))

    def loss_with_l2_regularizer(self, _p, _x, _y):
        return np.append(self.loss(_p, _x, _y), self._lambda / 2 * np.sqrt(np.square(_p)))

    def fit(self, _m, _x, _y, _loss_func):
        """
        使用多项式去拟合目标函数，通过最小二乘法学习到拟合函数
        :param M: 多项式次数
        :return: 拟合函数
        """
        init = np.random.rand(_m+1)
        ls = leastsq(_loss_func, init, args=(_x, _y))
        ls_coeff = ls[0]
        print('Fitted polynomial coefficients: ', ls_coeff)
        return ls_coeff


if __name__ == '__main__':

    x = np.linspace(0, 1, 10)
    x_points = np.linspace(0, 1, 100)

    ls = LeastSquare()

    y_train_data = ls.object_func(x)
    y_noise_data = [np.random.normal(0, 0.1)+yi for yi in y_train_data]
    y_func = ls.object_func(x_points)

    M = 9

    y_noise_coeff = ls.fit(M, x, y_noise_data, ls.loss)
    y_noise_coeff_with_l1 = ls.fit(M, x, y_noise_data, ls.loss_with_l1_regularizer)
    y_noise_coeff_with_l2 = ls.fit(M, x, y_noise_data, ls.loss_with_l2_regularizer)

    # 多项式函数拟合噪声点数据
    y_noise_fit_func = ls.poly_func(y_noise_coeff, x_points)
    # 多项式函数使用l1正则化损失，拟合噪声点数据
    y_noise_fit_func_with_l1 = ls.poly_func(y_noise_coeff_with_l1, x_points)
    # 多项式函数使用l2正则化损失，拟合噪声点数据
    y_noise_fit_func_with_l2 = ls.poly_func(y_noise_coeff_with_l2, x_points)

    plt.plot(x_points, y_func, label='Ideal function')
    plt.plot(x, y_noise_data, 'ro', label='Training noise data')
    plt.plot(x_points, y_noise_fit_func, label='Fitted noise function')
    plt.plot(x_points, y_noise_fit_func_with_l1, label='Fitted noise function with l1 regularization')
    plt.plot(x_points, y_noise_fit_func_with_l2, label='Fitted noise function with l2 regularization')
    plt.legend() # 显示备注标签
    plt.show()
