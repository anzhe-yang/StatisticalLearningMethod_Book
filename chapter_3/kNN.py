import numpy as np
from sys import maxsize


class KNN:
    def __init__(self, _data, _p):
        self.data = _data
        self.p = _p

    def distance_p(self, _x1, _x2, _cur_p=None):
        if _cur_p is None:
            _cur_p = self.p
        l_p = np.sum(np.power(np.abs(_x1 - _x2), _cur_p))
        return np.power(l_p, 1/_cur_p)

    def find_neighbor(self, _landmark, _x):
        _min_d = maxsize
        _min_p = 0
        for _i in range(1, self.p+1):
            dis = self.distance_p(_landmark, _x, _cur_p=_i)
            if dis < _min_d:
                _min_d = dis
                _min_p = _i
            print('The distance between landmark x {} and xi {} is {:.2f}, and the selected p is {}'.format(_landmark, _x, dis, _i))
        print('The minimize distance in {} zone is {:.2f}'.format(_min_p, _min_d))


if __name__ == '__main__':
    x1 = np.array([[1, 1]])
    x2 = np.array([[5, 1]])
    x3 = np.array([[4, 4]])
    x = np.concatenate((x1, x2, x3), axis=0)

    p = 4
    kNN = KNN(x, p)
    kNN.find_neighbor(_landmark=x1, _x=x2)
    kNN.find_neighbor(_landmark=x1, _x=x3)
