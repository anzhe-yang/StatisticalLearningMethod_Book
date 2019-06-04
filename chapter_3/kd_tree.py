import numpy as np
from collections import namedtuple
import math


def sort_k(data, start, end, k):
    if start < end:
        i = start
        j = end
        target = data[i][k]
        target_val = []
        for val in range(len(data[i])):
            target_val.append(data[i][val])
        while i < j:
            while i < j and data[j][k] > target:
                j -= 1
            if i < j:
                data[i] = data[j]
                i += 1
            while i < j and data[i][k] < target:
                i += 1
            if i < j:
                data[j] = data[i]
        for val in range(len(data[i])):
            data[i][val] = target_val[val]
        sort_k(data, start, i-1, k)
        sort_k(data, i+1, end, k)


class KdNode(object):
    def __init__(self, _val, _left, _right, _split):
        self.val = _val
        self.split = _split
        self.left = _left
        self.right = _right


class KdTree(object):
    def __init__(self, _data, _k):
        self.data = _data
        self.num = _data.shape[0]
        self.fea = _data.shape[1]
        self.root = self.create_node(_data, _k)

    def create_node(self, _cur_data, _k):
        """
        构造一个平衡 kd 树，先对数据按 _k 维排序，之后选取中位数作为分裂点
        每次选取的维度都会变化
        :param _cur_data: 每次分裂后的数据
        :param _k: 维度
        :return: 平衡 kd 树
        """
        if len(_cur_data) == 0:
            return None
        sort_k(_cur_data, 0, _cur_data.shape[0]-1, _k)
        split_index = _cur_data.shape[0] // 2
        split_point = _cur_data[split_index]
        next_k = (_k + 1) % self.fea
        return KdNode(split_point,
                      self.create_node(_cur_data[:split_index], next_k),
                      self.create_node(_cur_data[split_index+1:], next_k),
                      _k)

    def pre_order(self, root):
        print('Node is {}'.format(root.val))
        if root.left:
            self.pre_order(root.left)
        if root.right:
            self.pre_order(root.right)


result = namedtuple('res', 'nearest_point nearest_distance num_of_nodes_visited')


def travel(kd_node, target, cur_dis):
    """
    搜索过程
    1 从根节点出发，如果 target 的第 split_k 维坐标小于切分点 split_point 的坐标，则移动至其左子树，否则移动右子树，直到叶节点为止
    2 将当前叶节点作为"当前最近点" nearest
    3 递归向上回退，即访问父节点或者父节点的右节点
      a 如果该节点离目标更近，更新 nearest
      b nearest 一定存在于该节点一哥 子节点对应的区域 nearer_zone ，检查该子节点的父节点的右节点（另一子节点）对应的区域 further_zone是否有更近的点
        以目标点为中心，计算当前最近点切分点与目标点的欧式距离 temp_dis ，以这个距离为半径构成一个圆形区域，查看子节点区域 zone 是否与这个圆形区域相交
        如果相交，可能在这个子节点区域内存在更近的点，移动到此节点进行搜索
        如果不相交，则回退
    4 当回退到根节点时，搜索结束，当前最近点 nearest 即为目标的最近邻点
    计算复杂度为 O(logN)
    kd 树适用于训练数据远大于特征数量的情况
    :param kd_node: 当前搜索节点
    :param target: 目标点
    :param cur_dis: 当前搜索的最大距离，即上述条件中构成的圆形区域
    :return:
    """
    if kd_node is None:
        return result([0]*len(target), float('inf'), 0)

    num_of_nodes_visited = 1
    split_k = kd_node.split
    split_point = kd_node.val
    if target[split_k] <= split_point[split_k]:
        nearer_node = kd_node.left
        further_node = kd_node.right
    else:
        nearer_node = kd_node.right
        further_node = kd_node.left
    nearer_zone = travel(nearer_node, target, cur_dis)

    nearest = nearer_zone.nearest_point
    dis = nearer_zone.nearest_distance
    num_of_nodes_visited += nearer_zone.num_of_nodes_visited

    if dis < cur_dis:
        cur_dis = dis

    temp_dis = abs(split_point[split_k] - target[split_k])
    if cur_dis < temp_dis:
        return result(nearest, dis, num_of_nodes_visited)

    temp_dis = math.sqrt(sum((p1-p2)**2 for p1,p2 in zip(split_point, target)))
    if temp_dis < dis:
        nearest = split_point
        dis = temp_dis
        cur_dis = dis

    further_zone = travel(further_node, target, cur_dis)
    num_of_nodes_visited += further_zone.num_of_nodes_visited
    if further_zone.nearest_distance < dis:
        nearest = further_zone.nearest_point
        dis = further_zone.nearest_distance

    return result(nearest, dis, num_of_nodes_visited)


def neighbor_search(kd_tree, node):
    return travel(kd_tree.root, node, float('inf'))


if __name__ == '__main__':
    data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    k = 0
    kd_tree = KdTree(data, k)
    # kd_tree.pre_order(kd_tree.root)
    search_point = [4, 1]
    print(neighbor_search(kd_tree, search_point))
