#有from 把库的东西搬到本地 有* 全部导入
from numpy import *
#无from 记住库地址用时去取 无* 模块级导入
import operator

def createDataSet():
    # 6行1列的训练集
    group = array([[90,100],[88,90],[85,95],[10,20],[30,40],[50,30]])
    labels = ['A','A','A','D','D','D']
    # 输出训练集
    print("Points in class A:")
    print(group[:3,:])
    print("Points in class D:")
    print(group[3:,:])
    # 返回训练集
    return group,labels

def classify0(inX, dataSet,labels,k):
    # 欧式距离sqrt((x1-x2)**2 + (y1-y2)**2)
    
    # 用numpy的方法获取矩阵的行数（样本个数）shape[0]为行 shape[1]为列
    dataSetSize = dataSet.shape[0] 
    # 求差 x1-x2 / y1-y2 
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    # 平方 (x1-x2)**2
    sqDiffMat = diffMat ** 2
    # 求和 ((x1-x2)**2 + (y1-y2)**2)
    sqDistance = sqDiffMat.sum(axis = 1)
    # 开方 0.5*((x1-x2)**2 + (y1-y2)**2)
    distance = sqDistance ** 0.5
    # 排序并返回索引(最近邻)
    sortedDistance = distance.argsort()
    classCount = {}
    #遍历前k个最近邻的数据点
    for i in range(k):
        #暂时储存标签
        voteLabel = labels[sortedDistance[i]]
        #计数
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
        #降序排列，返回元组key= ??? reverse = ???
        sortedClassCount = sorted(classCount.items(),
                                  key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]
    
if __name__ == '__main__':
    group,labels = createDataSet()
    test = [85,90]
    test_class = classify0(test,group,labels,6)
    print("Point {} belongs to class {}".format(test,test_class))