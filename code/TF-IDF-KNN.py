import numpy as np
import pandas as pd
import copy
import math
from sklearn.neighbors import KDTree
import json

def io_operation():
    # 读取训练集和测试集
    train_set = pd.read_csv("code/classification/classification_dataset/train_set.csv",
                            encoding='utf-8',engine = "python")
    # 替换指定列的列名
    train_set = train_set.rename(columns={'Words (split by space)': 'sentences'})
    test_set = pd.read_csv("code/classification/classification_dataset/test_set.csv",
                           encoding='utf-8',engine = "python")
    # 替换指定列的列名
    test_set = test_set.rename(columns={'Words (split by space)': 'sentences'})

    return train_set,test_set

def initialize(train_set,test_set):
    class_list = {}  # 保存每个样本所属类别的字典
    IDF = {}  # 保存每个词的逆文档频率的字典
    words = set()  # 保存所有训练集和测试集中出现过的词的集合
    # 遍历训练集，将出现的词添加到words集合中
    for sentence in train_set['sentences']:
        word = sentence.split()
        words.update(word)
    # 遍历测试集，将出现的词添加到words集合中
    for sentence in test_set['sentences']:
        word = sentence.split()
        words.update(word)
    words = list(words)  # 将集合转换为列表
    sentence_length = len(train_set['sentences'])  # 训练集的句子数
    words_length = len(words)  # 词的个数
    train_matrix = np.zeros((sentence_length,words_length))  # 初始化训练集矩阵
    # 计算每个词的逆文档频率
    for i in range(words_length):
        for sentence in train_set['sentences']:
            sentence_word = sentence.split()
            if words[i] in sentence_word:
                if i not in IDF:
                    IDF[i] = 1
                else:
                    IDF[i] = IDF[i] + 1
    for i in range(words_length):
        if i not in IDF.keys():
            IDF[i] = 0
    for index in IDF.keys():
        IDF[index] = math.log(sentence_length / (IDF[index] + 1),10)
    # 构建训练集矩阵
    for index,row in train_set.iterrows():
        sentence_words = row['sentences'].split()
        class_list[index] = row['label']
        for sentence_word in sentence_words:
            train_matrix[index][words.index(sentence_word)] = train_matrix[index][words.index(sentence_word)] + 1
        sum = np.sum(np.abs(train_matrix[index]))
        for j in range(len(train_matrix[index])):
            train_matrix[index][j] = train_matrix[index][j]  * IDF[j] / sum
    return train_matrix,words,class_list,IDF


def predict(train_matrix,train_set,sample,total_words,class_list,IDF):
    """
    预测函数，根据训练集和测试集的TF-IDF矩阵，计算测试集中每个样本与训练集样本的距离，
    找出距离最近的k个样本，统计它们所属的类别，选择出现次数最多的类别作为预测结果。
    参数：
    train_matrix -- 训练集的TF-IDF矩阵，维度为(len(train_set), len(words))
    train_set -- 训练集，Pandas DataFrame对象，包含'sentences'和'label'两列
    sample -- 测试集，Pandas DataFrame对象，包含'sentences'和'label'两列
    total_words -- 词汇表，Python list对象，包含所有在训练集和测试集中出现的单词
    class_list -- 训练集中每个样本所属的类别，Python dict对象，键为样本的下标，值为样本的类别
    IDF -- 每个单词的逆文档频率，Python dict对象，键为单词在词汇表中的下标，值为其逆文档频率

    返回值：
    无返回值，输出预测准确率
    """
    accuracy = 0
    choose_k = int(math.sqrt(len(train_set['sentences'])))  # 选择距离最近的k个样本
    for index,row in sample.iterrows():
        ans_list = []  # 存储每个训练集样本与测试集样本的距离和所属类别的元组
        words = row['sentences'].split()
        test_matrix =  np.zeros(len(total_words))
        for word in words:
            test_matrix[total_words.index(word)] = test_matrix[total_words.index(word)] + 1
        sum = np.sum(np.abs(train_matrix[index]))
        for j in range(len(test_matrix)):
            test_matrix[j] = test_matrix[j] * IDF[j] / sum
        distances = np.linalg.norm(train_matrix - test_matrix, axis=1)  # 计算欧氏距离
        for i in range(len(distances)):
            ans_list.append((distances[i],class_list[i]))
        sorted_list = sorted(ans_list, key=lambda x: x[0])  # 按元组中第一个元素排序
        ans_count = {}  # 存储前k个距离最近的样本所属的类别及其出现次数
        for i in range(choose_k):
            if sorted_list[i][1] not in ans_count:
                ans_count[sorted_list[i][1]] = 1
            else:
                ans_count[sorted_list[i][1]] = ans_count[sorted_list[i][1]] + 1
        max_class = max(ans_count, key=ans_count.get)  # 选择出现次数最多的类别作为预测结果
        if max_class == sample.at[index, 'label']:
            accuracy = accuracy + 1
        sample.at[index,'label'] = max_class
    sample.to_csv('result/21307040_liujiayi_TF-IDF-KNN_classification.csv',index = False)
    print(accuracy / len(sample["label"]))

def KDtree_predict(train_matrix,train_set,sample,total_words,class_list):
    accuracy = 0
    choose_k = int(math.sqrt(len(train_set['sentences']))) - 2
    
    # 构建KD树
    kdt = KDTree(train_matrix, leaf_size=30, metric='euclidean')
    
    for index,row in sample.iterrows():
        ans_list = []
        words = row['sentences'].split()
        test_matrix =  np.zeros(len(total_words))
        for word in words:
            test_matrix[total_words.index(word)] = 1
        # 使用KD树进行快速搜索
        distances, indices = kdt.query([test_matrix], k=choose_k)
        for i in range(len(distances[0])):
            ans_list.append((distances[0][i],class_list[indices[0][i]]))
        sorted_list = sorted(ans_list, key=lambda x: x[0])  # 按元组中第一个元素排序
        ans_count = {}
        for i in range(choose_k):
            if sorted_list[i][1] not in ans_count:
                ans_count[sorted_list[i][1]] = 1
            else:
                ans_count[sorted_list[i][1]] = ans_count[sorted_list[i][1]] + 1
        max_class = max(ans_count, key=ans_count.get)
        if max_class == sample.at[index, 'label']:
            accuracy = accuracy + 1
        sample.at[index,'label'] = max_class
    sample.to_csv('result/21307040_liujiayi_One-hot-KNN_classification.csv',index = False)
    print(accuracy / len(sample["label"]))


def main():
    train_set,test_set = io_operation()
    train_matrix,words,class_list,IDF = initialize(copy.deepcopy(train_set),copy.deepcopy(test_set))
    predict(copy.deepcopy(train_matrix),copy.deepcopy(train_set),copy.deepcopy(test_set),
            copy.deepcopy(words),copy.deepcopy(class_list),copy.deepcopy(IDF))


if __name__ == "__main__":
    main()