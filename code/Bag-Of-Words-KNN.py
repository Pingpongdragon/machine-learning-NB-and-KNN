import numpy as np
import pandas as pd
import copy
import math
import json

def io_operation():
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
    class_list = {}
    words = set()
    # 获取训练集和测试集中所有的单词
    for sentence in train_set['sentences']:
        word = sentence.split()
        words.update(word)
    for sentence in test_set['sentences']:
        word = sentence.split()
        words.update(word)
    # 将单词转换成列表
    words = list(words)
    sentence_length = len(train_set['sentences'])
    words_length = len(words)
    train_matrix = np.zeros((sentence_length,words_length))
    for index,row in train_set.iterrows():
        sentence_words = row['sentences'].split()
        class_list[index] = row['label']
        # 构建训练矩阵
        for sentence_word in sentence_words:
            train_matrix[index][words.index(sentence_word)] = train_matrix[index][words.index(sentence_word)] + 1
    return train_matrix,words,class_list


def predict(train_matrix,train_set,sample,total_words,class_list):
    accuracy = 0
    # 选取K值，这里使用k=sqrt(n)-2
    choose_k = int(math.sqrt(len(train_set['sentences']))) - 2
    for index,row in sample.iterrows():
        ans_list = []
        words = row['sentences'].split()
         # 构建测试矩阵
        test_matrix =  np.zeros(len(total_words))
        for word in words:
            test_matrix[total_words.index(word)] = test_matrix[total_words.index(word)] + 1
        distances = np.linalg.norm(train_matrix - test_matrix, axis=1)
        for i in range(len(distances)):
            ans_list.append((distances[i],class_list[i]))
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
         # 将预测结果保存到样本数据中
        sample.at[index,'label'] = max_class
    # 将预测结果保存到文件中
    sample.to_csv('result/21307040_liujiayi_Bag-Of-Words-KNN_classification.csv',index = False)
    print(accuracy / len(sample["label"]))


def main():
    train_set,test_set = io_operation()
    train_matrix,words,class_list = initialize(copy.deepcopy(train_set),copy.deepcopy(test_set))
    predict(copy.deepcopy(train_matrix),copy.deepcopy(train_set),copy.deepcopy(test_set),copy.deepcopy(words),copy.deepcopy(class_list))


if __name__ == "__main__":
    main()