import numpy as np
import pandas as pd
import copy
import json

langda = 0.5

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

def get_probality(class_df, length):
    # 初始化一个字典，用于记录每个特征值在当前类别中出现的次数
    temp_likelihood = {}
    # 初始化一个集合，用于存储所有出现在当前类别中的特征值
    words = set()
    
    # 对当前类别中的每个句子，获取其中所有特征值，并将这些特征值加入到集合words中
    for sentence in class_df['sentences']:
        word = sentence.split()
        words.update(word)
    
    # 对于每个特征值，统计其在当前类别中出现的次数
    for word in words:
        for sentence in class_df['sentences']:
            if word in sentence.split():
                if word not in temp_likelihood:
                    temp_likelihood[word] = 1
                else:
                    temp_likelihood[word] = temp_likelihood[word] + 1
    
    # 计算每个特征值在当前类别中的条件概率
    for word in words:
        temp_likelihood[word] = (temp_likelihood[word] + langda) / (len(class_df['sentences']) + langda * length)
    
    # 返回每个特征值在当前类别中的条件概率
    return temp_likelihood


def initialize(train_set):
    # 获取数据集中所有类别的列表
    class_list = train_set.label.unique()
    # 初始化一个字典，用于记录每个类别在训练集中的样本数量
    class_length = {}
    # 初始化一个字典，用于记录每个类别的先验概率
    prob = {}
    # 初始化一个字典，用于记录每个类别中每个特征值的条件概率
    likelihood = {}

    # 针对每个类别，获取该类别在训练集中的样本数量，计算其先验概率以及每个特征值的条件概率
    for class_ in class_list:
        # 按类别对训练集进行划分
        class_df = train_set[train_set['label'] == class_]
        # 记录该类别在训练集中的样本数量
        class_length[class_] = len(class_df['sentences'])
        # 计算该类别的先验概率
        values = train_set['label'].value_counts()
        prob[class_] = values.get(class_, 0) / len(train_set)
        # 计算该类别中每个特征值的条件概率
        temp_likelihood = get_probality(copy.deepcopy(class_df), len(train_set["sentences"]))
        likelihood[class_] = temp_likelihood

    # 返回所有类别的条件概率、类别列表、每个类别的先验概率、每个类别在训练集中的样本数量以及整个训练集中的句子总数
    return likelihood, class_list, prob, class_length, len(train_set["sentences"])


# 定义一个预测函数，函数名为“predict”，参数包括各种分类所需的数据和待分类数据
def predict(likeli_hood,class_list,prob,class_length,sample,length):
    # 初始化准确率为0
    accuracy = 0
    # 遍历待分类数据中的每个样本
    for index,row in sample.iterrows():
        # 将样本拆分成单词列表
        words = row['sentences'].split()
        # 初始化最大概率为0，最大概率所属类别为空
        max_prob = 0
        max_class = ""
        # 遍历所有类别
        for class_label in class_list:
            # 获取该类别的先验概率
            probality = prob[class_label]
            # 遍历样本中的每个单词
            for word in words:
                # 如果该单词不在该类别的似然概率字典中，将其似然概率设为λ / (类别样本数量 + λ * 单词表长度)
                if word not in likeli_hood[class_label]:
                    probality *= (langda) / (class_length[class_label] + langda * length)
                # 如果该单词在该类别的似然概率字典中，将其似然概率乘上该单词在该类别中的似然概率
                else:
                    probality *= likeli_hood[class_label][word]

            # 如果该类别的概率大于当前最大概率，更新最大概率和最大概率所属类别
            if probality > max_prob:
                max_prob = probality
                max_class = class_label
        # 如果预测结果与样本标签相同，准确率加1
        if max_class == sample.at[index, 'label']:
            accuracy = accuracy + 1
        # 将样本的预测结果更新到数据中
        sample.at[index,'label'] = max_class
    # 将预测结果写入CSV文件中
    sample.to_csv('result/21307040_liujiayi_NB-Bernoulli_classification.csv',index = False)
    # 输出准确率
    print(accuracy / len(sample["label"]))



def main():
    train_set,test_set = io_operation()
    likelihood,class_list,class_length,prob,tot_len = initialize(copy.deepcopy(train_set))
    predict(copy.deepcopy(likelihood),copy.deepcopy(class_list),
            copy.deepcopy(prob),copy.deepcopy(class_length),copy.deepcopy(test_set),tot_len)


if __name__ == "__main__":
    main()