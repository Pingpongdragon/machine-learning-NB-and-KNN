import numpy as np
import pandas as pd
import copy
import json

langda = 50

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

def get_probality(class_df,length):
    temp_likelihood = {}
    words = class_df['sentences'].str.cat(sep=' ').split()
    for word in words:
        for sentence in class_df['sentences']:
            if word in sentence.split():
                if word not in temp_likelihood:
                    temp_likelihood[word] = sentence.count(word)
                else:
                    temp_likelihood[word] = temp_likelihood[word] + sentence.count(word)
    for word in words:
        temp_likelihood[word] = (temp_likelihood[word] + langda) / (len(words) + langda * length)
    return temp_likelihood

def initialize(train_set):
    class_list = train_set.label.unique()
    class_length = {}
    prob = {}
    likelihood = {}
    for class_ in class_list:
        class_df = train_set[train_set['label'] == class_]  # 按类划分数据集
        class_length[class_] = len(class_df['sentences'].str.cat(sep=' ').split())
        values = train_set['label'].value_counts()
        prob[class_] = values.get(class_, 0) / len(train_set)  # 初始化为类的先验概率
        temp_likelihood = {}
        temp_likelihood = get_probality(copy.deepcopy(class_df),len(train_set["sentences"].str.split()))
        likelihood[class_] = temp_likelihood
    return likelihood,class_list,prob,class_length,len(train_set["sentences"].str.cat(sep=' ').split())

def predict(likeli_hood,class_list,prob,class_length,sample,length):
    accuracy = 0
    for index,row in sample.iterrows():
        words = row['sentences'].split()
        max_prob = 0
        max_class = ""
        for class_label in class_list:
            probality = prob[class_label]
            for word in words:
                if word not in likeli_hood[class_label]:
                    probality *= (langda) / (class_length[class_label] + langda * length)
                else:
                    probality *= likeli_hood[class_label][word]

            if probality > max_prob:
                max_prob = probality
                max_class = class_label
        if max_class == sample.at[index, 'label']:
            accuracy = accuracy + 1
        sample.at[index,'label'] = max_class
    sample.to_csv('result/21307040_liujiayi_NB-Multinomial_classification.csv',index = False)
    print(accuracy / len(sample["label"]))


def main():
    train_set,test_set = io_operation()
    likelihood,class_list,class_length,prob,tot_len = initialize(train_set)
    predict(copy.deepcopy(likelihood),copy.deepcopy(class_list),
            copy.deepcopy(prob),copy.deepcopy(class_length),copy.deepcopy(test_set),tot_len)


if __name__ == "__main__":
    main()