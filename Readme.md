# 项目介绍
本项目是实现文本情感分析的预测，有分类和回归问题：

	• 在朴素贝叶斯分类、k-NN分类与k-NN回归中，三者至少完成两项；
	
	• 鼓励尝试多种算法及算法中的不同策略/参数，并进行结果对比分析；
	
	• 完成一份实验报告，注意实验报告要求。

# 目录结构描述
    ├── Readme.md           // 帮助文档
    
    ├── classification      // 分类数据集
    
    ├── regression     		// 回归数据集
    
    ├── Bag-Of-Words-KNN.py // Bag-Of-Words-KNN分类
    
    ├── TF-IDF-KNN.py  		//TF-IDF-KNN分类
    
    ├── One-hot-KNN.py      //One-hot-KNN分类
    
    ├── NaiveBayesBernoulli.py //朴素贝叶斯伯努利分类
    
    ├── NaiveBayesMultinomial.py //朴素贝叶斯多项式分类

# 使用说明

  通过打开相应文件，如果要进行验证，则把打开路径中的test_set.csv文件改成validation.csv文件，终端会输出正确率，如果要进行测试，则可直接测试，测试结果在上一级目录的result文件夹中。

