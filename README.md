# Kaggle-Titanic
针对Kaggle-Titanic （prediction结果0.76555）
## 学习产物，大部分思想来自hanxiaoyang 
https://github.com/HanXiaoyang/Kaggle_Titanic         
http://blog.csdn.net/han_xiaoyang/article/details/49797143

### 预测方法：LogisticRegression
使用sklearn.linear_model.LogisticRegression

### 处理三个缺失值Age, Cabin, Embarked
Age两种处理方法(在这里使用了2):
1. randomForest预测缺失的Age
2. 根据不同的称呼（Miss，Mrs，Mr），分成三类，并分别填补平均值

Cabin:
处理的较多
1. 根据Cabin有无构造特征Cabinyn（1/0）
2. 根据Cabin前的字母构造特征Cabinletter（无=0，A/B/C/D/E/F/G=1/2/3/4/5/6,有一个特别的值T=0）
3. 根据Cabin的数值构造特征Cabinnum

Embarked：
缺失两个值，因此直接丢掉了

### 构造了一些新特征
* SP
Parch+SibSp 判断乘客是不是一个人来的
* Chi
以12岁为分界线，判断是不是小孩
* Old
以50岁为分界线，判断是不是老人
* Mother
以姓名中的Mrs以及Parch>=1分标准，判断是不是母亲
* PS
Pclass+Sex 由于Pclass和Sex的影响过大，将两个特征结合起来构造一个新特征

11/5/2017

