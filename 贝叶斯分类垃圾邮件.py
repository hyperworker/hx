import numpy
import re
import jieba 
import random
from numpy import array

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet | set(document)#创建并集
    return list(vocabSet)
    
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)#spam类别
    p0Num=numpy.ones(numWords);p1Num=numpy.ones(numWords)
    p0Denom=2.0;p1Denom=2.0 #初始化概率，拉普拉斯平滑，避免出现0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]#类别1的各单词数
            p1Denom+=sum(trainMatrix[i])#类别1的所有单词数
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=numpy.log(p1Num/p1Denom) #取对数
    p0Vect=numpy.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive
    
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+numpy.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+numpy.log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
        
def textParse(document):
    with open(document, 'rb') as fp:
        text = fp.read()
        wordCut = jieba.cut(text, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
        wordList = list(wordCut)  # genertor转化为list，每个词unicode格式
        return wordList
def spamText():
    docList=[];classList=[];
    for i in range(1,6):
        wordList=textParse('C:/Users/14006/Desktop/邮件/spam/{}.txt'.format(i))
        docList.append(wordList)  # 训练集list #list里套list,每个list是每篇文章分完词的Unicode编码
        classList.append(1) # spam类别
        wordList=textParse('C:/Users/14006/Desktop/邮件/pam/{}.txt'.format(i))
        docList.append(wordList)
        classList.append(0) # pam类别
    vocabList=createVocabList(docList)#返回所有文档中不重复词集
    trainingSet=list(range(10));testSet=[]
    for i in range(5):
        #随机构建训练集,留存交叉验证
        randomIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randomIndex])
        del(trainingSet[randomIndex])
    trainMat=[];trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v,p1v,pSpam=trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    for dicIndex in testSet:
        #对测试集分类
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0v,p1v,pSpam)!=classList[docIndex]:
            errorCount+=1
    print("the error rate is {}".format(float(errorCount)/len(testSet)))
    return float(errorCount)/len(testSet)
    
errorPercent=0.0
for i in range(10):
    errorPercent+=spamText()
print("the average error persent is : {}%".format(errorPercent/10*100))
