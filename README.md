## &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;技术专列
### 一、ML内容--工程测试代码
#### 各文件说明
- attention&&memory:工程代码。基于注意力机制和记忆网络实现企业信用评级和科研投入的模型构建与训练。  
- CNN:主要包括手写数字识别实验与基于卷积神经网络的企业信用评级算法模型构建与训练实验。输出在out文件夹中  
- GAN:对抗神经网络实验，生成手写数字。  
- gcForestTest: 多粒度级联森林源码，由南大周志华老师团队提出。  
- MemNN:记忆网络实验相关。  
- MINST_DATA:手写数字数据集。  
- ML_Demo:机器学习实验相关：PM2.5、反欺诈等。  
- NLP:自然语言处理相关：基于卷积的情感分析。  
- RBM:受限玻尔兹曼机相关的手写数字实验。  
- RiskPredict:工程代码。企业风险分析，主要方法是卷积神经网络。  
- RNN:工程代码。行业风险分析，主要方法有RNN、ARIMA。包括汽车行业、信息服务业、房地产业。  


***
#### 实例代码--[ML Demo](https://github.com/lihanghang/ML/)
1. 机器学习实验，关于CNN、RNN、GAN等神经网络算法的入门、中级、高级运用，运用在图像处理或医疗诊断等领域
2. 机器学习基础0108（regression、classification）
3. **ML中回归、分类问题实例：**
* **PM2.5各成分影响比重分析及预测未来几日PM2.5浓度值**
* **E-mail反究欺诈预测**
4. 记忆神经网络研学习(0506)
***
#### 二、入门级--- 各大neural network 算法在Mnist上的运用
##### (1)CNN
1. 利用卷积神经网络对手写体数字进行训练测试，使用2层神经网络的准确率为92.42%。

##### (2)RNN
1. 利用循环神经网络同样对手写体数字进行训练测试，准确率高达97.99%，从我的实验数据来分析明显高于卷积神经网络。
***
##### (3)GAN
1. 生成式对抗网络生成手写数字图片，即教会神经网络算法自己学习写数字。（nivdia k80 GPU）
2. 在进行训练算法时，由于在远程服务器运行，本地调试，对于图像化显示就出现了错误，最后找到一个可行的方案就是在本地安装Xming软件并在
服务器端设置环境变量其中的将图像显示指向本地的ip:0.0,前提是要保证服务器端ping通本地机器，如果无法连接本地机器可关闭本地防火墙后再次尝试
，就能很好解决这个问题，我使用的Xshell连接远程服务器。
3. 我用GAN教神经网路算法自己书写数字，在10000、20000、30000次的迭代下观察变化不是很大，而且也并不是迭代次数与输出结果有线性关系即次数越多并
不一定会增加训练结果的友好度，适当即可！

***
#### 三、NLP情感分析实践
##### (1)对企业舆情数据进行情感分析，目前能够进行正负向分析
1. 实践过程中，一是语料库不是很多，即便有数据量也不够大。那么没有数据，我们也不能随意创造数据。只有去寻找了。随即自己写个爬虫去爬
2. 基于TensorFlow的keras进行建模。主要使用jieba分词、word2vector、知乎已训练的词向量等。
3. 主要[数据和代码](https://github.com/lihanghang/ML/tree/master/NLP)

## &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;博客
***
- [研究生守则20条](http://blog.sciencenet.cn/home.php?mod=space&uid=220220&do=blog&id=444499)
- [偏爱什么样的学生？](http://blog.sciencenet.cn/home.php?mod=space&uid=265898&do=blog&id=241678)
- [王泛森院士：研究生和本科生的区别](http://www.folo.cn/user1/18593/archives/2009/79758.html)
- - -
## 日常基于Python的模型开发项目目录结构
├── conf  
   │ ├── conf.ini  // 相关路径配置 
├── logs  // 输出日志目录 
├── dataSets  // 数据集 
├── save_model  // 模型保存目录
├── config.py  // 模型参数配置 
├── model.py  // 模型文件 
├── train.py  // 训练文件 
├── test.py  // 测试文件
├── utils.py  // 数据预处理等组件 
├── README.md  // 项目说明

---
[个人文章地址](http://lihanghang.top)
* 备注：本仓库代码仅为工程项目实验调试时的整理，仅供参考！*
#### 最后一次更新于20190423  by HangHang Li 


