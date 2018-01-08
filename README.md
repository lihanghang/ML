## &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;技术专列
#### 一、ML内容
***
#### 实例代码在 ML Demo中
1. 机器学习实验，关于CNN、RNN、GAN等神经网络算法的入门、中级、高级运用，运用在图像处理或医疗诊断等领域
2. 机器学习基础0108（regression、classification）
3. ML中回归、分类问题实例：
**PM2.5各成分影响比重分析及预测未来几日PM2.5浓度值**
**E-mail反欺诈预测**
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
#### 三、应用级--- 各大Neural Network算法在LUNG Cancer Image中利用，来进行临床辅助诊断
##### (1)图像预处理
1.
2.
3. ……

***

## &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;有启示意义的博客
***
- [研究生守则20条](http://blog.sciencenet.cn/home.php?mod=space&uid=220220&do=blog&id=444499)
- [偏爱什么样的学生？](http://blog.sciencenet.cn/home.php?mod=space&uid=265898&do=blog&id=241678)
- [王泛森院士：研究生和本科生的区别](http://www.folo.cn/user1/18593/archives/2009/79758.html)
- - -
#### 这里不仅有技术还有诗和远方！
[我的个人文章地址](http://www.lenhard.cf)
#### 更新于20170711