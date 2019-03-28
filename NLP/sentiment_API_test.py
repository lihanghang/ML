<<<<<<< HEAD
# import tensorflow as tf
#
# a = tf.constant(12)
# b = tf.constant(54)
# # limit work filed
# with tf.Session() as sess:
#     # sess = tf.Session()
#     print(sess.run(a+b))
# c = tf.placeholder(tf.int16)
# d = tf.placeholder(tf.int16)
# add = tf.add(c, d)
# mul = tf.multiply(c, d)
# with tf.Session() as sess:
#     print(sess.run(add, feed_dict={c: 2, d: 3}))
#     print(sess.run(mul, feed_dict={c: 5, d: 6}))
#
# # save graph of compute
# writer = tf.summary.FileWriter(logdir="logs", graph=tf.get_default_graph())
# writer.flush()
import json
import urllib.request


def sentiment_classify(text):
    """
    调用BaiDu NLP 接口实现情感分析
    获取文本的感情偏向（消极 or 积极 or 中立）
    参数：
    text:str 本文
    返回值
    log_id	uint64	请求唯一标识码
    sentiment	int	表示情感极性分类结果，0:负向，1:中性，2:正向
    confidence	float	表示分类的置信度，取值范围[0,1]
    positive_prob	float	表示属于积极类别的概率 ，取值范围[0,1]
    negative_prob	float	表示属于消极类别的概率，取值范围[0,1]
    """
    raw = {"text":"内容"}
    raw['text'] = text
    data = json.dumps(raw).encode('utf-8')
    AT = "24.d81216e18bf47c413848d50d1c2c6285.2592000.1544067794.282335-14690965"
    host = "https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?charset=UTF-8&access_token="+AT
    request = urllib.request.Request(url=host, data=data)
    request.add_header('Content-Type', 'application/json')
    response = urllib.request.urlopen(request)
    content = response.read().decode('utf-8')
    rdata = json.loads(content)
    return rdata


# 所要分析的文本信息
news = '''
公司作为央企地产龙头，融资成本优势和资源整合能力非常明显，并且周转有所提速，新增拿地盈利能力相对可控，激励体系亦有所完善，预计未来业绩仍有望延续较快速增长。公司目前市值较RNAV折价30%+，2018-2020年PE分别为7.3X/6.0X/5.0X，安全边际显著；维持“强烈推荐-A”的投资评级，目标价17.1元/股（对应10倍PE）。

业绩增长平稳。公司前三季度营收/归母净利分别为950/96亿，同比分别增长+26%/+16%。收入增速相对平稳，结算毛利率上半年较高而三季度单季有所回落，但整体看前三季度较去年同期仍提升0.5PCT至32.7%；不过，期内少数股东损益占比提升10PCT至29%，拉低归母净利增速至16%。全年看业绩较高增长预期不变。考虑到：1）全年看，公司计划竣工面积1900万平米，较17年实际竣工面积增长22%，而前三季度仅竣工1173万平米（完成率62%），同比+9.2%，即若全年竣工计划不变，四季度结算量或有更高增长；另外，16/17年公司销售均价持续回升（同比分别+4%/+5%），且均>1.3万元/平米，而17年结算均价仅0.99万元/平米，所以18年结算均价亦有望回升，全年看结算收入有望较高增长；2）18Q3末公司预收款3175亿（同比+38%），约为17年收入的2.2倍，也即18年全年业绩保障性较强。

EPS分别为1.71、2.08和2.48元，对应PE分别为7.3X、6.0X和5.0X，目前市值较RNAV折价30%+，安全边际显著；维持“强烈推荐-A”的投资评级，目标价17.1元/股（对应10倍PE）
'''


res = sentiment_classify(news)
print(res)
lable = res['items'][0]['sentiment']
if lable == 0:
    print("情感偏负向！" + "有"+str(res['items'][0]['negative_prob']*100)+"%的可能性")
elif lable == 1:
    print("情感偏中立！"+ "有"+str(res['items'][0]['confidence']*100)+"%的可能性")
else:
    print("情感偏正向"+ "有"+str(res['items'][0]['positive_prob']*100)+"%的可能性")
=======
# import tensorflow as tf
#
# a = tf.constant(12)
# b = tf.constant(54)
# # limit work filed
# with tf.Session() as sess:
#     # sess = tf.Session()
#     print(sess.run(a+b))
# c = tf.placeholder(tf.int16)
# d = tf.placeholder(tf.int16)
# add = tf.add(c, d)
# mul = tf.multiply(c, d)
# with tf.Session() as sess:
#     print(sess.run(add, feed_dict={c: 2, d: 3}))
#     print(sess.run(mul, feed_dict={c: 5, d: 6}))
#
# # save graph of compute
# writer = tf.summary.FileWriter(logdir="logs", graph=tf.get_default_graph())
# writer.flush()
import json
import urllib.request


def sentiment_classify(text):
    """
    调用BaiDu NLP 接口实现情感分析
    获取文本的感情偏向（消极 or 积极 or 中立）
    参数：
    text:str 本文
    返回值
    log_id	uint64	请求唯一标识码
    sentiment	int	表示情感极性分类结果，0:负向，1:中性，2:正向
    confidence	float	表示分类的置信度，取值范围[0,1]
    positive_prob	float	表示属于积极类别的概率 ，取值范围[0,1]
    negative_prob	float	表示属于消极类别的概率，取值范围[0,1]
    """
    raw = {"text":"内容"}
    raw['text'] = text
    data = json.dumps(raw).encode('utf-8')
    AT = "xxxxxxxx"
    host = "https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?charset=UTF-8&access_token="+AT
    request = urllib.request.Request(url=host, data=data)
    request.add_header('Content-Type', 'application/json')
    response = urllib.request.urlopen(request)
    content = response.read().decode('utf-8')
    rdata = json.loads(content)
    return rdata


news = '''
1月5日，一张海报在网上流传：抬头是小米和华润置地的logo，正文是“支持实体经济，助力产业发展”，
海报下面几行小字写的是“恭贺小米科技携手华润置地荣摘昌平区沙河镇七里渠南北村公建混合住宅项目地块'''
res = sentiment_classify(news)
lable = res['items'][0]['sentiment']
if lable == 0:
    print("情感偏负向！")
elif lable == 1:
    print("情感偏中立！")
else:
    print("情感偏正向")
>>>>>>> 9ff18c814f28b1f202e1c4a975cce2b12a564df6
