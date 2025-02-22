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
