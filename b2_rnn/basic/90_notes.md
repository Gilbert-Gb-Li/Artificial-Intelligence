## 笔记

### 数据
- 表格数据 [batch_size, feature]
    - 使用浅层或多层全连接网络
    - 关注的是特征间的依赖关系

- 二维连续数据 (Batch_size, High, weight, channel)
    - 二维，high和weight不可变
    - 图像
    - 雷达数据
    - 卷积网络，需要深度

- 一维连续数据 (batch_size, time, channel)
    - 信号或波形数据： 语音 conv1d
    - 自然语言类型：文本 conv2d
    - 循环神经网络、卷积神经网络，需要广度


### 1.Embedding

1. 使用0,1,2...n连续数值为每个字设定ID编号
2. 对ID编码进行one hot
3. 降维 x = vW
    - W: [sequence_length, embedding_size] => [字数量，降维后长度] 
    - 常用的embedding_size有64, 128, 256, 512

- 参考代码【Basic/01_embedding】

### 2.基础循环神经网络
- 数学公式
    ```math
    y_t = f(X_tW_1 + y_{t-1}W_2) \tag{1}
    ```
    ```math
    y_t = f(hstack[X_t, y_{t-1}] W) \tag{2}
    ```
    ```math
    y_t = f([X_t, X_{t-1}, X_{t-2}] W) \tag{3}
    ```
    - (1)式与(2)式等价，完全可以通过调整W1与W2的维度转换成(2)式
    - `$yt$`的维度与`$y_{t-1}$`一致，所以(1)式中W2为方阵
    - (3)式可理解为卷积神经网络
- 参考代码【Basic/02_rnn_basic】
- 循环神经网络输入    
    [Batch_size,sequence_len,embadding_size]，每次输入为一个序列的所有样本数据
- 循环网络的意义    
    特征间含有潜在的相关性，使每个向量携带前（后）向量的信息，向量间建立关联关系
- 预测    
    与训练类似，样本中term[i]逐个与相应权重计算，迭代得出最终值

### 3.LSTM神经网络
- 添加记忆向量，来计算某个特征的重要程度，相当于加权
- 参考文档【reference/深度神经网络1】
- 参考代码【Basic/03_rnn_opt】
- 说明：        
    1. cell 的权重是共享的，这是什么意思呢？   
    这是指这张图片上有三个绿色的大框，代表三个cell 对吧，但是实际上，它只是代表了一个 cell在不同时序时候的状态，所有的数据只会通过一个cell，然后不断更新它的权重。
    2. 那么一层的 LSTM 的参数有多少个？     
    根据以上说明，我们知道参数的数量是由cell 的数量决定的，这里只有一个cell，所以参数的数量就是这个cell里面用到的参数个数。假设 num_units【h_t-1】是128，输入是28位的，可以得到，四个小黄框的参数一共有（128+28）×（128×4），也就是156×512，可以看看 TensorFlow的最简单的LSTM的案例，中间层的参数就是这样，不过还要加上输出的时候的激活函数的参数，假设是10个类的话，就是128*10的W参数和10个bias参数

    
### 4.双向RNN
- 意义:    
   添加后续向量对该向量的影响 
- 方法:
    - 以输出`$X_t,X_{t-1}... $`为输入创建反向神经网络`$z_t$`
    - `$z_t$`包含了`$X_t$`之后的信息,`$y_t$`包含了`$X_t$`之前的信息,将`$y_t$`与`$z_t$`连接, 则变成了双向RNN网络
- 参考代码【text/10_text_segment_rnn】
- 优劣
    - 可以考虑后续向量的影响
    - 缺少实时性, 需要等到所有词输入后才可预测
    - 训练速度慢

### 5.Loss函数
引入全连接层进行分类或者解决回归问题，所以损失函数为之前常用的交叉熵或均方误差。

#### 5.1分类问题
- 此类问题，最后一个序列的向量可代表全文信息【表示学习】，所以可使用最后一个序列的向量来做分类，因此只计算最后一个向量的损失函数即可


#### 5.2文本分词及生成问题
- 需要预测每个词的输出类别，所以需要计算每个词的损失函数，即将所有词的损失函数相加取平均


### 6.循环神经网络示例

#### 文本分类模型
1. 文本向量化  
    - 文本转ID，为每个词编号
        - 例，"今天天气不错" -> [96, 22, 22 ...]
    - ID进行one-hot转换
        - 例，[96, 22, 22 ...] -> shape [1, 6, 5388] : 
    - Embedding, one-hot 降维
2. 建立模型
    - 定义固定的输入长度，可选
    - 建立RNN
        - 选取最后一个时间步为输出
    - 建立FC层完成分类
        - 若只有一层输出，可以直接使用RNN替代
3. 损失函数
    - 交叉熵
4. 预测
    - 向量文本化
    - 调用RNN及FC模型输出概率
    - 转换为最大标签
    
#### 文本分词模型 
1. 输入
    - 数据：今天天气预报有雨。
    - 标签: B E B M M E B E S
2. 模型建立
    - RNN
        - 单向RNN，无法综合前后文信息判断
        - 使用双向RNN，将数据正向输入后，反向输入回网络
    - FC模型
3. 损失函数
    - 使用Sequence Loss
4. 预测
5. 思考
    - 文本分词并不需要太长的前后文，可以使用卷积神经网络

#### 文本生成模型
- 输入：万人凿磐石，无由达江浒。
- 标签: 人凿磐石，无由达江浒。\n
    - 标签与输入差一个时间步
##### 1 训练
1. Embedding
    - input : [BatchSize, Time] 整型
    - output: [BatchSize, Time, Feature1] 浮点型
2. RNN：循环神经网络
    - input : [BatchSize, Time, Feature1]
    - output: [BatchSize, Time, Feature2]
        - 此时向量特征携带了前一向量的信息
3. Dense：全链接
    - input : [BatchSize, Time, Feature2]
    - output: [BatchSize, Time, n_class]
        - 将属性转换为指定的类别
4. Loss | SequenceLoss
    - 最后使用全连接层，使用交叉熵为损失函数 

##### 2 预测，推断
1. input: 第一个字符
2. Embedding转换为向量V
3. 将v, s输入到RNN中获取 (y，s)    
    y为概率
4. 依概率选择
    - 贪心算法,
        - 选择当前概率最高的.
        - 每次输出都相同，无随机性
        - 非全局最优解
    - 集束搜索 (beam search)
        - 选择概率最大的n个序列
            - 选择time_1概率最大的前个词
            - 计算time_2
        - 可接近全局最优，没有随机性
    - 随机选择
        - 非全局最优，有随机性
5. 将下一时刻字符循环输入回神经网络
6. 直到终止字符或迭代足够步数停止


## 课程内容
- part 1 Embed
- part 2 RNN (单层、多层)
- part 3 LSTM
- part 4 Loss； 文本分类
- part 5 文本生成
- part 6 文本生成
- part 7 文本生成双向RNN
- part 8 文本生成双向RNN

## RNN基础使用
- 主要代码：
    - 01, embedding
    - 02, rnn的基本写法
    - 03, rnn tf 高阶写法
    - 20, loss 函数 

## 问题
- tf.Saver(var_list) ？
- tf.dynamic_rnn(marks) ?
- batch_size的使用?
- tf.contrib.seq2seq.sequence_loss(weights, mask)？

- 循环神经网络 ctc_loss？


- 文本分类过程及代码？
    - word_seg, part 5，6
    - word_seg_bi, part 7
- 分词
    - 交叉熵？
    - 不需要计算softmax?

- 卷积实现文本分词代码？

- tf.contrib.seq2seq.sequence_loss