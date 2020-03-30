## 笔记

#### CNN网络
- 参见10_CNN.pdf

#### 卷积
- 正向卷积输出
    ```math
    o=\lfloor\frac{i-k+2p}{s}+1\rfloor
    ```
    - o：输出的尺寸
    - i：输入的尺寸
    - k：kernel size
    - s：stride
    - p：padding的像素个数
        - valid: 0
        - same : 1
        - 也可以直接指定个数
- 可训练参数
    ```math
    (ks1 * ks2 + b) * fs
    ```
    - ks1, ks2: kernel size的大小
    - b : bias偏置，b=1
    - fs: 卷积核个数
- 卷积层连接数：   
    ```math
    (ks1 * ks2) * fs * (o1 * o2)
    ```
    - o1, o2: 每个特征图的输出大小
- 卷积层的输出神经元个数：
    ```math
    fs * (o1 * o2)
    ```
- pooling层可训练参数     
由于只涉及到降采样，直接除以kernel size
    ```math
    (1+b) * fs
    ```
    - c_{ks1}, c_{ks1}: 卷积层kernel size
    - p_{ks1}, p_{ks2}: 池化层kernel size
- 池化层连接数与输出神经元个数与卷积层相同


 
#### 反卷积
- 概念：       
反卷积是一种特殊的正向卷积，先按照一定的比例通过补零来扩大输入图像的尺寸，接着旋转卷积核，再进行正向卷积。

- 反向卷积的输出
    1. 对输入进行stride变换，stride可理解为为在输入的相邻元素之间添加s-1个零元素
        ```math
        i_s=i+(s-1)(i-1)
        ```
    2. 对stride变换后的图片按照卷积形式求解输出尺寸
        ```math
        o=\lfloor\frac{i+2p-k}{s}+1\rfloor
        ```
        - 由于在第一步中已经将stride进行运算，式中的s此时恒等于1，即s=1
    3. 最终合并得
        ```math
        o=s(i-1)+2p-k+2
        ```
        - 式子恒为整数，可将下取整运算符去掉
- 反卷积的问题    
    如果参数配置不当，很容易出现输出feature map带有明显棋盘状的现象