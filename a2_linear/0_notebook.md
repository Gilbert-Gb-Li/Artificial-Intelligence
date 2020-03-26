# 笔记与问题

## 笔记

#### 1.线性回归
- 讲义
- 代码
    - 梯度下降实现    [9_regression_implement/SDG/0_SGD_refined]
- 细节
    - `$\theta$`的取值，可以随机初始化为分布
    - 当使用线性回归选特征时，比较多个系数的绝对值
- 公式
    ```math
    \begin{aligned}
    model&:&h_\theta(x)&=&\theta^TX \\
    loss&:&J(\theta)&=&\frac{1}{2m}\sum_{i=1}^m(h_\theta(x_i)-y_i)^2\\
    GD&:&\theta_j&=&\theta_j-\alpha\frac 1m\sum_{i=1}^m(h_\theta(x_i)-y_i)x_i
    \end{aligned}
    ```

#### 2.逻辑回归
- sklearn 参数
    - pently 正则化
    - c 
    - solver 梯度，优化算法
    - mutil 若选择mutil_module为softmax
- 公式
    ```math
    \begin{aligned}
    model&:&h_\theta(x)&=&\frac{1}{1+e^{\theta^TX}} \\
    loss&:&J(\theta)&=&-[y_ilog(h_\theta(x_i))+(1-y_i)log(1-h_\theta(x_i))]\\
    GD&:&\theta_j&=&\theta_j-\alpha\frac 1m\sum_{i=1}^m(h_\theta(x_i)-y_i)x_i
    \end{aligned}
    ```
- 说明      
当`$y=1$`时损失函数`$-y_ilog(h_\theta(x_i))$`，如果想要损失函数尽可能的小，那么`$h_\theta(x_i)$`就要尽可能的大，因为sigmoid函数取值[0,1]，所以`$h_\theta(x_i)$`会无限接近于1。      
当`$y=0$`时损失函数`$-y_ilog(1-h_\theta(x_i))$`，如果想要损失函数尽可能的小，那么`$h_\theta(x_i)$`就要尽可能的小，因为sigmoid函数取值[0,1]，所以`$h_\theta(x_i)$`会无限接近于0。
#### 3.softmax
- 公式
    ```math
    \begin{aligned}
    model&:&h_\theta(x)&=&\frac{e^{\theta^T_jx_i}}{\sum_c^k\theta^T_cx_i} \\
    loss&:&J(\theta)&=&-y_ilog(h_\theta(x_i))\\
    GD&:& &=&(h_\theta(x_i)-1)x_i
    \end{aligned}
    ```
#### 4.正则化

## 课程内容
- part 1 
    - 线性回归
    - 损失函数
    - 梯度下降
- part 2
    - 量纲统一
    - 线性回归细节讲解
- part 3
    - 随机梯度下降python实现
- part 4
    - 作业讲解
    - 逻辑回归
- part 5
    - 牛顿法
    - 点击率
    - softmax