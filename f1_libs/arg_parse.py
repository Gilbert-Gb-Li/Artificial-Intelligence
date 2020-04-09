"""
参数解析
https://zhuanlan.zhihu.com/p/56922793
"""
import argparse


def main():
    """
    # 使用步骤：
        1、创建ArgumentParser()对象。
        2、调用add_argument()方法添加参数。
        3、调用parse_args()解析添加的参数。
    # 参数：
        default：没有设置值情况下的默认参数
        required：表示这个参数是否一定需要设置
        type：参数类型
        choices：参数值只能从几个选项里面选择
        help：指定参数的说明信息
        nargs： 设置参数在使用可以提供的个数
             N   参数的绝对个数（例如：3）
            '?'   0或1个参数
            '*'   0或所有参数
            '+'   所有，并且至少一个参数
    """
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('-n', '--name', default=' Li ')
    parser.add_argument('-y', '--year', default='20')
    args = parser.parse_args()
    print(args)
    name = args.name
    year = args.year
    print('Hello {}  {}'.format(name, year))


if __name__ == '__main__':
    main()
