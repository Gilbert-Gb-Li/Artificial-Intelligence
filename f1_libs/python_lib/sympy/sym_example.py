import sympy

# 首先定义x为一个符号，表示一个变量
# 打印时会使用Symbol来替换定义的变量
# 等同于sympy.S('x')
# 常量, sympy.S("2)
x = sympy.Symbol('x')

# type(fx): sympy.core.add.Add
fx = 2 * x + 1

# 用evalf函数，传入变量的值对表达式进行求值
fx.evalf(subs={x: 3})

# 解方程：x^2 - 1 = 0
# x1 = [-1, 1]
fx1 = x ** 2 - 1
x1 = sympy.solve(fx1, x)

# 多元函数求偏导
# d = 2*x + 2
y = sympy.Symbol('y')
fx2 = x**2 + 2*x + y**3
d = sympy.diff(fx2, x)


# w, xi, b, di, ai = sympy.symbols("w x_i b d_i a_i")


# J = sympy.S("1/2")*w*w-ai*((w*xi+b)*di-1)
# print(J)
#
# print(J.diff(w))
#
# print(J.diff(b))
#
# wr = sympy.solveset(J.diff(w), w)
# print(wr)
#
# sympy.expand(J)
