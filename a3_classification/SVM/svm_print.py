
import sympy as sym
sym.init_printing()

#%%

w, xi, b, di, ai = sym.symbols("w x_i b d_i a_i")


J = sym.S("1/2")*w*w-ai*((w*xi+b)*di-1)
print(J)

print(J.diff(w))

print(J.diff(b))

wr = sym.solveset(J.diff(w), w)
print(wr)

sym.expand(J)
