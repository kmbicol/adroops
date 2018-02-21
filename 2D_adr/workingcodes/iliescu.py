from __future__ import print_function
from fenics import *
import sympy as sym

## About: Uses SymPy to compute f from the manufactured solution u
''

x, y, t = sym.symbols('x[0], x[1], t')

## Illiescu Ex 4.1 Test Function Parameters 

mu = 10**(-6)
sigma = 1.0
velocity = as_vector([2.0, 3.0])
a = velocity[0]
b = velocity[1]

c = 16*sym.sin(pi*t)
h = x*(1-x)*y*(1-y)
g = 2*sym.sqrt(mu)*( 0.25**2 - (x - 0.5)**2 - (y - 0.5)**2 )

u1 = c*(h*0.5+h*sym.atan(g)/pi)
u2 = c*h*(0.5+sym.atan(g)/pi)

# Final Source Function
f1 = sym.diff(u1,t)-mu*(sym.diff(sym.diff(u1,x),x)+sym.diff(sym.diff(u1,y),y)) + a*sym.diff(u1,x) + b*sym.diff(u1,y) + sigma*u1


f2 = sym.diff(u2,t)-mu*(sym.diff(sym.diff(u2,x),x)+sym.diff(sym.diff(u2,y),y)) + a*sym.diff(u2,x) + b*sym.diff(u2,y) + sigma*u2

f = f1-f2
f = sym.simplify(f)
diff_f = sym.printing.ccode(f)

print(diff_f)
'''
u1_code = sym.printing.ccode(u1)

u2_code = sym.printing.ccode(u2)


print('u1 = Expression(\'', u1_code,'\', degree=R, t=t)')
print('\n')
print('u2 = Expression(\'', u2_code,'\', degree=R, t=t)')

#print('adr_f = Expression(\'', f_code,'\', degree=R, t=t)')

diffy = u1-u2
diffy = sym.simplify(diffy)
difference = sym.printing.ccode(diffy)

print(difference)
'''