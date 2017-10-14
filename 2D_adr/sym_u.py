from __future__ import print_function
from fenics import *
import sympy as sym
'''
## Test_time
sigma = 1.0       # reaction coefficient
mu = 0.001        # diffision coefficient
velocity = as_vector([1.0, 1.0])
a = velocity[0]
b = velocity[1]
## Illiescu Ex 4.1 Test Function Parameters 
'''
mu = 10**-6
sigma = 1.0
velocity = as_vector([2.0, 3.0])
a = velocity[0]
b = velocity[1]

## Use SymPy to compute f from the manufactured solution u
x, y, t = sym.symbols('x[0], x[1], t')

## Iliescu
c = 1 #6#*sym.sin(pi*t)
h = x*(1-x)*y*(1-y)
g = 2*mu**(-0.5)*(0.25*0.25 - (x - 0.5)*(x - 0.5) - (y - 0.5)*(y - 0.5) )

u = c*(0.5*h+1/pi*h*sym.atan(g))



u = sym.simplify(u)

# Final Source Function
f = sym.diff(u,t)-mu*(sym.diff(sym.diff(u,x),x)+sym.diff(sym.diff(u,y),y)) + a*sym.diff(u,x) + b*sym.diff(u,y) + sigma*u
f = sym.simplify(f)

u_code = sym.printing.ccode(u)
f_code = sym.printing.ccode(f)

print('u =', u_code)
print('f =', f_code)

'''
## Code for Exact Solution Output

ue = Expression(u_code, degree=1)
uee = interpolate(ue, Q)

fe = Expression(f_code, degree=1)
fee = project(fe, Q)

out_file_ue = File(folder+"/u_exact_h"+str(nx)+".pvd")
out_file_ue << uee
out_file_fe = File(folder+"/f_exact_h"+str(nx)+".pvd")
out_file_fe << fee
'''
