from __future__ import print_function
from fenics import *
import sympy as sym

## Calculates cpp code for Illiescu Test Function for Ex 4.1

mu = 10**-6
sigma = 1.0
velocity = as_vector([2.0, 3.0])
a = velocity[0]
b = velocity[1]

# Use SymPy to compute f from the manufactured solution u
'''
x, y, t = sym.symbols('x[0], x[1], t')

# R = atan(g)

old code where i was dumb and didn't know sym.atan was a thing....

c = 16*sym.sin(pi*t)*sym.atan(t)
h = x*(1-x)*y*(1-y)
g = 2*mu**(-0.5)*(0.25*0.25 - (x - 0.5)*(x - 0.5) - (y - 0.5)*(y - 0.5) )

Rx = (1+g**2)**(-1)*sym.diff(g,x)
Ry = (1+g**2)**(-1)*sym.diff(g,y)

u = c*(0.5*h+1/pi*h*sym.atan(g))
u = sym.simplify(u)

ux = 0.5*sym.diff(h,x)+1/pi*(h*Rx+sym.diff(h,x)*sym.atan(g))
uy = 0.5*sym.diff(h,y)+1/pi*(h*Ry+sym.diff(h,y)*sym.atan(g))

uxx = 0.5*sym.diff(sym.diff(h,x),x) + 1/pi*(h*sym.diff(Rx,x) + 2*sym.diff(h,x)*Rx + sym.diff(sym.diff(h,x),x)*sym.atan(g))
uyy = 0.5*sym.diff(sym.diff(h,y),y) + 1/pi*(h*sym.diff(Ry,y) + 2*sym.diff(h,y)*Ry + sym.diff(sym.diff(h,y),y)*sym.atan(g))
'''

x, y, t = sym.symbols('x[0], x[1], t')

c = 16*sym.sin(pi*t)
h = x*(1-x)*y*(1-y)
g = 2*mu**(-0.5)*(0.25*0.25 - (x - 0.5)*(x - 0.5) - (y - 0.5)*(y - 0.5) )

u = c*(0.5*h+1/pi*h*sym.atan(g))
u = sym.simplify(u)

# final source function
f = -mu*(sym.diff(sym.diff(u,x),x)+sym.diff(sym.diff(u,y),y)) + a*sym.diff(u,x) + b*sym.diff(u,y) + sigma*u
f = sym.simplify(f)

u_code = sym.printing.ccode(u)
f_code = sym.printing.ccode(f)

print('u =', u_code)
print('f =', f_code)

'''
## Code Output with atan replaced with atan(g_code)
u = x[0]*x[1]*(5.09295817894065*atan(-2000.0*pow(x[0] - 0.5, 2) - 2000.0*pow(x[1] - 0.5, 2) + 125.0) + 8.0)*(x[0] - 1)*(x[1] - 1)
g = -2000.0*pow(x[0] - 0.5, 2) - 2000.0*pow(x[1] - 0.5, 2) + 125.0
f = (3.18309886183791e-7*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(-(4000.0*x[0] - 2000.0)*(8000.0*x[0] - 4000.0)*(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - (4000.0*x[1] - 2000.0)*(8000.0*x[1] - 4000.0)*(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) + 8000.0*pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 8000.0) + pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)*(0.954929658551372*atan(-2000.0*pow(x[0] - 0.5, 2) - 2000.0*pow(x[1] - 0.5, 2) + 125.0)*x[0]*(x[0] - 1)*(2*x[1] - 1) - 6.36619772367581e-7*atan(-2000.0*pow(x[0] - 0.5, 2) - 2000.0*pow(x[1] - 0.5, 2) + 125.0)*x[0]*(x[0] - 1) + 0.636619772367581*atan(-2000.0*pow(x[0] - 0.5, 2) - 2000.0*pow(x[1] - 0.5, 2) + 125.0)*x[1]*(2*x[0] - 1)*(x[1] - 1) - 6.36619772367581e-7*atan(-2000.0*pow(x[0] - 0.5, 2) - 2000.0*pow(x[1] - 0.5, 2) + 125.0)*x[1]*(x[1] - 1) + 1.0*x[0]*x[1]*(5.09295817894065*atan(-2000.0*pow(x[0] - 0.5, 2) - 2000.0*pow(x[1] - 0.5, 2) + 125.0) + 8.0)*(x[0] - 1)*(x[1] - 1) + 1.5*x[0]*x[1]*(x[0] - 1) + 1.0*x[0]*x[1]*(x[1] - 1) + 1.5*x[0]*(x[0] - 1)*(x[1] - 1) - 1.0e-6*x[0]*(x[0] - 1) + 1.0*x[1]*(x[0] - 1)*(x[1] - 1) - 1.0e-6*x[1]*(x[1] - 1)) + (pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)*(-0.636619772367581*x[0]*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1) - 0.954929658551372*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 6.36619772367581e-7*x[0]*(x[0] - 1)*(2*x[1] - 1)*(4000.0*x[1] - 2000.0) + 6.36619772367581e-7*x[1]*(2*x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1)))/pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)
'''