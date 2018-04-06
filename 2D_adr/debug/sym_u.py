
from __future__ import print_function
from fenics import *
import sympy as sym

## About: Uses SymPy to compute f from the manufactured solution u

x, y, t = sym.symbols('x[0], x[1], t')

# ---------------------------------------------------------------- #

## Super Simple Uniform Convective Velocity
sigma = 1.0       # reaction coefficient
mu = 0.001        # diffision coefficient
velocity = as_vector([1.0, 1.0])
a = velocity[0]
b = velocity[1]


# ---------------------------------------------------------------- #
## Non-uniform Convective Velocity 


## Transport Velocity Options
as_vector([1.0, 1.0])			# constant
('x[0]','x[1]') 				# increasing in (1,1) direction
('x[0]-0.5', 'x[1]-0.5') 		# increasing out of center of unit square
('x[0]-0.5', '-(x[1]-0.5)')     # increasing with areas of 0 velocity
('x[1]-.5', '-(x[0]-.5)') 		# clockwise rotation
('-(x[1]-.5)', 'x[0]-.5')  		# counterclockwise rotation
('x[1]-x[0]', '-x[0]-x[1]')		# spiral inward




sigma = 1.0       # reaction coefficient
mu = 0.001        # diffision coefficient

# velocity = Expression(('cos(t)', 'sin(t)') , degree=1, t=t)
# make sure to change a,b in source function code


u = sym.exp(y) + sym.exp(x)
u = sym.simplify(u)


# Final Source Function

#ADR

'''
# 2D convective velocity
a = sym.cos(t)
b = sym.sin(t)

f = sym.diff(u,t)-mu*(sym.diff(sym.diff(u,x),x)+sym.diff(sym.diff(u,y),y)) + a*sym.diff(u,x) + b*sym.diff(u,y) + sigma*u
f = sym.simplify(f)
'''

# Filter


f = sym.diff(sym.diff(u,x),x) + sym.diff(sym.diff(u,y),y) 
f = sym.simplify(f)


u_code = sym.printing.ccode(u)
f_code = sym.printing.ccode(f)


print('u_exact = Expression(\'', u_code,'\', degree=R, t=t)')
print('adr_f = Expression(\'', f_code,'\', degree=R, t=t)')

