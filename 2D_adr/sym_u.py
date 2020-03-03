from __future__ import print_function
from fenics import *
import sympy as sym

## About: Uses SymPy to compute f from the manufactured solution u

'''
x, y, t = sym.symbols('x[0], x[1], t')

## Illiescu Ex 4.1 Test Function Parameters 

mu = 10**-6
sigma = 1.0
velocity = as_vector([2.0, 3.0])
a = velocity[0]
b = velocity[1]

c = 1 #6#*sym.sin(pi*t)
h = x*(1-x)*y*(1-y)
g = 2*mu**(-0.5)*(0.25*0.25 - (x - 0.5)*(x - 0.5) - (y - 0.5)*(y - 0.5) )

u = c*(0.5*h+1/pi*h*sym.atan(g))
'''
# ---------------------------------------------------------------- #
'''
## Super Simple Uniform Convective Velocity
sigma = 1.0       # reaction coefficient
mu = 0.001        # diffision coefficient
velocity = as_vector([1.0, 1.0])
a = velocity[0]
b = velocity[1]
'''

# ---------------------------------------------------------------- #
## Non-uniform Convective Velocity 

'''
## Transport Velocity Options
as_vector([1.0, 1.0])			# constant
('x[0]','x[1]') 				# increasing in (1,1) direction
('x[0]-0.5', 'x[1]-0.5') 		# increasing out of center of unit square
('x[0]-0.5', '-(x[1]-0.5)')     # increasing with areas of 0 velocity
('x[1]-.5', '-(x[0]-.5)') 		# clockwise rotation
('-(x[1]-.5)', 'x[0]-.5')  		# counterclockwise rotation
('x[1]-x[0]', '-x[0]-x[1]')		# spiral inward
'''


'''
sigma = 1.0       # reaction coefficient
mu = 0.001        # diffision coefficient

# velocity = Expression(('cos(t)', 'sin(t)') , degree=1, t=t)
# make sure to change a,b in source function code


u = x*x + y**2
u = sym.simplify(u)

# 2D convective velocity
a = (x-0.5)*sym.cos(t)
b = (y-0.5)*sym.sin(t)

# Final Source Function
f = sym.diff(u,t)-mu*(sym.diff(sym.diff(u,x),x)+sym.diff(sym.diff(u,y),y)) + a*sym.diff(u,x) + b*sym.diff(u,y) + sigma*u
f = sym.simplify(f)

a_code = sym.printing.ccode(a)
b_code = sym.printing.ccode(b)
u_code = sym.printing.ccode(u)
f_code = sym.printing.ccode(f)

print('velocity = Expression((\'', a_code, '\',\'', b_code, '\'), degree=R, t=t)' )
print('u_exact = Expression(\'', u_code,'\', degree=R, t=t)')
print('adr_f = Expression(\'', f_code,'\', degree=R, t=t)')

'''



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

class deconv():
	def order(self):
		uptoN = raw_input('Up to what order N are you testing? ')
		utilde = { "N" + str(i) : self.do_something(i) for i in range(int(uptoN)+1) }

		keys = utilde.keys()
		keys.sort()
		for x in keys:
		    print(x, '=', utilde[x])



	def do_something(self, i):
		return "something " + str(i)

deconv().order()


'''


'''