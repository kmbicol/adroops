"""
FEniCS tutorial demo program: Nonlinear Poisson equation.
  -div(q(u)*grad(u)) = f   in the unit square.
                   u = u_D on the boundary.
"""

from __future__ import print_function
# Warning: from fenics import * will import both `sym` and
# `q` from FEniCS. We therefore import FEniCS first and then
# overwrite these objects.
from fenics import *

mu = 10**-6
sigma = 1.0
velocity = as_vector([2.0, 3.0])
a = velocity[0]
b = velocity[1]

# Use SymPy to compute f from the manufactured solution u
x, y = sym.symbols('x[0], x[1]')
u = (2*(0.25*0.25-(x-0.5)*(x-0.5)-(y-0.5)*(y-0.5)))/pow(mu,0.5)
u = sym.simplify(u)
u_poly = 16*x*(1-x)*y*(1-y)
#u_atan = (0.5 + atan(x)/pi)
f = u_poly#-mu*sym.diff(sym.diff(u, x), x) + sym.diff(sym.diff(u, y), y) + a*sym.diff(u, x) + b*sym.diff(u, y) + sigma*u
f = sym.simplify(f)
u_code = sym.printing.ccode(u)
u_poly_code = sym.printing.ccode(u_poly)
f_code = sym.printing.ccode(f)
print('u =', u_code)
print('f =', f_code)

#sin(pi*t)

'''
# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression(u_code, degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = Function(V)  # Note: not TrialFunction!
v = TestFunction(V)
f = Expression(f_code, degree=2)
F = q(u)*dot(grad(u), grad(v))*dx - f*v*dx

# Compute solution
solve(F == 0, u, bc)

# Plot solution
plot(u)

# Compute maximum error at vertices. This computation illustrates
# an alternative to using compute_vertex_values as in poisson.py.
u_e = interpolate(u_D, V)
import numpy as np
error_max = np.abs(u_e.vector().array() - u.vector().array()).max()
print('error_max = ', error_max)
'''