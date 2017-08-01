# Copyright (C) 2007 Kristian B. Oelgaard
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2008
# Modified by Johan Hake, 2008
# Modified by Garth N. Wells, 2009
#
# This demo solves the time-dependent convection-diffusion equation by
# a SUPG stabilized method. The velocity field used in the simulation
# is the output from the Stokes (Taylor-Hood) demo.  The sub domains
# for the different boundary conditions are computed by the demo
# program in src/demo/subdomains.

from dolfin import *
from fenics import *
import numpy as np

# Load mesh and subdomains
mesh = UnitSquareMesh(20,20) # divides [0,1]x[0,1] into 20x20 
h = CellSize(mesh)

# Create FunctionSpaces
Q = FunctionSpace(mesh, "CG", 1)

# Initialise source function and previous solution function
f  = Constant(1.0)

# Boundary values
u_D = Constant(0.0)

# Parameters
sigma = 0.01
mu = 0.001
velocity = as_vector([1.0, 1.0]) # this is b
delta = h  # this is where we change the filtering radius

# Test and trial functions
u, v = TrialFunction(Q), TestFunction(Q)

# Residual
r = - mu*div(grad(u)) + dot(velocity, grad(u)) + sigma*u - f

# Galerkin variational problem
F = mu*dot(grad(v), grad(u))*dx + v*dot(velocity, grad(u))*dx + sigma*v*u*dx - f*v*dx

# SUPG stabilisation terms
vnorm = sqrt(dot(velocity, velocity))
F_SUPG = F + (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx

# GLS stabilization terms
F_GLS = F + (h/(2.0*vnorm))*(dot(velocity, grad(v)) - mu*div(grad(v)) + sigma*v)*r*dx

# Create bilinear and linear forms
a = lhs(F)
L = rhs(F)
a_SUPG = lhs(F_SUPG)
L_SUPG = rhs(F_SUPG)
a_GLS = lhs(F_GLS)
L_GLS = rhs(F_GLS)

# Set up boundary condition
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(Q, u_D, boundary)

# Output file
out_file_u = File("results_steady/u.pvd")
out_file_usupg = File("results_steady/u_SUPG.pvd")
out_file_ugls = File("results_steady/u_GLS.pvd")
out_file_utilde = File("results_steady/u_tilde.pvd")
out_file_ubar = File("results_steady/u_bar.pvd")
out_file_ind = File("results_steady/indicator.pvd")

# Assemble matrix
A = assemble(a)
bc.apply(A)

A_SUPG = assemble(a_SUPG)
bc.apply(A_SUPG)

A_GLS = assemble(a_GLS)
bc.apply(A_GLS)

# Assemble vector and apply boundary conditions
b = assemble(L)
bc.apply(b)

b_SUPG = assemble(L_SUPG)
bc.apply(b_SUPG)

b_GLS = assemble(L_GLS)
bc.apply(b_GLS)

# Create linear solver and factorize matrix
solver = LUSolver(A)
u = Function(Q)
solver.solve(u.vector(), b)

print 'norm_u_L2  =',   norm(u, 'L2')

solver = LUSolver(A_SUPG)
u_SUPG = Function(Q)
solver.solve(u_SUPG.vector(), b_SUPG)

print 'norm_u_SUPG_L2  =',   norm(u_SUPG, 'L2')

solver = LUSolver(A_GLS)
u_GLS = Function(Q)
solver.solve(u_GLS.vector(), b_GLS)

print 'norm_u_GLS_L2  =',   norm(u_GLS, 'L2')

# Compute difference between SUPG and GLS solution in L2 norm
#diff_SUPG_GLS_L2 = norm(u_SUPG.vector() - u_GLS.vector(), 'L2')/norm(u_SUPG.vector(), 'L2')
#print 'difference_SUPG_GLS_L2  =', diff_SUPG_GLS_L2 

# Save the solution to file

out_file_u << u

out_file_usupg << u_SUPG

out_file_ugls << u_GLS

##################################################################################

# Helmholtz filter to compute the indicator function
u_tilde = TrialFunction(Q)

#deltaH = h*vnorm/(2*mu)

F_Hfilter = v*u_tilde*dx + delta*delta*dot(grad(v), grad(u_tilde))*dx - v*u*dx 

a_Hfilter = lhs(F_Hfilter)
L_Hfilter = rhs(F_Hfilter)

A_Hfilter = assemble(a_Hfilter)
bc.apply(A_Hfilter)

b_Hfilter = assemble(L_Hfilter)
bc.apply(b_Hfilter)

solver = LUSolver(A_Hfilter)
u_tilde = Function(Q)
solver.solve(u_tilde.vector(), b_Hfilter)

out_file_utilde << u_tilde

# Compute the indicator function N = 0
indicator = Expression('sqrt((a-b)*(a-b))', degree = 2, a = u, b = u_tilde)
indicator = interpolate(indicator, Q)
max_ind = np.amax(indicator.vector().array())

if max_ind < 1:
   max_ind = 1.0

indicator = Expression('a/b', degree = 2, a = indicator, b = max_ind)
indicator = interpolate(indicator, Q)

out_file_ind << indicator

# Apply the filter
u_bar = TrialFunction(Q)
F_filter = v*u_bar*dx + delta*delta*dot(grad(v), indicator*grad(u_bar))*dx - v*u*dx 

a_filter = lhs(F_filter)
L_filter = rhs(F_filter)

A_filter = assemble(a_filter)
bc.apply(A_filter)

b_filter = assemble(L_filter)
bc.apply(b_filter)

solver = LUSolver(A_filter)
u_bar = Function(Q)
solver.solve(u_bar.vector(), b_filter)

print 'norm_u_bar_L2  =',   norm(u_bar, 'L2')

out_file_ubar << u_bar
