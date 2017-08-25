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
mesh = UnitSquareMesh(20,20)
h = CellSize(mesh)
delta = 1.0/20.0

# Create FunctionSpaces
Q = FunctionSpace(mesh, "CG", 1) # CG == Continuous Galerkin (Lagrange)

# Create velocity Function from file
velocity = as_vector([1.0, 1.0])

# Parameters
T = 1
dt = 0.01
t = dt
sigma = 1.0
mu = 0.001

# Initialise source function and previous solution function 
# f0 = Constant(1.0) 
t0=0
f = Expression('3+sigma*(t+x[0]+x[1])', degree = 1, sigma = sigma, t = t0) 
f0 = f
f.t += dt
u0 = Function(Q)

# Boudanry values
u_D = Constant(0.0)

# Test and trial functions
u, v = TrialFunction(Q), TestFunction(Q)

# Mid-point solution  (for Crank-Nicolson)
u_mid = 0.5*(u0 + u)
f_mid = 0.5*(f0 + f)

# Residual
r = u - u0 + dt*(- mu*div(grad(u_mid)) + dot(velocity, grad(u_mid)) + sigma*u_mid - f)

# Galerkin variational problem
F = v*(u - u0)*dx + dt*(mu*dot(grad(v), grad(u_mid))*dx \
                        + v*dot(velocity, grad(u_mid))*dx \
                        + sigma*v*u*dx \
                        - f_mid*v*dx)

# Add SUPG stabilisation terms
# vnorm = sqrt(dot(velocity, velocity))
# F += (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx

# Create bilinear and linear forms
a = lhs(F)
L = rhs(F)

# Set up boundary condition
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(Q, u_D, boundary)

# Assemble matrix
A = assemble(a)
bc.apply(A)

# Create linear solver and factorize matrix
u = Function(Q)
solver = LUSolver(A)
solver.parameters["reuse_factorization"] = True

# Set intial condition
u = u0

# Output files
out_file = File("results_time_dep/u_nofilter.pvd")
out_file_utilde = File("results_time_dep/utilde.pvd")
out_file_ind = File("results_time_dep/a.pvd")
out_file_ubar = File("results_time_dep/ubar.pvd")

#while t - T < DOLFIN_EPS:
    # Assemble vector and apply boundary conditions
b = assemble(L)
bc.apply(b)
# Solve the linear system (re-use the already factorized matrix A)
solver.solve(u.vector(), b)

out_file << (u, t)

########################################################

# Helmholtz filter to compute the indicator function
u_tilde = TrialFunction(Q)
#F_Hfilter = v*u_tilde*dx + delta*delta*dot(grad(v), grad(u_tilde))*dx - v*u*dx 
a_Hfilter = v*u_tilde*dx + delta*delta*dot(grad(v), grad(u_tilde))*dx #lhs(F_Hfilter)
L_Hfilter = v*u*dx #rhs(F_Hfilter)

A_Hfilter = assemble(a_Hfilter)
bc.apply(A_Hfilter)
solver1 = LUSolver(A_Hfilter)
solver1.parameters["reuse_factorization"] = True

########################################################
b_Hfilter = assemble(L_Hfilter)
bc.apply(b_Hfilter)
u_tilde = Function(Q)
solver1.solve(u_tilde.vector(), b_Hfilter)

out_file_utilde << (u_tilde, t)

########################################################

u_bar = TrialFunction(Q)
indicator = Expression('sqrt((a-b)*(a-b))', degree = 2, a = u, b = u_tilde)
indicator = interpolate(indicator, Q)

F_filter = v*u_bar*dx + delta*delta*dot(grad(v), indicator*grad(u_bar))*dx - v*u*dx 
a_filter = lhs(F_filter)
L_filter = rhs(F_filter)

A_filter = assemble(a_filter)
bc.apply(A_filter)
solver2 = LUSolver(A_filter)
solver2.parameters["reuse_factorization"] = True

#######################################################

# Compute the indicator function N = 0
max_ind = np.amax(indicator.vector().array())

if max_ind < 1:
    max_ind = 1.0

indicator = Expression('a/b', degree = 2, a = indicator, b = max_ind)
indicator = interpolate(indicator, Q)

out_file_ind << (indicator, t)

# Apply the filter dependent on indicator output
b_filter = assemble(L_filter)
bc.apply(b_filter)
u_bar = Function(Q)
solver2.solve(u_bar.vector(), b_filter)

out_file_ubar << (u_bar, t)

# Copy solution from previous interval
u0.assign(u_bar)

#u0.assign(u) # no filter code
f0=f

# Move to next interval and adjust boundary condition
t += dt
f.t = t    



################################################################## next time step

b = assemble(L)
bc.apply(b)
# Solve the linear system (re-use the already factorized matrix A)
solver.solve(u.vector(), b)

out_file << (u, t)

########################################################

# Helmholtz filter to compute the indicator function
u_tilde = TrialFunction(Q)
#F_Hfilter = v*u_tilde*dx + delta*delta*dot(grad(v), grad(u_tilde))*dx - v*u*dx 
a_Hfilter = v*u_tilde*dx + delta*delta*dot(grad(v), grad(u_tilde))*dx #lhs(F_Hfilter)
L_Hfilter = v*u*dx #rhs(F_Hfilter)

A_Hfilter = assemble(a_Hfilter)
bc.apply(A_Hfilter)
solver1 = LUSolver(A_Hfilter)
solver1.parameters["reuse_factorization"] = True

########################################################
b_Hfilter = assemble(L_Hfilter)
bc.apply(b_Hfilter)
u_tilde = Function(Q)
solver1.solve(u_tilde.vector(), b_Hfilter)

out_file_utilde << (u_tilde, t)

########################################################

u_bar = TrialFunction(Q)
indicator = Expression('sqrt((a-b)*(a-b))', degree = 2, a = u, b = u_tilde)
indicator = interpolate(indicator, Q)

F_filter = v*u_bar*dx + delta*delta*dot(grad(v), indicator*grad(u_bar))*dx - v*u*dx 
a_filter = lhs(F_filter)
L_filter = rhs(F_filter)

A_filter = assemble(a_filter)
bc.apply(A_filter)
solver2 = LUSolver(A_filter)
solver2.parameters["reuse_factorization"] = True

#######################################################

# Compute the indicator function N = 0
max_ind = np.amax(indicator.vector().array())

if max_ind < 1:
    max_ind = 1.0

indicator = Expression('a/b', degree = 2, a = indicator, b = max_ind)
indicator = interpolate(indicator, Q)

out_file_ind << (indicator, t)

# Apply the filter dependent on indicator output
b_filter = assemble(L_filter)
bc.apply(b_filter)
u_bar = Function(Q)
solver2.solve(u_bar.vector(), b_filter)

out_file_ubar << (u_bar, t)

# Copy solution from previous interval
u0.assign(u_bar)

#u0.assign(u) # no filter code
f0=f

# Move to next interval and adjust boundary condition
t += dt
f.t = t    