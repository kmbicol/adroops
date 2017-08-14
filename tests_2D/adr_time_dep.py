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
import matplotlib.pyplot as plt
import numpy as np

nx = 20
ny = nx

# Load mesh and subdomains
mesh = UnitSquareMesh(nx,ny)
h = CellSize(mesh)

# Create FunctionSpaces
Q = FunctionSpace(mesh, "CG", 1)

# Create velocity Function from file
velocity = as_vector([1.0, 1.0])

# Initialise source function and previous solution function
f  = Constant(1.0)
u0 = Function(Q)

# Boudanry values
u_D = Constant(0.0)

# Parameters
T = 0.05
dt = 0.01
t = dt
sigma = 0.01
mu = 0.001
nxx = float(nx)		
delta = 1.0/nxx 

# Test and trial functions
u, v = TrialFunction(Q), TestFunction(Q)

# Mid-point solution (Crank-Nicolson)
u_mid = 0.5*(u0 + u)    

# Residual
r = u - u0 + dt*(- mu*div(grad(u_mid)) + dot(velocity, grad(u_mid)) + sigma*u_mid - f)

# Galerkin variational problem
F = v*(u - u0)*dx + dt*(mu*dot(grad(v), grad(u_mid))*dx \
                        + v*dot(velocity, grad(u_mid))*dx \
                        + sigma*v*u*dx \
                        - f*v*dx)

# Add SUPG stabilisation terms
vnorm = sqrt(dot(velocity, velocity))
F += (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx

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
solver = LUSolver(A)
solver.parameters["reuse_factorization"] = True

# Output file
out_file = File("results_time_dep/u.pvd")
out_file_ind = File("results_time_dep/a.pvd")
# Set intial condition
u = u0

# Time-stepping, plot initial condition.
i = 0





while t - T < DOLFIN_EPS:
	# Assemble vector and apply boundary conditions
	b = assemble(L)
	bc.apply(b)

	# Solve the linear system (re-use the already factorized matrix A)
	solver.solve(u.vector(), b)
	'''

	u_1tilde = TrialFunction(Q)
	F_Hfilter0 = v*u_1tilde*dx + delta*delta*dot(grad(v), grad(u_1tilde))*dx - v*u*dx
	a_Hfilter0 = lhs(F_Hfilter0)
	L_Hfilter0 = rhs(F_Hfilter0)

	A_Hfilter0 = assemble(a_Hfilter0)
	bc.apply(A_Hfilter0)
	b_Hfilter0 = assemble(L_Hfilter0)
	bc.apply(b_Hfilter0)

	solver0 = LUSolver(A_Hfilter0)
	u_1tilde = Function(Q)
	solver0.solve(u_1tilde.vector(), b_Hfilter0)
	DF = u_1tilde
	#out_file_utilde << u_1tilde

	indicator = Expression('sqrt((a-b)*(a-b))', degree = 2, a = u, b = DF)
	indicator = interpolate(indicator, Q)
	max_ind = np.amax(indicator.vector().array())

	# Normalize indicator such that it's between [0,1].
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

	solver_filter = LUSolver(A_filter)
	u_bar = Function(Q)
	solver_filter.solve(u_bar.vector(), b_filter)

	# Copy solution from previous interval
	u0 = 0.5*u + 0.5*u_bar
	'''
	# Save the solution to file
	out_file << (u, t)

	# Move to next interval and adjust boundary condition
	t += dt
	i += 1
