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


# Load mesh and subdomains
mesh = UnitSquareMesh(20,20)
h = CellSize(mesh)

# Create FunctionSpaces
Q = FunctionSpace(mesh, "CG", 1)

# Create velocity Function from file
velocity = as_vector([1.0, 1.0])

# Parameters
T = 0.03
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

# Mid-point solution
u_mid = 0.5*(u0 + u)
f_mid = 0.5*(f0 + f)
print type(u_mid)
# Residual
r = u - u0 + dt*(- mu*div(grad(u_mid)) + dot(velocity, grad(u_mid)) + sigma*u_mid - f)

# Galerkin variational problem
F = v*(u - u0)*dx + dt*(mu*dot(grad(v), grad(u_mid))*dx \
                        + v*dot(velocity, grad(u_mid))*dx \
                        + sigma*v*u*dx \
                        - f_mid*v*dx)

# Add SUPG stabilisation terms
vnorm = sqrt(dot(velocity, velocity))
#F += (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx

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
out_file = File("results_time_dep/u_test.pvd")
out_filef = File("results_time_dep/f.pvd")
out_filefmid = File("results_time_dep/fmid.pvd")
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
    #out_filef << (f,t)
    #out_filefmid << (f_mid,t)
    # Copy solution from previous interval
    u0.assign(u)
    f0=f
    # Save the solution to file
    out_file << (u, t)


    # Move to next interval and adjust boundary condition
    t += dt
    f.t = t    
    i += 1
