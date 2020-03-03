from dolfin import *
import math as m
import numpy as np
import time


# Load mesh and subdomains
nx = 20
dt = 0.01

T = 2.0
t = dt - dt

S = 1.0  # filtering radius factor
P = 1    # polynomial degree of FE
R = 1

method = 0 # EFR
N = 1	 # deconvolution order

#method = 1 # SUPG

sigma = 1.0
mu = 0.001
velocity = as_vector([1.0, 1.0])
f  = Constant(1.0)


mesh = UnitSquareMesh(nx,nx)
h = CellSize(mesh)
Q = FunctionSpace(mesh, "CG", P)


u_n = Function(Q)


u_D = Constant(0.0)


# Set up boundary condition
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(Q, u_D, boundary)


# Output files directory

folder = "EFRvsSUPG/"


# Don't Modify Below This! -----------#

# Test and trial functions
u, v = TrialFunction(Q), TestFunction(Q)
u_ = Function(Q)

# Galerkin variational problem
# Backward Euler
F = v*(u - u_n)*dx + dt*(mu*dot(grad(v), grad(u))*dx \
                        + v*dot(velocity, grad(u))*dx \
                        + sigma*v*u*dx \
                        - f*v*dx)


if method == 1:
    r = u - u_n + dt*(- mu*div(grad(u)) + dot(velocity, grad(u)) + sigma*u - f)
    vnorm = sqrt(dot(velocity, velocity))
    F += (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx
    methodname = "SUPG"


# Create bilinear and linear forms
a1 = lhs(F)
L = rhs(F)

# --- Begin EFR --- #
delta = S*1.0/nx

u_tilde0 = Function(Q)
u_tilde1 = Function(Q)
u_bar = Function(Q)


# Define indicator function to evaluate current time step
def a(u_tilde, u_, t):
    indicator = Expression('sqrt((a-b)*(a-b))', degree = 2, a = u_, b = u_tilde)
    indicator = interpolate(indicator, Q)
    max_ind = np.amax(indicator.vector().array())

    # Normalize indicator such that it's between [0,1].
    if max_ind < 1:
       max_ind = 1.0

    indicator = Expression('a/b', degree = 2, a = indicator, b = max_ind)
    indicator = interpolate(indicator, Q) 
    indicator.rename('a','a')
    out_file_ind << (indicator, float(t))
    return indicator


# Define variational problem for step 2a (apply Helmholz filter)
# Note: only RHS changes, so we can keep the same a2 throughout

a2 = v*u*dx + delta*delta*dot(grad(v), grad(u))*dx #lhs(F_Hfilter)

def L2(u_): # input is intermediate velocity OR previous u_tilde solution
	L2 = v*u_*dx
	return L2

# Define variational problem for step 2b (evaluate indicator and find filtered solution)
def a3(ind):
    a3 = v*u*dx + delta*delta*dot(grad(v), ind*grad(u))*dx
    return a3
L3 = v*u_*dx

# --- End EFR --- #

# Assemble matrices
A1 = assemble(a1)
bc.apply(A1)

A2 = assemble(a2)
bc.apply(A2)

# Create progress bar
progress = Progress('Time-stepping')
set_log_level(PROGRESS)


num_steps = int(round(T / dt, 0)) 


out_file = File(folder+"u_nofilter.pvd")
out_file_utilde = File(folder+"utilde_EFR.pvd")  # u tilde
out_file_ind = File(folder+"a_EFR.pvd")          # indicator function

# Time-stepping

if method == 1:
    out_file_ubar = File(folder+"u_SUPG.pvd")      # filtered solution
    for n in range(num_steps):
        b = assemble(L)
        bc.apply(b)
        solve(A1, u_bar.vector(), b, "gmres")
        out_file_ubar << (u_bar, float(t))

        # Update previous solution and source term
        u_n.assign(u_bar)

        # Update current time
        t += dt

else:
	out_file_ubar = File(folder+"ubar_EFR_N"+str(N)+".pvd")      # filtered solution
	for n in range(num_steps):
		# Step 1 Solve on Coarse Grid
		b = assemble(L)
		bc.apply(b)
		solve(A1, u_.vector(), b, "gmres")

		# Step 2a Solve Helmholtz filter
		if N == 1:
			b2_0 = assemble(L2(u_))
			bc.apply(b2_0)
			solve(A2, u_tilde0.vector(), b2_0, "cg")
			b2_1 = assemble(L2(u_tilde0))
			bc.apply(b2_1)
			solve(A2, u_tilde1.vector(), b2_1, "cg")
			DF = Expression('2*a-b', degree = R, a=u_tilde0, b=u_tilde1)
		else:
			b2 = assemble(L2(u_))
			bc.apply(b2)
			solve(A2, u_tilde0.vector(), b2, "cg")
			DF = Expression('a', degree = R, a=u_tilde0)

		# Step 2b Calculate Indicator and solve Ind Problem
		ind = a(DF, u_, float(t))

		A3 = assemble(a3(ind))
		bc.apply(A3)

		b3 = assemble(L3)
		bc.apply(b3)

		solve(A3, u_bar.vector(), b3, "gmres")    
		progress.update(t / T)

		out_file_ind << (ind, float(t))
		out_file_ubar << (u_bar, float(t))

		# Update previous solution and source term
		u_n.assign(u_bar)

		# Update current time
		t += dt    
