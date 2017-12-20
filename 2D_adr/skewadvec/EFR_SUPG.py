from dolfin import *
import math as m
import numpy as np
import time

Theta = m.pi/6 # 30 degrees like in Tayfun 2003 paper
aTheta = '30deg'

nx = 100
dt = 2.0*(1.0/nx)

T = 2.0
t = dt - dt

S = 1.0  # filtering radius factor
P = 2    # polynomial degree of FE
R = 2

method = 0 # EFR
N = 0	 # deconvolution order

#method = 1 # SUPG

sigma = 0
mu = 0.001

velocity = Expression(('cos(Theta)',' sin(Theta) '), degree=R, Theta=Theta)
f  = Constant(0.0)


mesh = UnitSquareMesh(nx,nx)
h = CellSize(mesh)
Q = FunctionSpace(mesh, "CG", P)


u_n = Function(Q)

# Define boundaries
inflow = 'near(x[0],0) && x[1]<0.2 && x[1]>=0'		# x=0, 0<y<.2
no_inflow = 'near(x[0],0) && x[1]>=0.2 && x[1]<=1.0'	# x=0, .2<y<1.0
#inflow = 'near(x[0],0)'		# x=0
top = 'near(x[1],1)'		# y=1
bottom = 'near(x[1],0)'		# y=0
outflow = 'near(x[0],1)'	# x=1

# Boundary Conditions for Problem

bc1 = DirichletBC(Q,Constant(0.0),outflow)
bc2 = DirichletBC(Q,Constant(1.0),inflow)
bc3 = DirichletBC(Q,Constant(0.0),top)
bc4 = DirichletBC(Q,Constant(0.0),bottom)
bc5 = DirichletBC(Q,Constant(0.0),no_inflow)

bcu = [bc1, bc2, bc3, bc4, bc5]

# BC for Helmholtz Problem

def boundary(x, on_boundary):
    return on_boundary

# Output files directory

folder = "EFRvsSUPG_changedBC2/"
folder += "1theta"+aTheta+"h"+str(nx)


# Don't Modify Below This! -----------#

# Test and trial functions
u, v = TrialFunction(Q), TestFunction(Q)
u_ = Function(Q)

# Galerkin variational problem
# Backward Euler
F = v*(u - u_n)*dx + dt*(mu*dot(grad(v), grad(u))*dx \
                        + v*dot(velocity, grad(u))*dx \
                        - f*v*dx)


if method == 1:
    r = u - u_n + dt*(- mu*div(grad(u)) + dot(velocity, grad(u)) - f)
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

# Define variational problem for step 2b (evaluate indicator and find filtered solution)
def a3(ind):
    a3 = v*u*dx + delta*delta*dot(grad(v), ind*grad(u))*dx
    return a3
L3 = v*u_*dx

# --- End EFR --- #

# Assemble matrices
A1 = assemble(a1)
[bc.apply(A1) for bc in bcu]

A2 = assemble(a2)
[bc.apply(A2) for bc in bcu]

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
        [bc.apply(b) for bc in bcu]
        solve(A1, u_bar.vector(), b, "gmres")
        out_file_ubar << (u_bar, float(t))

        # Update previous solution and source term
        u_n.assign(u_bar)

        # Update current time
        t += dt

else:
	out_file_ubar = File(folder+"N"+str(N)+"ubar_EFR.pvd")      # filtered solution
	for n in range(num_steps):
		# Step 1 Solve on Coarse Grid
		b = assemble(L)
		[bc.apply(b) for bc in bcu]
		solve(A1, u_.vector(), b, "gmres")

		# Step 2a Solve Helmholtz filter
		if N == 1:
			L2a = v*u_*dx
			b2_0 = assemble(L2a)
			bcF1 = DirichletBC(Q, u_, boundary)
			bcF1.apply(b2_0)
			solve(A2, u_tilde0.vector(), b2_0, "cg")

			L2b = v*u_tilde0*dx
			b2_1 = assemble(L2b)
			bcF2 = DirichletBC(Q, u_tilde0, boundary)
			bcF2.apply(b2_1)
			solve(A2, u_tilde1.vector(), b2_1, "cg")

			DF = Expression('2*a-b', degree = R, a=u_tilde0, b=u_tilde1)

		else: # N = 0
			L2a = v*u_*dx
			b2 = assemble(L2a)
			#[bc.apply(b2) for bc in bcu]
			bcF1 = DirichletBC(Q, u_, boundary)
			bcF1.apply(b2)
			solve(A2, u_tilde0.vector(), b2, "cg")

			DF = Expression('a', degree = R, a=u_tilde0)

		# Step 2b Calculate Indicator and solve Ind Problem
		ind = a(DF, u_, float(t))

		A3 = assemble(a3(ind))
		[bc.apply(A3) for bc in bcu]

		b3 = assemble(L3)
		[bc.apply(b3) for bc in bcu]

		solve(A3, u_bar.vector(), b3, "gmres")    
		progress.update(t / T)

		out_file_utilde << (u_tilde0, float(t))
		out_file_ind << (ind, float(t))
		out_file_ubar << (u_bar, float(t))

		# Update previous solution and source term
		u_n.assign(u_bar)

		# Update current time
		t += dt    
