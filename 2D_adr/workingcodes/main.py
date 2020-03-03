from dolfin import *
import math as m
import numpy as np
import time

# Load mesh and subdomains
method = input("0: No Filter, 1: SUPG, 2: EFR; method = ")
if method == 2:
	N = input("N = 0 or 1, N = ")
nx = input("h = 1/nx, let nx = ")
dt = 0.01

T = 1.0
t = dt - dt

S = 1.0  # filtering radius factor
P = 2    # polynomial degree of FE
R = 2

sigma = 1.0
mu = 0.001
velocity = as_vector([1.0, 1.0])
f  = Expression('t', degree=R, t=t)

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

folder = "test/h"+str(nx)+"_"


# Don't Modify Below This! -----------#

# Test and trial functions
u, v = TrialFunction(Q), TestFunction(Q)
u_ = Function(Q)
u_bar = Function(Q)

# Galerkin variational problem
# Backward Euler
F = v*(u - u_n)*dx + dt*(mu*dot(grad(v), grad(u))*dx \
                        + v*dot(velocity, grad(u))*dx \
                        + sigma*v*u*dx \
                        - f*v*dx)

if method == 0:
    methodname = "nofilter"

if method == 1:
    methodname = "SUPG"
    r = u - u_n + dt*(- mu*div(grad(u)) + dot(velocity, grad(u)) + sigma*u - f)
    vnorm = sqrt(dot(velocity, velocity))
    F += (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx

if method == 2:
	# --- Begin EFR --- #
	methodname = "EFR"
	delta = S*1.0/nx

	u_tilde0 = Function(Q)
	u_tilde1 = Function(Q)



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
	A2 = assemble(a2)

	def L2(u_): # input is intermediate velocity OR previous u_tilde solution
	    L2 = v*u_*dx
	    return L2

	# Define variational problem for step 2b (evaluate indicator and find filtered solution)
	def a3(ind):
	    a3 = v*u*dx + delta*delta*dot(grad(v), ind*grad(u))*dx
	    return a3
	L3 = v*u_*dx

	# --- End EFR --- #



# Create bilinear and linear forms
a1 = lhs(F)
L = rhs(F)

# Assemble matrices
A1 = assemble(a1)
bc.apply(A1)

# Create progress bar
progress = Progress('Time-stepping')
set_log_level(PROGRESS)


num_steps = int(round(T / dt, 0)) 

# --- Time-stepping --- #

if method == 1 or method == 0:
    out_file_ubar = File(folder+"u_"+methodname+".pvd")      # filtered solution
    for n in range(num_steps):
        b = assemble(L)
        bc.apply(b)
        solve(A1, u_bar.vector(), b, "gmres")
        out_file_ubar << (u_bar, float(t))

        # Update previous solution and source term
        u_n.assign(u_bar)

        # Update current time
        t += dt
        f.t += dt


else:
	out_file_utilde = File(folder+"utilde_N"+str(N)+"_EFR.pvd")  # u tilde
	out_file_ind = File(folder+"a_N"+str(N)+"_EFR.pvd")          # indicator function
	out_file_ubar = File(folder+"ubar_N"+str(N)+"_EFR.pvd")      # filtered solution

	for n in range(num_steps):
		# Step 1 Solve on Coarse Grid
		b = assemble(L)
		bc.apply(b)
		solve(A1, u_.vector(), b, "gmres")

		# Step 2a Solve Helmholtz filter
		if N == 1:
		    b2_0 = assemble(L2(u_))
		    bc_0 = DirichletBC(Q, u_, boundary)
		    bc_0.apply(b2_0)
		    bc_0.apply(A2)
		    solve(A2, u_tilde0.vector(), b2_0, "cg")

		    b2_1 = assemble(L2(u_tilde0))
		    bc_1 = DirichletBC(Q, u_tilde0, boundary)
		    bc_1.apply(b2_1)
		    bc_1.apply(A2)
		    solve(A2, u_tilde1.vector(), b2_1, "cg")
		    DF = Expression('2*a-b', degree = R, a=u_tilde0, b=u_tilde1)
		else: # N=0
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
		out_file_ind << (ind, float(t))

		progress.update(t / T)

		out_file_ubar << (u_bar, float(t))

		# Update previous solution and source term
		u_n.assign(u_bar)

		# Update current time
		f.t += dt


print(methodname+" ")
if method == 2: 
	print("N = "+str(N))