from dolfin import *
import math as m
import numpy as np
import time

#print("\n (1) SUPG \n (2) GLS \n (3) DW \n (4) VMS \n (5) Galerkin and Exact Solution \n (default) Galerkin")
#method = input("Choose a stabilization method: ")
#nx = input("h = 1/nx, let nx = ")

method = 1
Theta = m.pi/3 # 30 degrees like in Tayfun 2003 paper
aTheta = '60deg'

# Simulation Parameters
nx = 20
T = 5 #2.0
dt = .1
t = dt -dt
R = 2
P = 2
saveTimesteps = 1 # save every __ time steps
folder = "results_skewadv/theta"+aTheta+"h"+str(nx)

savetimes = 0
if savetimes != 0:
	compfile = "stab_comptime.txt"


# Load mesh and subdomains
mesh = UnitSquareMesh(nx,nx)
mesh_file = File("mesh.pvd")
mesh_file << mesh

h = CellSize(mesh)

# Create FunctionSpaces
Q = FunctionSpace(mesh, "CG", P)

# Initialise source function and previous solution function

#f  = Constant(1.0)
u_n = Function(Q) # automatically sets u_n as 0

# Test and trial functions
u, v = TrialFunction(Q), TestFunction(Q)
u_ = Function(Q)


sigma = 0       # reaction coefficient
mu = 10**(-6)        # diffision coefficient


velocity = Expression(('cos(Theta)',' sin(Theta) '), degree=R, Theta=Theta)
source_f = Constant(0.0)


f = Constant(0.0)
f0 = Constant(0.0)

# Define boundaries
inflow = 'near(x[0],0) && x[1]<=0.2 && x[1]>=0'		# x=0, 0<y<.2
no_inflow = 'near(x[0],0) && x[1]>0.2 && x[1]<=1.0'	# x=0, .2<y<1.0
#inflow = 'near(x[0],0)'		# x=0
top = 'near(x[1],1)'		# y=1
bottom = 'near(x[1],0)'		# y=0
outflow = 'near(x[0],1)'	# x=1

bc1 = DirichletBC(Q,Constant(0.0),outflow)
bc2 = DirichletBC(Q,Constant(0.0),inflow)
bc3 = DirichletBC(Q,Constant(0.0),top)
bc4 = DirichletBC(Q,Constant(0.0),bottom)
bc5 = DirichletBC(Q,Constant(0.0),no_inflow)

bcu = [bc1, bc2, bc3, bc4, bc5]


# --------- don't modify below this ---- #

# Galerkin variational problem

# Backward Euler
F = v*(u - u_n)*dx + dt*(mu*dot(grad(v), grad(u))*dx \
                        + v*dot(velocity, grad(u))*dx \
                        - f*v*dx)

# Residual and Stabilization Parameters
r = u - u_n + dt*(- mu*div(grad(u)) + dot(velocity, grad(u)) + sigma*u - f)
vnorm = sqrt(dot(velocity, velocity))
tau = h/(2.0*vnorm)

if method == 1: # SUPG
    F += (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx
    #F += tau*dot(velocity, grad(v))*r*dx
    methodname = "SUPG"
elif method == 2: # GLS
    F += tau*(- mu*div(grad(v)) + dot(velocity, grad(v)) + sigma*v)*r*dx
    methodname = "GLS"
elif method == 3: # DW
    F -= tau*(- mu*div(grad(v)) - dot(velocity, grad(v)) + sigma*v)*r*dx
    methodname = "DW"
elif method == 4: # VMS
    hh = 1.0/nx
    ttau = (4.0*mu/(hh*hh) + 2.0*vnorm/hh + sigma)**(-1)
    F -= (ttau)*(- mu*div(grad(v)) - dot(velocity, grad(v)) + sigma*v)*r*dx
    methodname = "VMS"
elif method == 6:   # EFR
    methodname = "EFR"
    #S = input("Set [S]cale of filtering radius delta (delta = S*mesh_size) ")
    S = 1
    delta = S*1.0/nx

    u_tilde = Function(Q)
    u_bar = Function(Q)
else:
    methodname = "Galerk"
    # Galerkin with no stabilization terms


if savetimes != 0:
	ffile=open(compfile,"a+")
	outputf = methodname+"\n"+"t,assemble,solve \n"
	ffile.write(outputf)

# Create bilinear and linear forms
a = lhs(F)
L = rhs(F)

# Assemble matrix
start1 = time.time()
A = assemble(a)
[bc.apply(A) for bc in bcu]
assemble_problem = time.time()-start1

# Create linear solver and factorize matrix

solver = LUSolver(A)
solver.parameters["reuse_factorization"] = True


# Output file
out_file = File(folder+"u_"+methodname+".pvd")
if method == 5:
    out_file_ue = File(folder+"u_exact.pvd")

# Set intial condition
u = u_n

# Create progress bar
progress = Progress('Time-stepping')
set_log_level(PROGRESS)

num_steps = int(round(T / dt, 0)) 


for n in range(num_steps):
	# Assemble vector and apply boundary conditions
	b = assemble(L)
	[bc.apply(b) for bc in bcu]

	# Solve the linear system (re-use the already factorized matrix A)
	start2 = time.time()
	solver.solve(u.vector(), b)
	solve_problem = time.time()-start2

	# Exact solution in time
	if method == 5:
	    ue = Expression(u_exact.cppcode, degree = R, t = t)
	    uee = interpolate(ue, Q)
	    uee.rename('Exact','Exact')
	    out_file_ue << (uee, float(t))
	    u_exact.t += dt

	# Copy solution from previous interval
	u_n = u

	# Save the solution to file
	# Save solution to file (VTK)
	#if n % saveTimesteps == 0:
	out_file << (u, t)

	# Move to next interval and adjust boundary condition
	t += dt

	# Update progress bar
	progress.update(t / T)
	if savetimes != 0:
		outputf = str(t)+','+str(assemble_problem)+','+ str(solve_problem)+'\n'

		# Computational Time Outputs
		ffile=open(compfile,"a+")
		ffile.write(outputf)