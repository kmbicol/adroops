from dolfin import *
from fenics import *
import numpy as np
import sympy as sym
import math as m

#print('h = 1/nx')
#print('Choose test: 1 f=t, 2 Iliescu, 3 Manufactured')
#nx, test = input('Set nx and choose test (Ex: 20,1) ')
nx = input('h=1/nx, set nx : ')
#nx = 20
test = 2

P = 1

# Create mesh
mesh = UnitSquareMesh(nx,nx) # divides [0,1]x[0,1] into 20x20 
h = CellSize(mesh)

# Define function spaces
Q = FunctionSpace(mesh, "CG", P)

# Simulation Parameters
if test == 1:
	# Simple Problem Parameters
	sigma = 0.01
	mu = 0.001
	u_D = Constant(0.0) #Constant(1.0/sigma)
	f = Constant(1.0)
	velocity = as_vector([1.0, 1.0]) #uniform in space velocity
	folder = "results_steady_simp"
	folder +="/P"+str(P)+"h1_"+str(nx)

elif test == 2:
	# Iliescu Problem Parameters (t=0.5)
	sigma = 1.0
	mu = 10**(-6)
	u_D = Constant(0.0)
	f = Expression('(-3.18309886183791e-7*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*((4000.0*x[0] - 2000.0)*(8000.0*x[0] - 4000.0) + (4000.0*x[1] - 2000.0)*(8000.0*x[1] - 4000.0))*(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) + pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)*(0.318309886183791*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5)*(-1.0*x[0]*x[1]*(x[0] - 1)*(x[1] - 1) - 3.0*x[0]*x[1]*(x[0] - 1) - 2.0*x[0]*x[1]*(x[1] - 1) - 3.0*x[0]*(x[0] - 1)*(x[1] - 1) + 2.0e-6*x[0]*(x[0] - 1) - 2.0*x[1]*(x[0] - 1)*(x[1] - 1) + 2.0e-6*x[1]*(x[1] - 1)) + (pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)*(-0.636619772367581*x[0]*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1) - 0.954929658551372*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 0.00254647908947033*x[0]*x[1]*(x[0] - 1)*(x[1] - 1) + 6.36619772367581e-7*x[0]*x[1]*(x[0] - 1)*(4000.0*x[1] - 2000.0) + 6.36619772367581e-7*x[0]*x[1]*(4000.0*x[0] - 2000.0)*(x[1] - 1) + 6.36619772367581e-7*x[0]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 6.36619772367581e-7*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1)))/pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)', degree = 1)
	u_exact = Expression('x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(-0.318309886183791*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) + 0.5)', degree = 1)

	velocity = as_vector([2.0, 3.0]) #uniform in space velocity
	folder = "results_steady_ilie_"
	folder +="/P"+str(P)+"h1_"+str(nx)

	ue = Expression(u_exact.cppcode, degree=1)
	uee = interpolate(ue, Q)

	fe = Expression(f.cppcode, degree=1)
	fee = project(fe, Q)

	out_file_ue = File(folder+"/u_exact_h"+str(nx)+".pvd")
	out_file_ue << uee
	out_file_fe = File(folder+"/f_exact_h"+str(nx)+".pvd")
	out_file_fe << fee

elif test == 3: 
	## Manufactured solution
	sigma = 0.01
	mu = 0.000001
	velocity = as_vector([1.0, 1.0]) #uniform in space velocity
	b1 = velocity[0]
	b2 = velocity[1]
	
	folder = "manu_"

	x, y = sym.symbols('x[0], x[1]')

	u_exact = x*x
	u_exact = sym.simplify(u_exact)

	f_exact = -mu*(sym.diff(sym.diff(u_exact, x), x) + sym.diff(sym.diff(u_exact, y), y)) + b1*sym.diff(u_exact, x) + b2*sym.diff(u_exact, y) + sigma*u_exact
	f_exact = sym.simplify(f_exact)

	u_code = sym.printing.ccode(u_exact)
	f_code = sym.printing.ccode(f_exact)

	print('u =', u_code)
	print('f =', f_code)

	ue = Expression(u_code, degree=1)
	uee = interpolate(ue, Q)

	fe = Expression(f_code, degree=1)
	fee = project(fe, Q)

	folder +="results_steady/P"+str(P)+"h1_"+str(nx)
	out_file_ue = File(folder+"/u_exact_h"+str(nx)+".pvd")
	out_file_ue << uee
	out_file_fe = File(folder+"/f_exact_h"+str(nx)+".pvd")
	out_file_fe << fee


	f = Expression(f_code, degree=1)
	u_D = Expression(u_code, degree=1)

else:
	print('Select either 1 (Simple) or 2 (Iliescu) or 3 (Manufactured)')



##################################################################################

# Define boundaries
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(Q, u_D, boundary)

# Define trial and test functions (not computed yet)
u = TrialFunction(Q)
v = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(Q)
u_  = Function(Q)
u_tilde = Function(Q)
u_bar = Function(Q)

##################################################################################

## Define expressions used in variational forms

## Galerkin variational problem
# No time dependence
F = mu*dot(grad(v), grad(u))*dx + v*dot(velocity, grad(u))*dx + sigma*v*u*dx - f*v*dx

# First time step (time dependent code)
#F = mu*dot(grad(v), grad(u))*dx + v*dot(velocity, grad(u))*dx + (sigma + 1.0/dt)*v*u*dx - f*v*dx 
a = lhs(F)
L = rhs(F)

# Residual
r = - mu*div(grad(u)) + dot(velocity, grad(u)) + sigma*u - f # Lu - f
vnorm = sqrt(dot(velocity, velocity))

# SUPG stabilisation terms
# Skew-Symmetric for non uniform velocity b
ttau = h/(2.0*vnorm)
F_SUPG = F + ttau*(0.5*div(velocity*v)+0.5*dot(velocity, grad(v)))*r*dx 
# SS for uniform vel
#F_SUPG = F + (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx 
a_SUPG = lhs(F_SUPG)
L_SUPG = rhs(F_SUPG)

# GLS stabilization terms
F_GLS = F + (h/(2.0*vnorm))*(- mu*div(grad(v)) + dot(velocity, grad(v)) + sigma*v)*r*dx #LvRu
a_GLS = lhs(F_GLS)
L_GLS = rhs(F_GLS)

# DW stabilization terms
F_DW = F - (h/(2.0*vnorm))*(- mu*div(grad(v)) - dot(velocity, grad(v)) + sigma*v)*r*dx #LvRu
a_DW = lhs(F_DW)
L_DW = rhs(F_DW)

# DW stabilization terms
hh = 1.0/nx
tau = m.pow((4.0*mu/(hh*hh) + 2.0*vnorm/hh + sigma),-1)
F_VMS = F - (tau)*(- mu*div(grad(v)) - dot(velocity, grad(v)) + sigma*v)*r*dx #LvRu
a_VMS = lhs(F_VMS)
L_VMS = rhs(F_VMS)

# Assemble matrix
A = assemble(a)
bc.apply(A)

A_SUPG = assemble(a_SUPG)
bc.apply(A_SUPG)

A_GLS = assemble(a_GLS)
bc.apply(A_GLS)

A_DW = assemble(a_DW)
bc.apply(A_DW)

A_VMS = assemble(a_VMS)
bc.apply(A_VMS)

# Assemble vector and apply boundary conditions
b = assemble(L)
bc.apply(b)

b_SUPG = assemble(L_SUPG)
bc.apply(b_SUPG)

b_GLS = assemble(L_GLS)
bc.apply(b_GLS)

b_DW = assemble(L_DW)
bc.apply(b_DW)

b_VMS = assemble(L_VMS)
bc.apply(b_VMS)

# Create linear solver and factorize matrix
solver = LUSolver(A)
u = Function(Q)
solver.solve(u.vector(), b)

solver_SUPG = LUSolver(A_SUPG)
u_SUPG = Function(Q)
solver_SUPG.solve(u_SUPG.vector(), b_SUPG)

solver_GLS = LUSolver(A_GLS)
u_GLS = Function(Q)
solver_GLS.solve(u_GLS.vector(), b_GLS)

solver_DW = LUSolver(A_DW)
u_DW = Function(Q)
solver_DW.solve(u_DW.vector(), b_DW)

solver_VMS = LUSolver(A_VMS)
u_VMS = Function(Q)
solver_VMS.solve(u_VMS.vector(), b_VMS)

##################################################################################

# File Output
out_file_u = File(folder+"/u_nofilter_"+"h_"+str(nx)+".pvd")
out_file_usupg = File(folder+"/u_SUPG_"+"h_"+str(nx)+".pvd")
out_file_ugls = File(folder+"/u_GLS_"+"h_"+str(nx)+".pvd")
out_file_udw = File(folder+"/u_DW_"+"h_"+str(nx)+".pvd")
out_file_uvms = File(folder+"/u_VMS_"+"h_"+str(nx)+".pvd")

out_file_u << u
out_file_usupg << u_SUPG
out_file_ugls << u_GLS
out_file_udw << u_DW
out_file_uvms << u_VMS