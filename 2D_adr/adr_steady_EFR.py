'''
Goal of this code is to write EFR in a object-oriented way
or at least use functions to make the code more readable!
'''

from dolfin import *
from fenics import *
import numpy as np
import sympy as sym
import math as m

#print('h = 1/nx')
#print('Choose test: 1 f=t, 2 Iliescu, 3 Manufactured')
#nx, test = input('Set nx and choose test (Ex: 20,1) ')
#nx = input('h=1/nx, set nx : ')
nx = 20
test = 2
scalename = 1
P = 1
delta = 1/20.0

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
	# this iliescu has c = 1 (not 16sin(pi*t))
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

else:
	test = input('1 or 2')

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
a = lhs(F)
L = rhs(F)

# Assemble matrix
A = assemble(a)
bc.apply(A)

# Assemble vector and apply boundary conditions
b = assemble(L)
bc.apply(b)

# Create linear solver and factorize matrix
solver = LUSolver(A)
u = Function(Q)
solver.solve(u.vector(), b)

##################################################################################

## EFR Algorithm

def a(u_, DF):
	# Compute the indicator function
	indicator = Expression('sqrt((a-b)*(a-b))', degree = 2, a = u_, b = DF)
	indicator = interpolate(indicator, Q)
	max_ind = np.amax(indicator.vector().array())

	# Normalize indicator such that it's between [0,1].
	if max_ind < 1:
	   max_ind = 1.0

	indicator = Expression('a/b', degree = 2, a = indicator, b = max_ind)
	indicator = interpolate(indicator, Q)
	return indicator

def filter(u, delta): 	# Recall: u_tilde is the solution on the coarse mesh
	# Evaluate u_tilde to see if it requires filtering
	u_tilde = TrialFunction(Q)
	v = TestFunction(Q)

	#F_Hfilter = u_tilde*v*dx + delta*delta*dot(grad(u_tilde), grad(v))*dx - v*u*dx
	a_Hfilter = u_tilde*v*dx + delta*delta*dot(grad(u_tilde), grad(v))*dx #lhs(F_Hfilter)
	L_Hfilter = v*u*dx #rhs(F_Hfilter)

	A_Hfilter = assemble(a_Hfilter)
	bc.apply(A_Hfilter)

	b_Hfilter = assemble(L_Hfilter)
	bc.apply(b_Hfilter)

	solver_filter = LUSolver(A_Hfilter)
	u_tilde = Function(Q)
	solver_filter.solve(u_tilde.vector(), b_Hfilter)
	return u_tilde

##################################################################################

for N in range(2):

	before = "/N"+str(N)
	after = "_h1_"+str(nx)+"_delta"+str(scalename)+"h_"

	out_file_utilde = File(folder+"/u_tilde.pvd")
	out_file_ind1 = File(folder+before+after+"a0.pvd")
	out_file_ind2 = File(folder+before+after+"a1.pvd")
	out_file_ind3 = File(folder+before+after+"a2.pvd")
	out_file_ind4 = File(folder+before+after+"a3.pvd")

##################################################################################

	# Helmholtz filter to compute the indicator function

	u_1tilde = TrialFunction(Q)
	u_2tilde = TrialFunction(Q)
	u_3tilde = TrialFunction(Q)
	u_4tilde = TrialFunction(Q)
	#deltaH = h*vnorm/(2*mu)

	out_file_u1tilde = File(folder+"/u1_tilde.pvd")
	out_file_u2tilde = File(folder+"/u2_tilde.pvd")
	out_file_u3tilde = File(folder+"/u3_tilde.pvd")
	out_file_u4tilde = File(folder+"/u4_tilde.pvd")




	## ______________________________________________________________________ N=0
	'''
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
	'''
	DF = filter(u, delta)
	if N == 0:
		u_1tilde = filter(u, delta)
		out_file_u1tilde << u_1tilde
		ind1 = a(u, DF)
		out_file_ind1 << ind1
		ind = ind1

	## ______________________________________________________________________ N=1
	if N>0:
		'''
		F_Hfilter1 = v*u_2tilde*dx + delta*delta*dot(grad(v), grad(u_2tilde))*dx - v*u_1tilde*dx

		a_Hfilter1 = lhs(F_Hfilter1)
		L_Hfilter1 = rhs(F_Hfilter1)

		A_Hfilter1 = assemble(a_Hfilter1)
		bc.apply(A_Hfilter1)

		b_Hfilter1 = assemble(L_Hfilter1)
		bc.apply(b_Hfilter1)

		solver1 = LUSolver(A_Hfilter1)
		u_2tilde = Function(Q)
		solver1.solve(u_2tilde.vector(), b_Hfilter1)
		'''
		u_2tilde = filter(u_1tilde, delta)
		
		#DF = Expression('a+b-c',degree=2,a=DF,b=u_1tilde,c=u_2tilde)
		if N==1:
			out_file_u2tilde << u_2tilde
		#	ind2 = a(u, DF)
		#	out_file_ind2 << ind2
		#	ind = ind2


	## ______________________________________________________________________ N=2
		if N>1:
			F_Hfilter2 = v*u_3tilde*dx + delta*delta*dot(grad(v), grad(u_3tilde))*dx - v*u_2tilde*dx

			a_Hfilter2 = lhs(F_Hfilter2)
			L_Hfilter2 = rhs(F_Hfilter2)

			A_Hfilter2 = assemble(a_Hfilter2)
			bc.apply(A_Hfilter2)

			b_Hfilter2 = assemble(L_Hfilter2)
			bc.apply(b_Hfilter2)

			solver2 = LUSolver(A_Hfilter2)
			u_3tilde = Function(Q)
			solver2.solve(u_3tilde.vector(), b_Hfilter2)
			DF = Expression('a+b-2*c+d',degree=2,a=DF,b=u_1tilde,c=u_2tilde,d=u_3tilde)
			if N==2:
				out_file_u3tilde << u_3tilde
				ind3 = a(u_tilde, DF)
				out_file_ind3 << ind3
				ind = ind3

	## ______________________________________________________________________ N=3
			if N>2:
				F_Hfilter3 = v*u_4tilde*dx + delta*delta*dot(grad(v), grad(u_4tilde))*dx - v*u_3tilde*dx

				a_Hfilter3 = lhs(F_Hfilter3)
				L_Hfilter3 = rhs(F_Hfilter3)

				A_Hfilter3 = assemble(a_Hfilter3)
				bc.apply(A_Hfilter3)

				b_Hfilter3 = assemble(L_Hfilter3)
				bc.apply(b_Hfilter3)

				solver3 = LUSolver(A_Hfilter3)
				u_4tilde = Function(Q)
				solver3.solve(u_4tilde.vector(), b_Hfilter3)
				DF = Expression('a+b-3*c+3*d-e',degree=2,a=DF,b=u_1tilde,c=u_2tilde,d=u_3tilde,e=u_4tilde)
				if N == 3:
					out_file_u4tilde << u_4tilde
					ind4 = a(u_tilde, DF)
					out_file_ind4 << ind4
					ind = ind4