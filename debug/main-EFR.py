# copied from ipython notebook EFR


from dolfin import *
import math as m
import numpy as np
import csv

myfile = '0409_nofilter_P2.csv'

def compute_errors(u_e, u, t, mesh):
	L2n = errornorm(u_e, u, norm_type='L2', degree_rise=3, mesh=mesh)
	H1n = errornorm(u_e, u, norm_type='H1', degree_rise=3, mesh=mesh)
	errors = {'L2 norm': L2n, 'H1 norm': H1n}
	return 'L2, ' + str(L2n) +', H1, '+ str(H1n) +', t, '+ str(t) +'\n'

def EFR(nx):
	delta = 'none'
	P = 2
	R = P
	sigma = 1.0
	mu = 10**(-6)
	velocity = as_vector([2.0, 3.0])
	dt = 0.01
	T = 0.05

	t=0.0

	u_exact = Expression('x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*sin(DOLFIN_PI*t)', degree = P+2, t=t)
	f = Expression('1.0*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*sin(DOLFIN_PI*t) + DOLFIN_PI*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*cos(DOLFIN_PI*t) + 3.0*x[0]*x[1]*(x[0] - 1)*sin(DOLFIN_PI*t) + 2.0*x[0]*x[1]*(x[1] - 1)*sin(DOLFIN_PI*t) + 3.0*x[0]*(x[0] - 1)*(x[1] - 1)*sin(DOLFIN_PI*t) - 2.0e-6*x[0]*(x[0] - 1)*sin(DOLFIN_PI*t) + 2.0*x[1]*(x[0] - 1)*(x[1] - 1)*sin(DOLFIN_PI*t) - 2.0e-6*x[1]*(x[1] - 1)*sin(DOLFIN_PI*t)', degree = P, t= dt)

	mesh = UnitSquareMesh(nx,nx)
	h = CellDiameter(mesh)
	Q = FunctionSpace(mesh, "CG", P)

	# Set up boundary condition
	u_D = Expression(u_exact.cppcode, degree = R, t=t)
	def boundary(x, on_boundary):
		return on_boundary
	bc = DirichletBC(Q, u_D, boundary)

	# Test and trial functions
	u, v = TrialFunction(Q), TestFunction(Q)
	u_n = Function(Q)
	u_ = Function(Q)

	# Galerkin variational problem
	# Backward Euler
	F = v*(u - u_n)*dx + dt*(mu*dot(grad(v), grad(u))*dx \
							+ v*dot(velocity, grad(u))*dx \
							+ sigma*v*u*dx \
							- f*v*dx)

	# Create bilinear and linear forms
	a1 = lhs(F)
	L = rhs(F)

	# Assemble matrices
	A1 = assemble(a1)
	bc.apply(A1)

	# Create progress bar
	progress = Progress('Time-stepping')
	set_log_level(PROGRESS)
	
	ffile = open(myfile,"a+")

	while t - T < DOLFIN_EPS:
		# Step 1 Solve on Coarse Grid
		b = assemble(L)
		bc.apply(b)
		solve(A1, u_.vector(), b, "gmres")

		u_n = u_

		progress.update(t / T)
		# Update current time
		t += dt
		f.t += dt
		u_D.t += dt

	out_file_ubar = File("results/nofilter_u_"+str(nx)+".pvd") 
	out_file_ubar << (u_, float(t))

	errors = compute_errors(u_exact,u_,t, mesh)
	print(errors)
	ffile.write('t = '+ str(t-dt)+ ', nx = '+ str(nx)+'\n')
	ffile.write(errors)

	print('End of nx '+str(nx)+"------------------------------xxx")

EFR(50)
EFR(100)
EFR(200)
EFR(400)
EFR(800)
