from dolfin import *
import math as m
import numpy as np
import csv

methodname = "EFR"
S = 1.0 #scale such that delta = S*h
P = 2    # polynomial degree of FE
R = 2

dt = 0.001
T = 0.5


N = 0
lvl = 1

sigma = 1.0
mu = 10**(-6)
velocity = as_vector([2.0,3.0])

myfile = 'errorEFR404.csv'

f = Expression('-1.6e-5*x[0]*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(8000.0*x[0] - 4000.0)*(x[1] - 1)*(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0)*sin(DOLFIN_PI*t)/(DOLFIN_PI*pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)) - 32.0*x[0]*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1)*sin(DOLFIN_PI*t)/(DOLFIN_PI*(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)) - 1.6e-5*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0)*(8000.0*x[1] - 4000.0)*(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0)*sin(DOLFIN_PI*t)/(DOLFIN_PI*pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)) - 48.0*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0)*sin(DOLFIN_PI*t)/(DOLFIN_PI*(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)) - 16.0*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*sin(DOLFIN_PI*t)/DOLFIN_PI - 16.0*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*cos(DOLFIN_PI*t) + 0.128*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*sin(DOLFIN_PI*t)/(DOLFIN_PI*(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)) + 3.2e-5*x[0]*x[1]*(x[0] - 1)*(4000.0*x[1] - 2000.0)*sin(DOLFIN_PI*t)/(DOLFIN_PI*(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)) - 48.0*x[0]*x[1]*(x[0] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*sin(DOLFIN_PI*t)/DOLFIN_PI + 3.2e-5*x[0]*x[1]*(4000.0*x[0] - 2000.0)*(x[1] - 1)*sin(DOLFIN_PI*t)/(DOLFIN_PI*(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)) - 32.0*x[0]*x[1]*(x[1] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*sin(DOLFIN_PI*t)/DOLFIN_PI + 3.2e-5*x[0]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0)*sin(DOLFIN_PI*t)/(DOLFIN_PI*(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)) - 48.0*x[0]*(x[0] - 1)*(x[1] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*sin(DOLFIN_PI*t)/DOLFIN_PI + 3.2e-5*x[0]*(x[0] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*sin(DOLFIN_PI*t)/DOLFIN_PI + 3.2e-5*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1)*sin(DOLFIN_PI*t)/(DOLFIN_PI*(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)) - 32.0*x[1]*(x[0] - 1)*(x[1] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*sin(DOLFIN_PI*t)/DOLFIN_PI + 3.2e-5*x[1]*(x[1] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*sin(DOLFIN_PI*t)/DOLFIN_PI', degree = R, t = t)



for nx in [50,100,200,400,800]:

	delta = 0.1 #S*1.0/nx
	print(delta)
	t = 0.0
	f.t = t
	f.t += dt
	print(f.t)

	folder = "040418/dt"+str(dt)+"h"+str(nx)+"_"

	u_exact = Expression(' 16*(-0.318309886183791*x[0]*x[1]*(-x[0] + 1)*(-x[1] + 1)*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) + 0.5*x[0]*x[1]*(-x[0] + 1)*(-x[1] + 1))*sin(3.14159265358979*t) ', degree=R, t=t)

	u_D = Expression(u_exact.cppcode, degree = R, t=t)

	mesh = UnitSquareMesh(nx,nx)
	h = CellDiameter(mesh)
	Q = FunctionSpace(mesh, "CG", P)


	# Set up boundary condition
	def boundary(x, on_boundary):
		return on_boundary
	bc = DirichletBC(Q, u_D, boundary)


	# Don't Modify Below This! -----------#

	# Test and trial functions
	u, v = TrialFunction(Q), TestFunction(Q)

	u_tilde0 = Function(Q)


	# Define variational problem for step 2a (apply Helmholz filter)
	# Note: only RHS changes, so we can keep the same a2 throughout

	a2 = v*u*dx + delta*delta*dot(grad(v), grad(u))*dx #lhs(F_Hfilter)
	A2 = assemble(a2)

	# Define indicator function to evaluate current time step
	def a(u_tilde, u_, t):
		indicator = Expression('sqrt((a-b)*(a-b))', degree = 2, a = u_, b = u_tilde)
		indicator = interpolate(indicator, Q)
		max_ind = np.amax(indicator.vector().get_local())#.vector().array())

		# Normalize indicator such that it's between [0,1].
		if max_ind < 1:
		   max_ind = 1.0

		indicator = Expression('a/b', degree = 2, a = indicator, b = max_ind)
		indicator = interpolate(indicator, Q) 
		indicator.rename('a','a')
		#out_file_ind << (indicator, float(t))
		return indicator
	L2 = v*u_*dx

	def L2(u_): # input is intermediate velocity OR previous u_tilde solution
		L2 = v*u_*dx
		return L2

	# Define variational problem for step 2b (evaluate indicator and find filtered solution)
	def a3(ind):
		a3 = v*u*dx + delta*delta*dot(grad(v), ind*grad(u))*dx
		return a3


	L3 = v*u_*dx

	# --- End EFR --- #

	num_steps = int(round(T / dt, 0)) 

	# Create bilinear and linear forms
	a1 = lhs(F)
	L = rhs(F)

	# Assemble matrices
	A1 = assemble(a1)
	bc.apply(A1)

	# Create progress bar
	progress = Progress('Time-stepping')
	set_log_level(PROGRESS)


	# --- Time-stepping --- #

	out_file_ind = File(folder+"a_N"+str(N)+"_EFR.pvd")          # indicator function
	out_file_ubar = File(folder+"ubar_N"+str(N)+"_EFR.pvd")      # filtered solution

	i=0
	for n in range(num_steps):
		# Step 1 Solve on Coarse Grid

		b = assemble(L)
		bc.apply(b)
		solve(A1, u_.vector(), b, "gmres")
		print(str(i)+'a')

		# Step 2a Solve Helmholtz filter
		# N=0
		b2_0 = assemble(L2(u_))
		bc_0 = DirichletBC(Q, u_D, boundary)
		bc_0.apply(b2_0)
		bc_0.apply(A2)
		solve(A2, u_tilde0.vector(), b2_0, "cg")
		print(str(i)+'b')
		DF = Expression('a', degree = R, a=u_tilde0)

		# Step 2b Calculate Indicator and solve Ind Problem
		ind = a(DF, u_, float(t))

		A3 = assemble(a3(ind))
		bc.apply(A3)

		b3 = assemble(L2(u_))
		bc.apply(b3)

		solve(A3, u_bar.vector(), b3, "gmres")
		print(str(i)+'c')

		progress.update(t / T)

		# Update previous solution and source term
		u_n.assign(u_bar)
		Update current time
		t += dt
		f.t += dt

	# out_file_ind << (ind, float(t))
	# out_file_ubar << (u_tilde0, float(t))

	L2 = errornorm(u_exact, u_bar, norm_type='L2', degree_rise=3)
	H1_0 = errornorm(u_exact, u_bar, norm_type='H10', degree_rise=3)
	H1 = np.sqrt(L2**2 + H1_0**2)

	ffile = open(myfile,"a+")
	if near(lvl, 1):
		outputf1 = methodname+' N = '+str(N)+', t = '+str(t)+"\n"
		ffile.write(outputf1)

	outputf = '\nLevel '+str(lvl)+', nx = '+str(nx)+', delta = '+str(delta)+'\n'
	outputf += 'L2,' + str(L2) + 'H1,' + str(H1) +'\n \n'
	ffile.write(outputf)

	lvl += 1