from dolfin import *
import math as m
import numpy as np
import csv

methodname = "EFR"
S = 1.0 #scale such that delta = S*h
P = 2    # polynomial degree of FE
R = 2


dt = 0.001
T = 0.1


N = 0
lvl = 1

sigma = 1.0
mu = 10**(-6)
velocity = as_vector([2.0,3.0])

myfile = 'errorEFR403b.csv'

for nx in [50,100,200,400,800]:
	delta = 1.0#S*1.0/nx
	print(delta)
	t = 0.0
	folder = "040218/dt"+str(dt)+"h"+str(nx)+"_"

	u_exact = Expression('sin(DOLFIN_PI*x[0])*cos(DOLFIN_PI*x[1])', degree = R)
	#u_ = Expression('2*pow(DOLFIN_PI, 2)*pow(delta, 2)*sin(DOLFIN_PI*x[0])*cos(DOLFIN_PI*x[1]) + sin(DOLFIN_PI*x[0])*cos(DOLFIN_PI*x[1])', degree = R, delta = delta)

	u_ = Expression('sin(DOLFIN_PI*x[0])*cos(DOLFIN_PI*x[1]) + 2*pow(DOLFIN_PI, 2)*sin(DOLFIN_PI*x[0])*cos(DOLFIN_PI*x[1])', degree=R)
	u_D = Expression(u_exact.cppcode, degree = R)

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

	# def L2(u_): # input is intermediate velocity OR previous u_tilde solution
		# L2 = v*u_*dx
		# return L2

	# Define variational problem for step 2b (evaluate indicator and find filtered solution)
	def a3(ind):
		a3 = v*u*dx + delta*delta*dot(grad(v), ind*grad(u))*dx
		return a3


	#L3 = v*u_*dx

	# --- End EFR --- #

	num_steps = int(round(T / dt, 0)) 

	# Create bilinear and linear forms
	# a1 = lhs(F)
	# L = rhs(F)

	# Assemble matrices
	# A1 = assemble(a1)
	# bc.apply(A1)

	# Create progress bar
	progress = Progress('Time-stepping')
	set_log_level(PROGRESS)


	# --- Time-stepping --- #

	out_file_ind = File(folder+"a_N"+str(N)+"_EFR.pvd")          # indicator function
	out_file_ubar = File(folder+"ubar_N"+str(N)+"_EFR.pvd")      # filtered solution

	# for n in range(num_steps):
		# Step 1 Solve on Coarse Grid

		# b = assemble(L)
		# bc.apply(b)
		# solve(A1, u_.vector(), b, "gmres")

		# Step 2a Solve Helmholtz filter
		# N=0
	#b2_0 = assemble(L2(u_))
	b2_0 = assemble(L2)
	bc_0 = DirichletBC(Q, u_D, boundary)
	bc_0.apply(b2_0)
	bc_0.apply(A2)
	solve(A2, u_tilde0.vector(), b2_0, "gmres")

	#DF = Expression('a', degree = R, a=u_tilde0)

		# Step 2b Calculate Indicator and solve Ind Problem
		# ind = a(DF, u_, float(t))

		# A3 = assemble(a3(ind))
		# bc.apply(A3)

		# b3 = assemble(L3)
		# bc.apply(b3)

		# solve(A3, u_bar.vector(), b3, "gmres")

		# progress.update(t / T)

		# Update previous solution and source term
		# u_n.assign(u_bar)
		# Update current time
		# t += dt
		# f.t += dt

	# out_file_ind << (ind, float(t))
	out_file_ubar << (u_tilde0, float(t))

	L2 = errornorm(u_exact, u_tilde0, norm_type='L2', degree_rise=3)
	H1_0 = errornorm(u_exact, u_tilde0, norm_type='H10', degree_rise=3)
	H1 = np.sqrt(L2**2 + H1_0**2)

	ffile = open(myfile,"a+")
	if near(lvl, 1):
		outputf1 = methodname+' N = '+str(N)+', t = '+str(t)+"\n"
		ffile.write(outputf1)

	outputf = '\nLevel '+str(lvl)+', nx = '+str(nx)+'\n'
	outputf += 'L2,' + str(L2) + 'H1,' + str(H1) +'\n \n'
	ffile.write(outputf)

	lvl += 1