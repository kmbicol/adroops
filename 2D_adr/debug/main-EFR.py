from dolfin import *
import math as m
import numpy as np
import csv

methodname = "EFR"
S = 1.0 #scale such that delta = S*h

P = 2    # polynomial degree of FE
R = 2

sigma = 1.0
mu = 10**(-6)
velocity = as_vector([2.0, 3.0])
dt = 0.001
T = 0.50
lvl = 'n'



def compute_errors(u_e, u):
    """Compute various measures of the error u - u_e, where
    u is a finite element Function and u_e is an Expression."""

    # L2 norm
    E5 = errornorm(u_e, u, norm_type='L2', degree_rise=3)

    # H1 seminorm
    E6 = errornorm(u_e, u, norm_type='H10', degree_rise=3)

    # Collect error measures in a dictionary with self-explanatory keys
    errors = {'L2 norm': E5,
              'H10 seminorm': E6}

    return errors

for nx in [100]: 
#for dt in [0.01, 0.005, 0.0025]:
	for N in [0]:#[0,1,2,3]:

		t = 0.0
		folder = "main-EFR/L"+str(lvl)+"dt"+str(dt)+"h"+str(nx)+"_"

		f = Expression('-1.6e-5*x[0]*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(8000.0*x[0] - 4000.0)*(x[1] - 1)*(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0)*sin(DOLFIN_PI*t)/(DOLFIN_PI*pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)) - 32.0*x[0]*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1)*sin(DOLFIN_PI*t)/(DOLFIN_PI*(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)) - 1.6e-5*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0)*(8000.0*x[1] - 4000.0)*(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0)*sin(DOLFIN_PI*t)/(DOLFIN_PI*pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)) - 48.0*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0)*sin(DOLFIN_PI*t)/(DOLFIN_PI*(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)) - 16.0*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*sin(DOLFIN_PI*t)/DOLFIN_PI - 16.0*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*cos(DOLFIN_PI*t) + 0.128*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*sin(DOLFIN_PI*t)/(DOLFIN_PI*(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)) + 3.2e-5*x[0]*x[1]*(x[0] - 1)*(4000.0*x[1] - 2000.0)*sin(DOLFIN_PI*t)/(DOLFIN_PI*(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)) - 48.0*x[0]*x[1]*(x[0] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*sin(DOLFIN_PI*t)/DOLFIN_PI + 3.2e-5*x[0]*x[1]*(4000.0*x[0] - 2000.0)*(x[1] - 1)*sin(DOLFIN_PI*t)/(DOLFIN_PI*(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)) - 32.0*x[0]*x[1]*(x[1] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*sin(DOLFIN_PI*t)/DOLFIN_PI + 3.2e-5*x[0]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0)*sin(DOLFIN_PI*t)/(DOLFIN_PI*(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)) - 48.0*x[0]*(x[0] - 1)*(x[1] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*sin(DOLFIN_PI*t)/DOLFIN_PI + 3.2e-5*x[0]*(x[0] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*sin(DOLFIN_PI*t)/DOLFIN_PI + 3.2e-5*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1)*sin(DOLFIN_PI*t)/(DOLFIN_PI*(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)) - 32.0*x[1]*(x[0] - 1)*(x[1] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*sin(DOLFIN_PI*t)/DOLFIN_PI + 3.2e-5*x[1]*(x[1] - 1)*(atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5*DOLFIN_PI)*sin(DOLFIN_PI*t)/DOLFIN_PI', degree = R, t = t)

		f.t +=dt

		mesh = UnitSquareMesh(nx,nx)
		h = CellDiameter(mesh)
		Q = FunctionSpace(mesh, "CG", P)

		u_n = Function(Q)
		u_D = Constant(0.0)


		# Set up boundary condition
		def boundary(x, on_boundary):
			return on_boundary
		bc = DirichletBC(Q, u_D, boundary)


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

		# --- Begin EFR --- #
		methodname = "EFR"
		delta = S*1.0/nx

		u_tilde0 = Function(Q)
		u_tilde1 = Function(Q)
		u_tilde2 = Function(Q)
		u_tilde3 = Function(Q)	



		# Define indicator function to evaluate current time step
		def a(u_tilde, u_, t):
			indicator = Expression('sqrt((a-b)*(a-b))', degree = 2, a = u_, b = u_tilde)
			indicator = interpolate(indicator, Q)
			max_ind = np.amax(indicator.vector().get_local())

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
		
		folder += "S"+str(S)

		out_file_ind = File(folder+"a_N"+str(N)+"_EFR.pvd")          # indicator function
		out_file_ubar = File(folder+"ubar_N"+str(N)+"_EFR.pvd")      # filtered solution


		out_file_ubar << (u_bar, float(t))

		for n in range(num_steps):
			u_bar.rename('N='+str(N),'N='+str(N))
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
			elif N == 2:
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

				b2_2 = assemble(L2(u_tilde1))
				bc_2 = DirichletBC(Q, u_tilde1, boundary)
				bc_2.apply(b2_2)
				bc_2.apply(A2)
				solve(A2, u_tilde2.vector(), b2_2, "cg")
				DF = Expression('3*a-3*b+c', degree = R, a=u_tilde0, b=u_tilde1, c=u_tilde2)
			if N == 3: # N == 3:
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

				b2_2 = assemble(L2(u_tilde1))
				bc_2 = DirichletBC(Q, u_tilde1, boundary)
				bc_2.apply(b2_2)
				bc_2.apply(A2)
				solve(A2, u_tilde2.vector(), b2_2, "cg")

				b2_3 = assemble(L2(u_tilde2))
				bc_3 = DirichletBC(Q, u_tilde2, boundary)
				bc_3.apply(b2_3)
				bc_3.apply(A2)
				solve(A2, u_tilde3.vector(), b2_3, "cg")
				DF = Expression('4*a-6*b+4*c-d', degree = R, a=u_tilde0, b=u_tilde1, c=u_tilde2, d=u_tilde3)
			else: # N=0
				b2_0 = assemble(L2(u_))
				bc_0 = DirichletBC(Q, u_, boundary)
				bc_0.apply(b2_0)
				bc_0.apply(A2)
				solve(A2, u_tilde0.vector(), b2_0, "cg")
				DF = Expression('a', degree = R, a=u_tilde0)

			# Step 2b Calculate Indicator and solve Ind Problem
			ind = a(DF, u_, float(t))

			A3 = assemble(a3(ind))
			bc.apply(A3)

			b3 = assemble(L3)
			bc.apply(b3)

			solve(A3, u_bar.vector(), b3, "gmres")    

			progress.update(t / T)
			if n % 20 == 0:
				out_file_ind << (ind, float(t))
				out_file_ubar << (u_bar, float(t))

			# Update previous solution and source term
			u_n.assign(u_bar)

			# Update current time
			t += dt
			f.t += dt

		out_file_ubar << (u_bar, float(t))
		u_exact = Expression(' 16*(-0.318309886183791*x[0]*x[1]*(-x[0] + 1)*(-x[1] + 1)*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) + 0.5*x[0]*x[1]*(-x[0] + 1)*(-x[1] + 1))*sin(3.14159265358979*t) ', degree=R, t=t)
		errors = compute_errors(u_exact,u_bar)
		'''
		myfile = 'errorEFR.csv'

		ffile = open(myfile,"a+")
		if near(lvl, 0):
			outputf1 = methodname+' N = '+str(N)+', t = '+str(t)+"\n"
			ffile.write(outputf1)

		outputf = '\nLevel '+str(lvl)+'\n'
		ffile.write(outputf)

		w = csv.writer(open(myfile, "a+"))
		for key, val in errors.items():
			w.writerow([key, val])

	print("Done with Level"+str(lvl))
	
	lvl += 1'''