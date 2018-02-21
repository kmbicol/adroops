from dolfin import *
import math as m
import numpy as np
import time

# Load mesh and subdomains

#method = input("0: No Filter, 1: SUPG, 2: EFR; method = ")

#method = 1
#if method == 2:
#	N = input("N = ")
#nx = 500#input("h = 1/nx, let nx = ")
method = 1

for dt in [0.01, 0.001]:
	for nx in [200,500]:
		# Output files directory
		#dt = 0.01

		T = 0.64
		t = dt - dt
		folder = "SUPG/mod_dt"+str(dt)+"h"+str(nx)+"_"
		#S = 1.0  # filtering radius factor
		P = 2    # polynomial degree of FE
		R = 2

		sigma = 1.0
		mu = 10**(-6)
		velocity = as_vector([2.0, 3.0])

		adr_f = Expression('(-5.09295817894065e-6*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*((4000.0*x[0] - 2000.0)*(8000.0*x[0] - 4000.0) + (4000.0*x[1] - 2000.0)*(8000.0*x[1] - 4000.0))*(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0)*sin(3.14159265358979*t) + pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)*(0.318309886183791*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5)*(-16.0*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) - 50.2654824574367*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*cos(3.14159265358979*t) - 48.0*x[0]*x[1]*(x[0] - 1)*sin(3.14159265358979*t) - 32.0*x[0]*x[1]*(x[1] - 1)*sin(3.14159265358979*t) - 48.0*x[0]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) + 3.2e-5*x[0]*(x[0] - 1)*sin(3.14159265358979*t) - 32.0*x[1]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) + 3.2e-5*x[1]*(x[1] - 1)*sin(3.14159265358979*t)) + (pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)*(-10.1859163578813*x[0]*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1) - 15.278874536822*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 0.0407436654315252*x[0]*x[1]*(x[0] - 1)*(x[1] - 1) + 1.01859163578813e-5*x[0]*x[1]*(x[0] - 1)*(4000.0*x[1] - 2000.0) + 1.01859163578813e-5*x[0]*x[1]*(4000.0*x[0] - 2000.0)*(x[1] - 1) + 1.01859163578813e-5*x[0]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 1.01859163578813e-5*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1))*sin(3.14159265358979*t))/pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)', degree = R, t = t)

		f = Expression(adr_f.cppcode, degree = R, t = t+dt)

		mesh = UnitSquareMesh(nx,nx)
		h = CellSize(mesh)
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

		if method == 0:
			methodname = "nofilter"
			out_file_ue = File(folder+"u_exact.pvd")
			u_exact = Expression(' 16*(-0.318309886183791*x[0]*x[1]*(-x[0] + 1)*(-x[1] + 1)*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) + 0.5*x[0]*x[1]*(-x[0] + 1)*(-x[1] + 1))*sin(3.14159265358979*t) ', degree=R, t=t)


		if method == 1:
			methodname = "SUPG"
			r = u - u_n + dt*(- mu*div(grad(u)) + dot(velocity, grad(u)) + sigma*u - f)
			vnorm = sqrt(dot(velocity, velocity))
			#F += (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx
			#F += (h*m.sqrt(2))*dot(velocity, grad(v))*r*dx
			F += dt*(h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx

		# Create bilinear and linear forms
		a1 = lhs(F)
		L = rhs(F)

		# Assemble matrices
		A1 = assemble(a1)
		bc.apply(A1)

		# Create progress bar
		progress = Progress('Time-stepping')
		set_log_level(PROGRESS)


		num_steps = int(round(T / dt, 0))+1

		# --- Time-stepping --- #


		if method == 1 or method == 0:
			out_file_ubar = File(folder+"u_"+methodname+".pvd")      # filtered solution
			out_file_ubar << (u_, 0.0)
			for n in range(num_steps):
				if method == 1:
					u_bar.rename('SUPG','SUPG')
				else:
					u_bar.rename('No Filter','No Filter')
				b = assemble(L)
				bc.apply(b)
				solve(A1, u_bar.vector(), b, "gmres")

				
				out_file_ubar << (u_bar, float(t+dt))

				# Update previous solution and source term
				u_n.assign(u_bar)
				progress.update(t / T)
				# Update current time
				t += dt
				f.t += dt

		print(methodname+" ")
