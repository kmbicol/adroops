# Copyright (C) 2007 Kristian B. Oelgaard
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2008
# Modified by Johan Hake, 2008
# Modified by Garth N. Wells, 2009
#
# This demo solves the time-dependent convection-diffusion equation by
# a SUPG stabilized method. The velocity field used in the simulation
# is the output from the Stokes (Taylor-Hood) demo.  The sub domains
# for the different boundary conditions are computed by the demo
# program in src/demo/subdomains.

from dolfin import *
from fenics import *
import numpy as np
import math

# Automated multiple solutions
list_of_nx = [20, 40, 80]
list_of_N = [0,1,2,3]    # choices: 0,1,2,3
list_of_scale = [1 , np.sqrt(2), 2] # choices: 1, np.sqrt(2), 2
P = 1 # choices: 1 (means P1) , 2 (means P2)

'''


# One simulation at a time
list_of_nx = [50]
list_of_N = [3]    # choices: 0,1,2,3
list_of_scale = [2.0] # choices: 1, np.sqrt(2), 2
P = 1 # choices: 1 (means P1) , 2 (means P2)
'''
f=open("P"+str(P)+"_infnorm.txt","a+")
g=open("P"+str(P)+"_2norm.txt","a+")

f.write("L inf-norm & No Filter & $N=0$ & $N=1$ & $N=2$ & $N=3$ \\\\ \n \\hline \n")
g.write("L 2-norm & No Filter & $N=0$ & $N=1$ & $N=2$ & $N=3$ \\\\ \n \\hline \n")

for nx in list_of_nx:
	#f=open("P"+str(P)+"_data.txt","a+")
	#f.write("--------------------------------h=1/"+str(nx)+"\n")
	for scale in list_of_scale:	
			for N in list_of_N:	
				# Load mesh and subdomains
				ny = nx
				mesh = UnitSquareMesh(nx,ny) # divides [0,1]x[0,1] into 20x20 
				h = CellSize(mesh)
				nxx = float(nx)		
				delta = scale/nxx       # this is where we change the filtering radius
				
				# Create FunctionSpaces
				Q = FunctionSpace(mesh, "CG", P)

				# Initialise source function and previous solution function
				f  = Constant(1.0)

				# Boundary values
				u_D = Constant(0.0)

				# Parameters
				sigma = 0.01
				mu = 0.001
				velocity = as_vector([1.0, 1.0]) # this is b


				string0 = 'N = '+str(N)+' and h = 1/'+str(nx)
				#print string0
				# Test and trial functions
				u, v = TrialFunction(Q), TestFunction(Q)

				# Residual
				r = - mu*div(grad(u)) + dot(velocity, grad(u)) + sigma*u - f

				# Galerkin variational problem
				F = mu*dot(grad(v), grad(u))*dx + v*dot(velocity, grad(u))*dx + sigma*v*u*dx - f*v*dx

				# SUPG stabilisation terms
				vnorm = sqrt(dot(velocity, velocity))
				F_SUPG = F + (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx

				# GLS stabilization terms
				F_GLS = F + (h/(2.0*vnorm))*(dot(velocity, grad(v)) - mu*div(grad(v)) + sigma*v)*r*dx

				# Create bilinear and linear forms
				a = lhs(F)
				L = rhs(F)
				a_SUPG = lhs(F_SUPG)
				L_SUPG = rhs(F_SUPG)
				a_GLS = lhs(F_GLS)
				L_GLS = rhs(F_GLS)

				# Set up boundary condition
				def boundary(x, on_boundary):
				    return on_boundary
				bc = DirichletBC(Q, u_D, boundary)

				# Output file

				if scale == np.sqrt(2):
					scalename = 'sqrt2'
				else:
					scalename = scale

				folder ="P"+str(P)+"h1_"+str(nx)+"_delta"+str(scalename)+"h"

				before = "/N"+str(N)
				after = "_h1_"+str(nx)+"_delta"+str(scalename)+"h_"

				out_file_u = File(folder+"/u_"+"h_"+str(nx)+".pvd")
				out_file_usupg = File(folder+"/u_SUPG_"+"h_"+str(nx)+".pvd")
				out_file_ugls = File(folder+"/u_GLS_"+"h_"+str(nx)+".pvd")

				out_file_utilde = File(folder+"/u_tilde.pvd")
				out_file_ubar = File(folder+before+after+"u_bar.pvd")
				out_file_ind = File(folder+before+after+"a.pvd") #indicator

				# Assemble matrix
				A = assemble(a)
				bc.apply(A)

				A_SUPG = assemble(a_SUPG)
				bc.apply(A_SUPG)

				A_GLS = assemble(a_GLS)
				bc.apply(A_GLS)

				# Assemble vector and apply boundary conditions
				b = assemble(L)
				bc.apply(b)

				b_SUPG = assemble(L_SUPG)
				bc.apply(b_SUPG)

				b_GLS = assemble(L_GLS)
				bc.apply(b_GLS)

				# Create linear solver and factorize matrix
				solver = LUSolver(A)
				u = Function(Q)
				solver.solve(u.vector(), b)

				#print 'norm_u_L2  =',   norm(u, 'L2')

				solver = LUSolver(A_SUPG)
				u_SUPG = Function(Q)
				solver.solve(u_SUPG.vector(), b_SUPG)

				#print 'norm_u_SUPG_L2  =',   norm(u_SUPG, 'L2')

				solver = LUSolver(A_GLS)
				u_GLS = Function(Q)
				solver.solve(u_GLS.vector(), b_GLS)

				#print 'norm_u_GLS_L2  =',   norm(u_GLS, 'L2')

				# Compute difference between SUPG and GLS solution in L2 norm
				#diff_SUPG_GLS_L2 = norm(u_SUPG.vector() - u_GLS.vector(), 'L2')/norm(u_SUPG.vector(), 'L2')
				#print 'difference_SUPG_GLS_L2  =', diff_SUPG_GLS_L2 

				# Save the solution to file

				out_file_u << u

				out_file_usupg << u_SUPG

				out_file_ugls << u_GLS

				##################################################################################

				# Helmholtz filter to compute the indicator function
				u_tilde = TrialFunction(Q)
				u_1tilde = TrialFunction(Q)
				u_2tilde = TrialFunction(Q)
				u_3tilde = TrialFunction(Q)
				u_4tilde = TrialFunction(Q)
				#deltaH = h*vnorm/(2*mu)


				## ______________________________________________________________________ N=0
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
				out_file_utilde << u_1tilde

				## ______________________________________________________________________ N=1
				if N>0:
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
					DF = Expression('a+b-c',degree=2,a=DF,b=u_1tilde,c=u_2tilde)
					out_file_utilde << u_2tilde

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
						out_file_utilde << u_3tilde

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
							out_file_utilde << u_4tilde

				# Compute the indicator function

				indicator = Expression('sqrt((a-b)*(a-b))', degree = 2, a = u, b = DF)
				indicator = interpolate(indicator, Q)
				max_ind = np.amax(indicator.vector().array())

				# Normalize indicator such that it's between [0,1].
				if max_ind < 1:
				   max_ind = 1.0

				indicator = Expression('a/b', degree = 2, a = indicator, b = max_ind)
				indicator = interpolate(indicator, Q)

				out_file_ind << indicator


				# Apply the filter
				u_bar = TrialFunction(Q)
				F_filter = v*u_bar*dx + delta*delta*dot(grad(v), indicator*grad(u_bar))*dx - v*u*dx 

				a_filter = lhs(F_filter)
				L_filter = rhs(F_filter)

				A_filter = assemble(a_filter)
				bc.apply(A_filter)

				b_filter = assemble(L_filter)
				bc.apply(b_filter)

				solver = LUSolver(A_filter)
				u_bar = Function(Q)
				solver.solve(u_bar.vector(), b_filter)

				#print 'norm_u_bar_L2  =',   norm(u_bar, 'L2')

				##   ----- Calculate L-2 and L-inf error norms using SUPG as true solution
				# as seen in ft01_poisson.py

				if scalename == 'sqrt2':
					scalename = '\sqrt{2}'
				if scalename == 1:
					scalename = ' '
				if N==0:
					nofilter_Linf_err = np.abs(u_SUPG.vector().array() - u.vector().array()).max()
					filtered_Linf_err = np.abs(u_SUPG.vector().array() - u_bar.vector().array()).max()
					nofilter_L2_err = errornorm(u_SUPG, u, 'L2')
					filtered_L2_err = errornorm(u_SUPG,u_bar,'L2')
					firstcol = "$h=1/"+str(nx)+"$, "+"$\delta = "+str(scalename)+"h $"
					outputf = firstcol+"& "+str(round(nofilter_Linf_err,4))+" & "+str(round(filtered_Linf_err,4))
					outputg = firstcol+"& "+str(round(nofilter_L2_err,4))+" & "+str(round(filtered_L2_err,4))
				if N >0:				
					filtered_Linf_err = np.abs(u_SUPG.vector().array() - u_bar.vector().array()).max()
					filtered_L2_err = errornorm(u_SUPG,u_bar,'L2')	
					outputf = " & "+str(round(filtered_Linf_err,4))
					outputg = " & "+str(round(filtered_L2_err,4))
					if N==3:
						outputf = outputf+"  \\\\ \n \\hline \n"	
						outputg = outputg+"  \\\\ \n \\hline \n"
				f=open("P"+str(P)+"_infnorm.txt","a+")
				g=open("P"+str(P)+"_2norm.txt","a+")			
				f.write(outputf)
				g.write(outputg)

				out_file_ubar << u_bar

		
