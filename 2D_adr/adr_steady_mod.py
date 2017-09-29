from dolfin import *
from fenics import *
import numpy as np
import sympy as sym
# Parameters

'''
# Automated multiple solutions
list_of_nx = [20]#, 40, 80, 200]
list_of_N = [0,1,2,3]    # choices: 0,1,2,3
list_of_scale = [1, np.sqrt(2), 2] # choices: 1, np.sqrt(2), 2
P=input('P = ')#1 # choices: 1 (means P1) , 2 (means P2)


for nx in list_of_nx:
	for scale in list_of_scale:	
			for N in list_of_N:	
				# Load mesh and subdomains
'''



## Single Solution Code
# Simulation Parameters
dt = 0.01
nx = 20
P = 1

# Filter Parameters
scale = 1.0
delta = scale/nx
# N = [0, 1, 2, 3]

# Problem Parameters
sigma = 0.01
mu = 0.001

# Initialize source function and previous solution function

## File Output Details

scalename = 1
folder ="P"+str(P)+"h1_"+str(nx)+"_delta"+str(scalename)+"h"


##################################################################################

# Create mesh
ny = nx
mesh = UnitSquareMesh(nx,ny) # divides [0,1]x[0,1] into 20x20 
h = CellSize(mesh)

# Define function spaces
Q = FunctionSpace(mesh, "CG", P)

# Define boundaries
def boundary(x, on_boundary):
    return on_boundary

# Define trial and test functions (not computed yet)
u = TrialFunction(Q)
v = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(Q)
u_  = Function(Q)
u_tilde = Function(Q)
u_bar = Function(Q)

##################################################################################

'''
## Transport Velocity Options
as_vector([1.0, 1.0])			# constant
('x[0]','x[1]') 				# increasing in (1,1) direction
('x[0]-0.5', 'x[1]-0.5') 		# increasing out of center of unit square
('x[0]-0.5', '-(x[1]-0.5)')     # increasing with areas of 0 velocity
('x[1]-.5', '-(x[0]-.5)') 		# clockwise rotation
('-(x[1]-.5)', 'x[0]-.5')  		# counterclockwise rotation
('x[1]-x[0]', '-x[0]-x[1]')		# spiral inward
'''

#velocity = as_vector([1.0, 1.0])
velocity = Expression(('x[1]-x[0]', '-x[0]-x[1]') , degree=1)
# make sure to change a,b in source function code

##################################################################################
'''
## Source Function Options
x*x + y*y + t
x + y
x + y + t
'''

x, y = sym.symbols('x[0], x[1]')

u_exact = x*x*x
u_exact = sym.simplify(u_exact)

b1 = -y
b2 = x

fpy = -mu*(sym.diff(sym.diff(u_exact, x), x) + sym.diff(sym.diff(u_exact, y), y)) + b1*sym.diff(u_exact, x) + b2*sym.diff(u_exact, y) + sigma*u_exact
fpy = sym.simplify(fpy)

u_code = sym.printing.ccode(u_exact)
f_code = sym.printing.ccode(fpy)

print('u =', u_code)
print('f =', f_code)

ue = Expression(u_code, degree=1)
uee = interpolate(ue, Q)

fe = Expression(f_code, degree=1)
fee = project(fe, Q)

out_file_ue = File(folder+"/u_exact_h"+str(nx)+".pvd")
out_file_ue << uee
out_file_fe = File(folder+"/f_exact_h"+str(nx)+".pvd")
out_file_fe << fee


f = Expression(f_code, degree=1)
u_D = Expression(u_code, degree=1)
bc = DirichletBC(Q, u_D, boundary)

##################################################################################


## Define expressions used in variational forms

# Galerkin variational problem
F = mu*dot(grad(v), grad(u))*dx + v*dot(velocity, grad(u))*dx + sigma*v*u*dx - f*v*dx
a = lhs(F)
L = rhs(F)

# Residual
r = - mu*div(grad(u)) + dot(velocity, grad(u)) + sigma*u - f # Lu - f
vnorm = sqrt(dot(velocity, velocity))

# SUPG stabilisation terms
F_SUPG = F + (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx
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

# Assemble matrix
A = assemble(a)
bc.apply(A)

A_SUPG = assemble(a_SUPG)
bc.apply(A_SUPG)

A_GLS = assemble(a_GLS)
bc.apply(A_GLS)

A_DW = assemble(a_DW)
bc.apply(A_DW)

# Assemble vector and apply boundary conditions
b = assemble(L)
bc.apply(b)

b_SUPG = assemble(L_SUPG)
bc.apply(b_SUPG)

b_GLS = assemble(L_GLS)
bc.apply(b_GLS)

b_DW = assemble(L_DW)
bc.apply(b_DW)

# Create linear solver and factorize matrix
solver = LUSolver(A)
u = Function(Q)
solver.solve(u.vector(), b)

solver = LUSolver(A_SUPG)
u_SUPG = Function(Q)
solver.solve(u_SUPG.vector(), b_SUPG)

solver = LUSolver(A_GLS)
u_GLS = Function(Q)
solver.solve(u_GLS.vector(), b_GLS)

solver = LUSolver(A_DW)
u_DW = Function(Q)
solver.solve(u_DW.vector(), b_DW)

## Compute difference between SUPG and GLS solution in L2 norm
# diff_SUPG_GLS_L2 = norm(u_DW.vector() - u_GLS.vector(), 'L2')/norm(u_DW.vector(), 'L2')
# print 'difference_SUPG_GLS_L2  =', diff_SUPG_GLS_L2 



##################################################################################

# File Output
out_file_u = File(folder+"/u_nofilter_"+"h_"+str(nx)+".pvd")
out_file_usupg = File(folder+"/u_SUPG_"+"h_"+str(nx)+".pvd")
out_file_ugls = File(folder+"/u_GLS_"+"h_"+str(nx)+".pvd")
out_file_udw = File(folder+"/u_DW_"+"h_"+str(nx)+".pvd")



out_file_u << u
out_file_usupg << u_SUPG
out_file_ugls << u_GLS
out_file_udw << u_DW


h=open("P"+str(P)+"_infnorm.txt","a+")
g=open("P"+str(P)+"_2norm.txt","a+")

h.write("L inf-norm & No Filter & $N=0$ & $N=1$ & $N=2$ & $N=3$ \\\\ \n \\hline \n")
g.write("L 2-norm & No Filter & $N=0$ & $N=1$ & $N=2$ & $N=3$ \\\\ \n \\hline \n")

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

for N in range(4):

	before = "/N"+str(N)
	after = "_h1_"+str(nx)+"_delta"+str(scalename)+"h_"

	out_file_utilde = File(folder+"/u_tilde.pvd")
	out_file_ind1 = File(folder+before+after+"a0.pvd")
	out_file_ind2 = File(folder+before+after+"a1.pvd")
	out_file_ind3 = File(folder+before+after+"a2.pvd")
	out_file_ind4 = File(folder+before+after+"a3.pvd")

	string0 = 'N = '+str(N)+' and h = 1/'+str(nx)

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

	if N == 0:
		out_file_u1tilde << u_1tilde
		ind1 = a(u_tilde, DF)
		out_file_ind1 << ind1
		ind = ind1

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
		if N==1:
			out_file_u2tilde << u_2tilde
			ind2 = a(u_tilde, DF)
			out_file_ind2 << ind2
			ind = ind2


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


	for SU in [0, 1]:

		# Apply the filter
		u_bar = TrialFunction(Q)
		if SU == 1:
			F_filter = v*u_bar*dx + delta*delta*dot(dot(velocity,grad(v)), ind*dot(velocity, grad(u_bar)))*dx - v*u*dx 
			out_file_ubar = File(folder+before+after+"u_bar_SU.pvd")
		else:
			F_filter = v*u_bar*dx + delta*delta*dot(grad(v), ind*grad(u_bar))*dx - v*u*dx 
			out_file_ubar = File(folder+before+after+"u_bar.pvd")

		a_filter = lhs(F_filter)
		L_filter = rhs(F_filter)

		A_filter = assemble(a_filter)
		bc.apply(A_filter)

		b_filter = assemble(L_filter)
		bc.apply(b_filter)

		solver = LUSolver(A_filter)
		u_bar = Function(Q)
		solver.solve(u_bar.vector(), b_filter)
		'''
		# Save Output Files
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
		'''

		out_file_ubar << u_bar