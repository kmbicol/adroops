from __future__ import print_function
from dolfin import *
from fenics import *
from copy import deepcopy
import numpy as np


code = 'Test'

# Parameters
if code == 'Test':

	# My test problem

	T = 1.0 # final time
	dt = 0.01 # time step size
	tt = 0.01
	num_steps = int(round(T / tt, 0))           

	sigma = 1.0       # reaction coefficient
	mu = 0.001        # diffision coefficient
	delta = 1.0/50.0  # filtering radius

	velocity = as_vector([1.0, 1.0]) # convection velocity


if code == 'Iliescu':

	# Iliescu Ex 4.1

	T = 2.0 # final time
	dt = 0.01 # time step size
	tt = 0.01
	num_steps = int(round(T / tt, 0))           

	sigma = 1.0       # reaction coefficient
	mu = 0.000001        # diffusion coefficient
	delta = 1.0/20.0  # filtering radius

	velocity = as_vector([2.0, 3.0]) # convection velocity

###########################################################

# Create mesh
mesh = UnitSquareMesh(20,20)

# Define function spaces
Q = FunctionSpace(mesh, "CG", 1)

# Define boundaries
def boundary(x, on_boundary):
    return on_boundary

# Define inflow profile ?

# Define boundary condition
u_D = Constant(0.0)
bc = DirichletBC(Q, u_D, boundary)

# Define trial and test functions (not computed yet)
u = TrialFunction(Q)
v = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(Q)
u_  = Function(Q)
u_tilde = Function(Q)
u_bar = Function(Q)

# Define expressions used in variational forms
t = 0

if code == 'Test':
	adr_f = Expression('3+sigma*(t+x[0]+x[1])', degree = 1, sigma = sigma, mu=mu, t = t)
if code == 'Iliescu':
	diff_u = Expression('-32*(1 - x[0])*x[0] *sin(pi*t) ((atan((2*(-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625))/mu^0.5))/pi + 1/2) - 32*(1 - x[1])*x[1]*sin(pi*t) ((atan((2*(-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625))/mu^0.5))/pi + 1/2) - (1024*(1 - x[0])*x[0]*(x[0] - 0.5)^2*(1 - x[1])*x[1]*sin(pi*t) (-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625))/(pi*mu^1.5 ((4*(-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625)^2)/mu^1. + 1)^2) - (1024*(1 - x[0])*x[0]*(1 - x[1]) (x[1] - 0.5)^2 x[1] *sin(pi*t) (-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625))/(pi*mu^1.5 ((4*(-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625)^2)/mu^1. + 1)^2) - (128 (1 - x[0]) (x[0] - 0.5) (1 - x[1])*x[1]*sin(pi*t))/(pi*mu^0.5 ((4*(-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625)^2)/mu^1. + 1)) + (128*x[0]*(x[0] - 0.5) (1 - x[1])*x[1]*sin(pi*t))/(pi*mu^0.5 ((4*(-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625)^2)/mu^1. + 1)) - (128 (1 - x[0])*x[0]*(1 - x[1]) (x[1] - 0.5) *sin(pi*t))/(pi*mu^0.5 ((4*(-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625)^2)/mu^1. + 1)) - (128 (1 - x[0])*x[0]*(1 - x[1])*x[1]*sin(pi*t))/(pi*mu^0.5 ((4*(-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625)^2)/mu^1. + 1)) + (128 (1 - x[0])*x[0]*(x[1] - 0.5)*x[1] *sin(pi*t))/(pi*mu^0.5 ((4*(-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625)^2)/mu^1. + 1))', degree = 1, sigma = sigma, mu=mu, t = t)
	conv_u = Expression('3*(16*(1 - x[0])*x[0]*(1 - x[1]) *sin(pi*t) ((atan((2*(-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625))/mu^0.5))/pi + 1/2) - 16*(1 - x[0])*x[0]*x[1] *sin(pi*t) ((atan((2*(-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625))/mu^0.5))/pi + 1/2) - (64*(1 - x[0])*x[0]*(1 - x[1]) (x[1] - 0.5)*x[1] *sin(pi*t))/(pi*mu^0.5 ((4*(-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625)^2)/mu^1. + 1))) + 2*(16*(1 - x[0]) (1 - x[1])*x[1]*sin(pi*t) ((atan((2*(-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625))/mu^0.5))/pi + 1/2) - 16*x[0]*(1 - x[1])*x[1]*sin(pi*t) ((atan((2*(-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625))/mu^0.5))/pi + 1/2) - (64*(1 - x[0]) (x[0] - 0.5)*x[0]*(1 - x[1])*x[1]*sin(pi*t))/(pi*mu^0.5 ((4*(-(x[0] - 0.5)^2 - (x[1] - 0.5)^2 + 0.0625)^2)/mu^1. + 1)))', degree = 1, sigma = sigma, mu=mu, t = t)
	rxn_u =Expression('16*sin(pi*t)*x[0]*(1-x[0])*x[1]*(1-x[1])*(1/2+arctan(2*mu^(-.5)*(0.25^2-(x[0]-0.5)^2-(x[1]-0.5)^2))/pi', degree = 1, sigma = sigma, mu=mu, t = t)
	adr_f = conv_u + diff_u + rxn_u


f_n = Expression(adr_f.cppcode, degree = 1, sigma = sigma, mu=mu, t = t) 
f = Expression(adr_f.cppcode, degree = 1, sigma = sigma, mu=mu, t = t+tt)

f_mid = 0.5*(f_n + f)
u_mid  = 0.5*(u_n + u)
dt   = Constant(dt)
mu = Constant(mu)

# Define indicator function to evaluate current time step
def a(u_tilde, u_):
    indicator = Expression('sqrt((a-b)*(a-b))', degree = 2, a = u_, b = u_tilde)
    indicator = interpolate(indicator, Q)
    max_ind = np.amax(indicator.vector().array())

    # Normalize indicator such that it's between [0,1].
    if max_ind < 1:
       max_ind = 1.0

    indicator = Expression('a/b', degree = 2, a = indicator, b = max_ind)
    indicator = interpolate(indicator, Q) 
    return indicator

# Define variational problem for step 1 (solve on coarse grid)
F1 = v*(u - u_n)*dx + dt*(mu*dot(grad(v), grad(u_mid))*dx \
                        + v*dot(velocity, grad(u_mid))*dx \
                        + sigma*v*u*dx \
                        - f_mid*v*dx)
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2a (apply Helmholz filter)
a2 = v*u*dx + delta*delta*dot(grad(v), grad(u))*dx #lhs(F_Hfilter)
L2 = v*u_*dx #rhs(F_Hfilter)

# Define variational problem for step 2b (evaluate indicator and find filtered solution)
def a3(ind):
	a3 = v*u*dx + delta*delta*dot(grad(v), ind*grad(u))*dx
	return a3
L3 = v*u_*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)


# Apply boundary conditions to matrices
bc.apply(A1)
bc.apply(A2)


# Create VTK files for visualization output
folder = "results_time_dep/"
out_file = File(folder+"u.pvd")              # no filter solution
out_file_utilde = File(folder+"utilde.pvd")  # u tilde
out_file_ind = File(folder+"a.pvd")          # indicator function
out_file_ubar = File(folder+"ubar.pvd")      # filtered solution
u_series = TimeSeries('velocity')

# Create progress bar
progress = Progress('Time-stepping')
set_log_level(PROGRESS)

# Time-stepping
for n in range(num_steps):
    # Update current time
    t += tt

    # Step 1
    b1 = assemble(L1)
    bc.apply(b1)
    solve(A1, u_.vector(), b1)

    # Step 2a
    b2 = assemble(L2)
    bc.apply(b2)
    solve(A2, u_tilde.vector(), b2)

    # Step 2b
    ind = a(u_tilde, u_)
    A3 = assemble(a3(ind))
    bc.apply(A3)
    b3 = assemble(L3)
    bc.apply(b3)
    solve(A3, u_bar.vector(), b3)    

    # Save solution to file (VTK)
    out_file << (u_, float(t))
    out_file_utilde << (u_tilde, float(t))
    out_file_ind << (ind, float(t))
    out_file_ubar << (u_bar, float(t))

    # Update previous solution and source term
    u_n.assign(u_bar)
    f_n = Expression(f.cppcode, degree = 1, sigma = sigma, t = t)
    f.t += tt
    f_n.t += tt

    # Update progress bar
    progress.update(t / T)
