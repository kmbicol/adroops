from __future__ import print_function
from dolfin import *
from fenics import *
from copy import deepcopy
import numpy as np
import sympy as sym

code = 'Iliescu'
EFR = 1

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

    T = 2.0 #2.0 # final time
    dt = 0.01 # time step size
    tt = 0.01
    num_steps = int(round(T / tt, 0))           

    sigma = 1.0       # reaction coefficient
    mu = 0.000001        # diffusion coefficient
    delta = 1.0/20.0  # filtering radius

    velocity = as_vector([2.0, 3.0]) # convection velocity
    a = velocity[0]
    b = velocity[1]
###########################################################

# Create mesh
mesh = UnitSquareMesh(20,20)
h = CellSize(mesh)

# Define function spaces
Q = FunctionSpace(mesh, "CG", 1)

# Define boundaries
def boundary(x, on_boundary):
    return on_boundary

# Define inflow profile ?



# Define trial and test functions (not computed yet)
u = TrialFunction(Q)
v = TestFunction(Q)

# Define functions for solutions at previous, current time steps
u_n, u_ = Function(Q), Function(Q)
u_n_SUPG, u_SUPG_ = Function(Q), Function(Q)
u_n_GLS, u_GLS_ = Function(Q), Function(Q)
u_n_DW, u_DW_ = Function(Q), Function(Q)
u_tilde = Function(Q)
u_bar = Function(Q)

# Define expressions used in variational forms
t = 0

if code == 'Test':
    ue = Expression('t+x[0]+x[1]', degree = 1, sigma = sigma, mu = mu)
    adr_f = Expression('3+sigma*(t+x[0]+x[1])', degree = 1, sigma = sigma, mu=mu, t = t)
    uee = sym.printing.ccode(ue)
if code == 'Iliescu':
    ue = Expression('-16*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(0.318309886183791*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5)*sin(3.14159265358979*t)', degree = 1, t = t)
    adr_f = Expression('(-5.09295817894065e-6*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*((4000.0*x[0] - 2000.0)*(8000.0*x[0] - 4000.0) + (4000.0*x[1] - 2000.0)*(8000.0*x[1] - 4000.0))*(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) + pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)*(0.318309886183791*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5)*(-16.0*x[0]*x[1]*(x[0] - 1)*(x[1] - 1) - 48.0*x[0]*x[1]*(x[0] - 1) - 32.0*x[0]*x[1]*(x[1] - 1) - 48.0*x[0]*(x[0] - 1)*(x[1] - 1) + 3.2e-5*x[0]*(x[0] - 1) - 32.0*x[1]*(x[0] - 1)*(x[1] - 1) + 3.2e-5*x[1]*(x[1] - 1)) + (pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)*(-10.1859163578813*x[0]*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1) - 15.278874536822*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 0.0407436654315252*x[0]*x[1]*(x[0] - 1)*(x[1] - 1) + 1.01859163578813e-5*x[0]*x[1]*(x[0] - 1)*(4000.0*x[1] - 2000.0) + 1.01859163578813e-5*x[0]*x[1]*(4000.0*x[0] - 2000.0)*(x[1] - 1) + 1.01859163578813e-5*x[0]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 1.01859163578813e-5*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1)))*sin(3.14159265358979*t)/pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)', degree = 1, t=t)
out_file_ue = File('results_time_dep/uexact.pvd')

f_n = Expression(adr_f.cppcode, degree = 1, t = t) 
f = Expression(adr_f.cppcode, degree = 1, t = t+tt)

# Define boundary condition
u_D = Expression(ue.cppcode, degree = 1, t = t)
bc = DirichletBC(Q, u_D, boundary)

# Define for time stepping
f_mid = 0.5*(f_n + f)
u_mid  = 0.5*(u_n + u)
dt   = Constant(dt)
mu = Constant(mu)

# Create progress bar
progress = Progress('Time-stepping')
set_log_level(PROGRESS)

###########################################################

## Non-filtered Solution 

# Define variational problem for step 1 (solve on coarse grid)
F1 = v*(u - u_n)*dx + dt*(mu*dot(grad(v), grad(u_mid))*dx \
                        + v*dot(velocity, grad(u_mid))*dx \
                        + sigma*v*u*dx \
                        - f_mid*v*dx)
a1 = lhs(F1)
L1 = rhs(F1)

###########################################################

## Other Stabilization Methods
if EFR == 0:

    # Residual
    r = - mu*div(grad(u)) + dot(velocity, grad(u)) + sigma*u - f # Lu - f
    vnorm = sqrt(dot(velocity, velocity))

    # SUPG stabilisation terms
    F_SUPG = F1 + (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx
    a_SUPG = lhs(F_SUPG)
    L_SUPG = rhs(F_SUPG)

    # GLS stabilization terms
    F_GLS = F1 + (h/(2.0*vnorm))*(- mu*div(grad(v)) + dot(velocity, grad(v)) + sigma*v)*r*dx #LvRu
    a_GLS = lhs(F_GLS)
    L_GLS = rhs(F_GLS)

    # DW stabilization terms
    F_DW = F1 - (h/(2.0*vnorm))*(- mu*div(grad(v)) - dot(velocity, grad(v)) + sigma*v)*r*dx #LvRu
    a_DW = lhs(F_DW)
    L_DW = rhs(F_DW)

    # Assemble matrix
    A1 = assemble(a1)
    bc.apply(A1)

    A_SUPG = assemble(a_SUPG)
    bc.apply(A_SUPG)

    A_GLS = assemble(a_GLS)
    bc.apply(A_GLS)

    A_DW = assemble(a_DW)
    bc.apply(A_DW)

    # Create VTK files for visualization output
    folder = "results_time_dep/"
    out_file_nofilter = File(folder+"u_nofilter.pvd")              # no filter solution
    out_file_supg = File(folder+"u_SUPG.pvd")  # u tilde
    out_file_gls = File(folder+"u_GLS.pvd")          # indicator function
    out_file_dw = File(folder+"u_DW.pvd")      # filtered solution

    for n in range(num_steps):
        # Update current time
        t += tt

        # Assemble vector and apply boundary conditions
        b1 = assemble(L1)
        bc.apply(b1)
        solve(A1, u_.vector(), b1)

        b_SUPG = assemble(L_SUPG)
        bc.apply(b_SUPG)
        solve(A_SUPG, u_SUPG_.vector(), b_SUPG)

        b_GLS = assemble(L_GLS)
        bc.apply(b_GLS)
        solve(A_GLS, u_GLS_.vector(), b_GLS)

        b_DW = assemble(L_DW)
        bc.apply(b_DW)
        solve(A_DW, u_DW_.vector(), b_DW)

        # Save solution to file (VTK)
        out_file_nofilter << (u_, float(t))
        out_file_supg << (u_SUPG_, float(t))
        out_file_gls << (u_GLS_, float(t))
        out_file_dw << (u_DW_, float(t))

        # Update previous solution and source term
        u_n.assign(u_)
        u_n_SUPG.assign(u_SUPG_)
        u_n_GLS.assign(u_GLS_)
        u_n_DW.assign(u_DW_)

        f_n = Expression(f.cppcode, degree = 1, sigma = sigma, t = t)
        f.t += tt
        f_n.t += tt

        # Exact solution in time
        ue = Expression('-16*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(0.318309886183791*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5)*sin(3.14159265358979*t)', degree = 1, t = t)
        uee = interpolate(ue, Q)
        uee.rename('u','u')
        out_file_ue << (uee, float(t+tt))
        ue.t += tt

        # Update progress bar
        progress.update(t / T)

###########################################################

## EFR Method

if EFR == 1:

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

    t = 0
    ## EFR Stabilization Method

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
    folder = "results_time_dep_iliescu/"
    out_file = File(folder+"u.pvd")              # no filter solution
    out_file_utilde = File(folder+"utilde.pvd")  # u tilde
    out_file_ind = File(folder+"a.pvd")          # indicator function
    out_file_ubar = File(folder+"ubar.pvd")      # filtered solution
    u_series = TimeSeries('velocity')

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