from dolfin import *
from fenics import *
import numpy as np
import math as m

#test = input('1 f=t, 2 Iliescu, 3 Manufactured ')
test = 4
EFR = 0 #EFR = input('EFR: (1) Yes (0) No ') #EFR = 0
nx = input('h=1/nx, set nx as ') #nx = 100
delta = 1.0/20.0  # filtering radius

P = 1
R = 2 # degree of all expressions

# Create mesh
mesh = UnitSquareMesh(nx,nx)
h = CellSize(mesh)

# Define function spaces
Q = FunctionSpace(mesh, "CG", P)
t = 0

# Simulation Parameters
if test == 1:
    # My test problem
    T = 0.01 # final time
    dt = 0.01 # time step size
    tt = 0.01
    num_steps = int(round(T / tt, 0))           

    sigma = 0.01
    mu = 0.001
    velocity = as_vector([1.0, 1.0]) # convection velocity
    u_D = Constant(0.0)
    #adr_f = Expression('t', degree = R, t=t)
    adr_f = Expression('1.0', degree = R, t=t)

    folder = 'results_time_dep_simp'
    folder +="/P"+str(P)+"h1_"+str(nx)+"/"

elif test == 2 or test == 4:
    # Iliescu Ex 4.1
    T = 2.0 #2.0 # final time
    dt = 0.01 # time step size
    tt = 0.01
    num_steps = int(round(T / tt, 0))           

    sigma = 1.0       # reaction coefficient
    mu = 10**(-6)        # diffusion coefficient

    velocity = as_vector([2.0, 3.0]) # convection velocity
    a = velocity[0]
    b = velocity[1]
    folder = 'results_time_dep_ilie'
    folder +="/P"+str(P)+"h1_"+str(nx)+"/"

elif test == 3:
    # My test problem
    T = 0.01 # final time
    dt = 0.01 # time step size
    tt = 0.01
    num_steps = int(round(T / tt, 0))           

    sigma = 0.01
    mu = 0.001
    velocity = as_vector([1.0, 1.0]) # convection velocity
    u_exact = Expression('t+x[0]+x[1]', degree = R, t = t)
    adr_f = Expression('0.01*t + 0.01*x[0] + 0.01*x[1] + 3.0', degree = R, t = t)
    u_D = Expression(u_exact.cppcode, degree = R, t = t)

    folder = 'results_time_dep_manu'
    folder +="/P"+str(P)+"h1_"+str(nx)+"/"

else:
    test = input('Choose 1, 2, or 3...')
###########################################################

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
u_n_VMS, u_VMS_ = Function(Q), Function(Q)
u_tilde = Function(Q)
u_bar = Function(Q)

# Define expressions used in variational forms
if test == 2: # magnitude 1
    u_D = Constant(0.0)
    folder += "1_"
    u_exact = Expression('-x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(0.318309886183791*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5)*sin(3.14159265358979*t)', degree = R, t = t)
    adr_f = Expression('(-3.18309886183791e-7*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*((4000.0*x[0] - 2000.0)*(8000.0*x[0] - 4000.0) + (4000.0*x[1] - 2000.0)*(8000.0*x[1] - 4000.0))*(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0)*sin(3.14159265358979*t) + pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)*(0.318309886183791*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5)*(-1.0*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) - 3.14159265358979*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*cos(3.14159265358979*t) - 3.0*x[0]*x[1]*(x[0] - 1)*sin(3.14159265358979*t) - 2.0*x[0]*x[1]*(x[1] - 1)*sin(3.14159265358979*t) - 3.0*x[0]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) + 2.0e-6*x[0]*(x[0] - 1)*sin(3.14159265358979*t) - 2.0*x[1]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) + 2.0e-6*x[1]*(x[1] - 1)*sin(3.14159265358979*t)) + (pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)*(-0.636619772367581*x[0]*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1) - 0.954929658551372*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 0.00254647908947033*x[0]*x[1]*(x[0] - 1)*(x[1] - 1) + 6.36619772367581e-7*x[0]*x[1]*(x[0] - 1)*(4000.0*x[1] - 2000.0) + 6.36619772367581e-7*x[0]*x[1]*(4000.0*x[0] - 2000.0)*(x[1] - 1) + 6.36619772367581e-7*x[0]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 6.36619772367581e-7*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1))*sin(3.14159265358979*t))/pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)', degree = R, t=t)

if test == 4: # magnitude 16
    folder += "16_"
    u_D = Constant(0.0)
    u_exact = Expression('-16*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(0.318309886183791*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5)*sin(3.14159265358979*t)', degree = R, t = t)
    adr_f = Expression('(-5.09295817894065e-6*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*((4000.0*x[0] - 2000.0)*(8000.0*x[0] - 4000.0) + (4000.0*x[1] - 2000.0)*(8000.0*x[1] - 4000.0))*(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0)*sin(3.14159265358979*t) + pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)*(0.318309886183791*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5)*(-16.0*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) - 50.2654824574367*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*cos(3.14159265358979*t) - 48.0*x[0]*x[1]*(x[0] - 1)*sin(3.14159265358979*t) - 32.0*x[0]*x[1]*(x[1] - 1)*sin(3.14159265358979*t) - 48.0*x[0]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) + 3.2e-5*x[0]*(x[0] - 1)*sin(3.14159265358979*t) - 32.0*x[1]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) + 3.2e-5*x[1]*(x[1] - 1)*sin(3.14159265358979*t)) + (pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)*(-10.1859163578813*x[0]*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1) - 15.278874536822*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 0.0407436654315252*x[0]*x[1]*(x[0] - 1)*(x[1] - 1) + 1.01859163578813e-5*x[0]*x[1]*(x[0] - 1)*(4000.0*x[1] - 2000.0) + 1.01859163578813e-5*x[0]*x[1]*(4000.0*x[0] - 2000.0)*(x[1] - 1) + 1.01859163578813e-5*x[0]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 1.01859163578813e-5*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1))*sin(3.14159265358979*t))/pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)', degree = R, t = t)

f_n = Expression(adr_f.cppcode, degree = R, t = t) 
f = Expression(adr_f.cppcode, degree = R, t = t+tt)

# Define boundary condition
bc = DirichletBC(Q, u_D, boundary)


# Define initial condition

u_n = interpolate(u_D, Q)
u_n_VMS = interpolate(u_D, Q)
u_n_DW = interpolate(u_D, Q)
u_n_GLS = interpolate(u_D, Q)
u_n_SUPG = interpolate(u_D, Q)

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
# Crank-Nicolson
'''
F1 = v*(u - u_n)*dx + dt*(mu*dot(grad(v), grad(u_mid))*dx \
                        + v*dot(velocity, grad(u_mid))*dx \
                        + sigma*v*u*dx \
                        - f_mid*v*dx)
'''
# Backward Euler

F1 = v*(u - u_n)*dx + dt*(mu*dot(grad(v), grad(u))*dx \
                        + v*dot(velocity, grad(u))*dx \
                        + sigma*v*u*dx \
                        - f*v*dx)


a1 = lhs(F1)
L1 = rhs(F1)

###########################################################


## Other Stabilization Methods
if EFR == 0:

    # Residual
    r = u - u_n + dt*(- mu*div(grad(u)) + dot(velocity, grad(u)) + sigma*u - f) # Lu - f
    vnorm = sqrt(dot(velocity, velocity))

    # tau Stabilization Parameter

    tau = h/(2.0*vnorm)

    # SUPG stabilisation terms
    F_SUPG = F1 + tau*dot(velocity, grad(v))*r*dx
#    F_SUPG = F1 + tau*(0.5*div(velocity*v)+0.5*dot(velocity, grad(v)))*r*dx
    a_SUPG = lhs(F_SUPG)
    L_SUPG = rhs(F_SUPG)

    # GLS stabilization terms
    F_GLS = F1 + tau*(- mu*div(grad(v)) + dot(velocity, grad(v)) + sigma*v)*r*dx #LvRu
    a_GLS = lhs(F_GLS)
    L_GLS = rhs(F_GLS)

    # DW stabilization terms
    F_DW = F1 - tau*(- mu*div(grad(v)) - dot(velocity, grad(v)) + sigma*v)*r*dx #LvRu
    a_DW = lhs(F_DW)
    L_DW = rhs(F_DW)

    # VMS stabilization terms
    hh = 1.0/nx
    ttau = m.pow((4.0*mu/(hh*hh) + 2.0*vnorm/hh + sigma),-1)
    F_VMS = F1 - (ttau)*(- mu*div(grad(v)) - dot(velocity, grad(v)) + sigma*v)*r*dx #LvRu
    a_VMS = lhs(F_VMS)
    L_VMS = rhs(F_VMS)

    # Assemble matrix
    A1 = assemble(a1)
    bc.apply(A1)

    A_SUPG = assemble(a_SUPG)
    bc.apply(A_SUPG)

    A_GLS = assemble(a_GLS)
    bc.apply(A_GLS)

    A_DW = assemble(a_DW)
    bc.apply(A_DW)

    A_VMS = assemble(a_VMS)
    bc.apply(A_VMS)

    # Create VTK files for visualization output
    out_file_nofilter = File(folder+"u_nofilter_h1_"+str(nx)+"t.pvd")              # no filter solution
    out_file_supg = File(folder+"u_SUPG_h1_"+str(nx)+"t.pvd")  # u tilde
    out_file_gls = File(folder+"u_GLS_h1_"+str(nx)+"t.pvd")          # indicator function
    out_file_ue = File(folder+"u_exact_h1_"+str(nx)+"t.pvd")
    out_file_fe = File(folder+"f_exact_h1_"+str(nx)+"t.pvd")
    out_file_dw = File(folder+"u_DW_h1_"+str(nx)+"t.pvd")      
    out_file_vms = File(folder+"u_VMS_h1_"+str(nx)+"t.pvd")      # filtered solution

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

        b_VMS = assemble(L_VMS)
        bc.apply(b_VMS)
        solve(A_VMS, u_VMS_.vector(), b_VMS)

        # Save solution to file (VTK)
        out_file_nofilter << (u_, float(t))
        out_file_supg << (u_SUPG_, float(t))
        out_file_gls << (u_GLS_, float(t))
        out_file_dw << (u_DW_, float(t))
        out_file_vms << (u_VMS_, float(t))

        # Update previous solution and source term
        u_n.assign(u_)
        u_n_SUPG.assign(u_SUPG_)
        u_n_GLS.assign(u_GLS_)
        u_n_DW.assign(u_DW_)
        u_n_VMS.assign(u_VMS_)

        f_n = Expression(f.cppcode, degree = R, sigma = sigma, t = t)
        fee = interpolate(f_n, Q)
        fee.rename('f','f')
        out_file_fe << (fee, float(t+tt))
        f.t += tt
        f_n.t += tt

        # Exact solution in time
        ue = Expression(u_exact.cppcode, degree = R, t = t)
        uee = interpolate(ue, Q)
        uee.rename('u','u')
        out_file_ue << (uee, float(t+tt))
        u_exact.t += tt

        # Update progress bar
        progress.update(t / T)

###########################################################

## EFR Method

if EFR == 1:

    # Define indicator function to evaluate current time step
    def a(u_tilde, u_, t):
        indicator = Expression('sqrt((a-b)*(a-b))', degree = 2, a = u_, b = u_tilde)
        indicator = interpolate(indicator, Q)
        max_ind = np.amax(indicator.vector().array())

        # Normalize indicator such that it's between [0,1].
        if max_ind < 1:
           max_ind = 1.0

        indicator = Expression('a/b', degree = 2, a = indicator, b = max_ind)
        indicator = interpolate(indicator, Q) 
        indicator.rename('a','a')
        out_file_ind << (indicator, float(t))
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
    out_file = File(folder+"u_nofilter.pvd")              # no filter solution
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
        ind = a(u_tilde, u_, t)
        A3 = assemble(a3(ind))
        bc.apply(A3)
        b3 = assemble(L3)
        bc.apply(b3)
        solve(A3, u_bar.vector(), b3)    

        # Save solution to file (VTK)
        out_file << (u_, float(t))
        out_file_utilde << (u_tilde, float(t))
        out_file_ubar << (u_bar, float(t))

        # Update previous solution and source term
        u_n.assign(u_bar)
        f_n = Expression(f.cppcode, degree = R, sigma = sigma, t = t)
        f.t += tt
        f_n.t += tt

        # Update progress bar
        progress.update(t / T)