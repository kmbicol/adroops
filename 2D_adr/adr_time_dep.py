from dolfin import *
import math as m
import numpy as np

#print("\n (1) SUPG \n (2) GLS \n (3) DW \n (4) VMS \n (5) Galerkin and Exact Solution \n (6) EFR \n (default) Galerkin")
#method = input("Choose a stabilization method: ")
#nx = input("h = 1/nx, let nx = ")

# Simulation Parameters
method = 6
nx = 300
T = 0.520 #2.0
dt = 0.001
t = dt
sigma = 1.0
mu = 10**(-6)
R = 2
P = 2
saveTimesteps = 5 # save every __ time steps
folder = "results_full_ilie_BE/h"+str(nx)

# Create velocity Function from file
velocity = as_vector([2.0, 3.0])

# Load mesh and subdomains
mesh = UnitSquareMesh(nx,nx)
h = CellSize(mesh)

# Create FunctionSpaces
Q = FunctionSpace(mesh, "CG", P)


# Boundary values
u_D = Constant(0.0)

# Initialise source function and previous solution function

#f  = Constant(1.0)
u_n = Function(Q)

# Test and trial functions
u, v = TrialFunction(Q), TestFunction(Q)
u_ = Function(Q)

u_exact = Expression('-16*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(0.318309886183791*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5)*sin(3.14159265358979*t)', degree = R, t = t)
adr_f = Expression('(-5.09295817894065e-6*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*((4000.0*x[0] - 2000.0)*(8000.0*x[0] - 4000.0) + (4000.0*x[1] - 2000.0)*(8000.0*x[1] - 4000.0))*(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0)*sin(3.14159265358979*t) + pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)*(0.318309886183791*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5)*(-16.0*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) - 50.2654824574367*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*cos(3.14159265358979*t) - 48.0*x[0]*x[1]*(x[0] - 1)*sin(3.14159265358979*t) - 32.0*x[0]*x[1]*(x[1] - 1)*sin(3.14159265358979*t) - 48.0*x[0]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) + 3.2e-5*x[0]*(x[0] - 1)*sin(3.14159265358979*t) - 32.0*x[1]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) + 3.2e-5*x[1]*(x[1] - 1)*sin(3.14159265358979*t)) + (pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)*(-10.1859163578813*x[0]*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1) - 15.278874536822*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 0.0407436654315252*x[0]*x[1]*(x[0] - 1)*(x[1] - 1) + 1.01859163578813e-5*x[0]*x[1]*(x[0] - 1)*(4000.0*x[1] - 2000.0) + 1.01859163578813e-5*x[0]*x[1]*(4000.0*x[0] - 2000.0)*(x[1] - 1) + 1.01859163578813e-5*x[0]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 1.01859163578813e-5*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1))*sin(3.14159265358979*t))/pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)', degree = R, t = t)

f = Expression(adr_f.cppcode, degree = R, t = t)
f0 = Expression(adr_f.cppcode, degree = R, t = 0)
'''
# Mid-point solution
u_mid = 0.5*(u_n + u)
f_mid = 0.5*(f0 + f)


# Residual
r = u - u_n + dt*(- mu*div(grad(u_mid)) + dot(velocity, grad(u_mid)) + sigma*u_mid - f_mid)


# Galerkin variational problem
F = v*(u - u_n)*dx + dt*(mu*dot(grad(v), grad(u_mid))*dx \
                        + v*dot(velocity, grad(u_mid))*dx \
                        + sigma*v*u*dx \
                        - f_mid*v*dx)
'''




# Set up boundary condition
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(Q, u_D, boundary)


# Galerkin variational problem

# Backward Euler
F = v*(u - u_n)*dx + dt*(mu*dot(grad(v), grad(u))*dx \
                        + v*dot(velocity, grad(u))*dx \
                        + sigma*v*u*dx \
                        - f*v*dx)

# Residual and Stabilization Parameters
r = u - u_n + dt*(- mu*div(grad(u)) + dot(velocity, grad(u)) + sigma*u - f)
vnorm = sqrt(dot(velocity, velocity))
tau = h/(2.0*vnorm)

if method == 1: # SUPG
    F += (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx
    #F += tau*dot(velocity, grad(v))*r*dx
    methodname = "SUPG"
elif method == 2: # GLS
    F += tau*(- mu*div(grad(v)) + dot(velocity, grad(v)) + sigma*v)*r*dx
    methodname = "GLS"
elif method == 3: # DW
    F -= tau*(- mu*div(grad(v)) - dot(velocity, grad(v)) + sigma*v)*r*dx
    methodname = "DW"
elif method == 4: # VMS
    hh = 1.0/nx
    ttau = m.pow((4.0*mu/(hh*hh) + 2.0*vnorm/hh + sigma),-1)
    F -= (ttau)*(- mu*div(grad(v)) - dot(velocity, grad(v)) + sigma*v)*r*dx
    methodname = "VMS"
elif method == 6:   # EFR
    methodname = "EFR"
    #S = input("Set [S]cale of filtering radius delta (delta = S*mesh_size) ")
    S = 1
    delta = S*1.0/nx

    u_tilde = Function(Q)
    u_bar = Function(Q)
else:
    methodname = "Galerk"
    # Galerkin with no stabilization terms

# Create bilinear and linear forms
a = lhs(F)
L = rhs(F)



# Assemble matrix
A = assemble(a)
bc.apply(A)

# Create linear solver and factorize matrix
solver = LUSolver(A)
solver.parameters["reuse_factorization"] = True

# Output file
out_file = File(folder+"u_"+methodname+".pvd")
if method == 5:
    out_file_ue = File(folder+"u_exact.pvd")
# Set intial condition
u = u_n

# Time-stepping, plot initial condition.
i = 0


# Create progress bar
progress = Progress('Time-stepping')
set_log_level(PROGRESS)

if method != 6:
    while t - T < DOLFIN_EPS:
        # Assemble vector and apply boundary conditions
        b = assemble(L)
        bc.apply(b)

        # Solve the linear system (re-use the already factorized matrix A)
        solver.solve(u.vector(), b)

        # Exact solution in time
        if method == 5:
            ue = Expression(u_exact.cppcode, degree = R, t = t)
            uee = interpolate(ue, Q)
            uee.rename('Exact','Exact')
            out_file_ue << (uee, float(t))
            u_exact.t += dt

        # Copy solution from previous interval
        u_n = u
        f0.t += dt
        f.t += dt

        # Save the solution to file
        out_file << (u, t)

        # Move to next interval and adjust boundary condition
        t += dt
        i += 1

        # Update progress bar
        progress.update(t / T)


else: # EFR Method
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
    A2 = assemble(a2)


    # Apply boundary conditions to matrices
    bc.apply(A2)


    # Create VTK files for visualization output
    out_file_utilde = File(folder+"utilde_EFR.pvd")  # u tilde
    out_file_ind = File(folder+"a_EFR.pvd")          # indicator function
    out_file_ubar = File(folder+"ubar_EFR.pvd")      # filtered solution

    # Time-stepping
    while t - T < DOLFIN_EPS:
        # Update current time
        t += dt

        # Step 1
        b = assemble(L)
        bc.apply(b)
        solve(A, u_.vector(), b)

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
        if n % saveTimesteps == 0:
            out_file << (u_, float(t))
            out_file_utilde << (u_tilde, float(t))
            out_file_ubar << (u_bar, float(t))

        # Update previous solution and source term
        u_n.assign(u_bar)
        f_n = Expression(f.cppcode, degree = R, sigma = sigma, t = t)
        f.t += dt
        f_n.t += dt

        # Update progress bar
        progress.update(t / T)