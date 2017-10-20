from dolfin import *
import math as m

print("\n (1) SUPG \n (2) GLS \n (3) DW \n (4) VMS \n (default) Galerkin")
method = input("Choose a stabilization method: ")
#nx = input("h = 1/nx, let nx = ")
nx = 300

# Load mesh and subdomains
mesh = UnitSquareMesh(nx,nx)
h = CellSize(mesh)

# Create FunctionSpaces
Q = FunctionSpace(mesh, "CG", 2)

# Create velocity Function from file
velocity = as_vector([2.0, 3.0])

# Initialise source function and previous solution function
#f  = Constant(1.0)
u0 = Function(Q)
R = 2

# Boudanry values
u_D = Constant(0.0)

# Parameters
T = 2.0
dt = 0.001
t = dt
sigma = 1.0
mu = 10**(-6)

# Test and trial functions
u, v = TrialFunction(Q), TestFunction(Q)
u_exact = Expression('-16*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(0.318309886183791*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5)*sin(3.14159265358979*t)', degree = R, t = t)
adr_f = Expression('(-5.09295817894065e-6*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*((4000.0*x[0] - 2000.0)*(8000.0*x[0] - 4000.0) + (4000.0*x[1] - 2000.0)*(8000.0*x[1] - 4000.0))*(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0)*sin(3.14159265358979*t) + pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)*(0.318309886183791*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5)*(-16.0*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) - 50.2654824574367*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*cos(3.14159265358979*t) - 48.0*x[0]*x[1]*(x[0] - 1)*sin(3.14159265358979*t) - 32.0*x[0]*x[1]*(x[1] - 1)*sin(3.14159265358979*t) - 48.0*x[0]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) + 3.2e-5*x[0]*(x[0] - 1)*sin(3.14159265358979*t) - 32.0*x[1]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) + 3.2e-5*x[1]*(x[1] - 1)*sin(3.14159265358979*t)) + (pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)*(-10.1859163578813*x[0]*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1) - 15.278874536822*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 0.0407436654315252*x[0]*x[1]*(x[0] - 1)*(x[1] - 1) + 1.01859163578813e-5*x[0]*x[1]*(x[0] - 1)*(4000.0*x[1] - 2000.0) + 1.01859163578813e-5*x[0]*x[1]*(4000.0*x[0] - 2000.0)*(x[1] - 1) + 1.01859163578813e-5*x[0]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 1.01859163578813e-5*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1))*sin(3.14159265358979*t))/pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)', degree = R, t = t)

f = Expression(adr_f.cppcode, degree = R, t = t)
f0 = Expression(adr_f.cppcode, degree = R, t = 0)
'''
# Mid-point solution
u_mid = 0.5*(u0 + u)
f_mid = 0.5*(f0 + f)


# Residual
r = u - u0 + dt*(- mu*div(grad(u_mid)) + dot(velocity, grad(u_mid)) + sigma*u_mid - f_mid)


# Galerkin variational problem
F = v*(u - u0)*dx + dt*(mu*dot(grad(v), grad(u_mid))*dx \
                        + v*dot(velocity, grad(u_mid))*dx \
                        + sigma*v*u*dx \
                        - f_mid*v*dx)
'''

# Backward Euler
r = u - u0 + dt*(- mu*div(grad(u)) + dot(velocity, grad(u)) + sigma*u - f)


# Galerkin variational problem
F = v*(u - u0)*dx + dt*(mu*dot(grad(v), grad(u))*dx \
                        + v*dot(velocity, grad(u))*dx \
                        + sigma*v*u*dx \
                        - f*v*dx)

# Add stabilisation terms

# residual
r = u - u0 + dt*(- mu*div(grad(u)) + dot(velocity, grad(u)) + sigma*u - f)
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
else:
    methodname = "Galerk"
    # Galerkin with no stabilization terms

# Create bilinear and linear forms
a = lhs(F)
L = rhs(F)

# Set up boundary condition
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(Q, u_D, boundary)

# Assemble matrix
A = assemble(a)
bc.apply(A)

# Create linear solver and factorize matrix
solver = LUSolver(A)
solver.parameters["reuse_factorization"] = True

# Output file
out_file = File("results_full_ilie_BE/h"+str(nx)+"u_"+methodname+".pvd")
if method == 5:
    out_file_ue = File("results_full_ilie_BE/h"+str(nx)+"u_exact.pvd")
# Set intial condition
u = u0

# Time-stepping, plot initial condition.
i = 0


# Create progress bar
progress = Progress('Time-stepping')
set_log_level(PROGRESS)


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
        uee.rename('u','u')
        out_file_ue << (uee, float(t))
        u_exact.t += dt

    # Copy solution from previous interval
    u0 = u
    f0.t += dt
    f.t += dt

    # Save the solution to file
    out_file << (u, t)

    # Move to next interval and adjust boundary condition
    t += dt
    i += 1

    # Update progress bar
    progress.update(t / T)