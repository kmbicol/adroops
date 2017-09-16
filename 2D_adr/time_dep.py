from dolfin import *

# Load mesh and subdomains
mesh = UnitSquareMesh(20,20)
h = CellSize(mesh)

# Create FunctionSpaces
Q = FunctionSpace(mesh, "CG", 1)

# Create velocity Function from file
velocity = as_vector([1.0, 1.0])

# Parameters
T = 0.03
dt = 0.01
t = dt
sigma = 1.0
mu = 0.001

# Boundary values
u_D = Constant(0.0)
