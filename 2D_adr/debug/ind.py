from dolfin import *
import math as m
import numpy as np

# want to check that a(utilde, u_, t) works properly

nx = 100
P = 2 

mesh = UnitSquareMesh(nx,nx)
h = CellDiameter(mesh)
Q = FunctionSpace(mesh, "CG", P)

def a(u_tilde, u_, t):
	indicator = Expression('sqrt((a-b)*(a-b))', degree = 2, a = u_, b = u_tilde)
	indicator = interpolate(indicator, Q)
	max_ind = np.amax(indicator.vector().get_local())#.vector().array())

	# Normalize indicator such that it's between [0,1].
	if max_ind < 1:
	   max_ind = 1.0

	indicator = Expression('a/b', degree = 2, a = indicator, b = max_ind)
	indicator = interpolate(indicator, Q) 
	indicator.rename('a','a')

	return indicator


out_file_utilde = File("indicator/utilde.pvd")
out_file_ubar = File("indicator/ubar.pvd")
out_file_ind = File("indicator/ind.pvd") 

u_tilde = Expression('cos(2*x[0])', degree = 2)
proj_utilde = interpolate(u_tilde,Q)

u_ = Expression('sin(2*x[0])', degree = 2)
proj_utilde = interpolate(u_,Q)



ind = a(u_tilde,u_)
out_file_ind = File("indicator/ind.pvd") 
out_file_ind << ind