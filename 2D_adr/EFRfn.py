# LHS of Helmholtz Filtering Step
a2 = v*u*dx + delta*delta*dot(grad(v), grad(u))*dx
A2 = assemble(a2)
bc.apply(A2)

def b2(newtilde): # RHS of Helmholtz Filtering Step
	L2 = v*newtilde*dx
	b2 = assemble(L2)
	bc.apply(b2)
	return b2

u_1tilde = Function(Q)
solve(A2, u_1tilde.vector(), b2(u_))
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