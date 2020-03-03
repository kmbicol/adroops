delta = 1.0/nx
# Define indicator function to evaluate current time step
def a(u_tilde, u_, t):
    # EFR, N = 0
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
out_file_utilde = File(folder+"utilde.pvd")  # u tilde
out_file_ind = File(folder+"a.pvd")          # indicator function
out_file_ubar = File(folder+"ubar.pvd")      # filtered solution

# Time-stepping
for n in range(num_steps):
    # Update current time
    t += tt

    # Step 1: Solve on Coarse Mesh
    b1 = assemble(L1)
    bc.apply(b1)
    solve(A1, u_.vector(), b1)

    # Step 2a: Solve "Filter Problem"
    b2 = assemble(L2)
    bc.apply(b2)
    solve(A2, u_tilde.vector(), b2)

    # Step 2b: Apply Indicator Function and Apply Filter
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