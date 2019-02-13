## Program Details
# Method: Galerkin (Evolve Only)
# Time Discretization: BDF2

from dolfin import *
import numpy as np
import sympy as sym
import csv


class SimADR(object):
    '''
    Common base class for all ADR simulations
    Attributes:
        method:
    
    '''
    
    ## Parameters for all simulations:
    folder = 'ADR-results/'
    T = 2*np.pi           	# total simulation time
    degree = 2         		# degree of finite element (Expressions will be degree+1)
    gridSize = [25, 50] 	#, 100, 200] # nx
    timeSize = [0.1] 		# dt
    
    def __init__(self, simName):
        self.simName = simName          # string: simulation name
        self.folder += simName + '/'
        # self.velocity = velocity        # Expression: advective velocity vector
        # self.sigma = sigma				# Scalar: reaction coefficient
        # self.mu = mu 					# Scalar: diffusivity coefficient
        # self.f_code = sourceFn          # Expression: source function
        # self.u_code = uExact            # Expression: exact solution or  boundary condition if no exact solution available

    def createOutput(self, nx):
        ''' 
        Creates pvd output files for ParaView 
        nx: this should be within run function

        '''

        self.save_uexact = File(self.folder+"Exact_u_"+str(nx)+".pvd") 
        self.save_ubar = File(self.folder+"Galerk_u_"+str(nx)+".pvd") 

    def modelSetup(self, nx):
        degree = self.degree
        u_code = self.u_code
        f_code = self.f_code

        self.u_exact = Expression(u_code, degree = degree+1, t = t)
        self.f = Expression(f_code, degree = degree+1, t = t)

        mesh = UnitSquareMesh(nx,nx)
        Q = FunctionSpace(mesh, "CG", degree)

        # Set up boundary condition
        self.u_D = Expression(u_exact.cppcode, degree = degree+1, t = t)

        # Test and trial functions
        self.u, self.v = TrialFunction(Q), TestFunction(Q)
        self.u_n0 = interpolate(self.u_D, Q)
        self.u_n1 = Function(Q)
        self.u_ = Function(Q)

    def updateTime(self, dt):
    	# Update time step
        self.t += dt
        self.u_D.t = self.t
        self.f.t = self.t
        self.velocity.t = self.t 
        self.u_exact.t = self.t

    def opSetup(self, dt):
        # Galerkin variational problem
        u,v = self.u, self.v
        u_n0, u_n1 = self.u_n0, self.u_n1
        
        velocity, mu, sigma = self.velocity, self.mu, self.sigma # ADR
        f = self.f # source function


        # Backward Euler (to get u_n1 for BDF2)
        F = (u - u_n0)*v*dx
        F += dt*(mu*dot(grad(v), grad(u))*dx + v*dot(velocity, grad(u))*dx + sigma*v*u*dx - f*v*dx)

        # Create bilinear and linear forms
        a1 = lhs(F)
        L1 = rhs(F)

        # Assemble matrices
        A1 = assemble(a1)
        b = assemble(L1)
    #    bc = DirichletBC(Q, u_D, boundary)
    #    bc.apply(A1)
    #    bc.apply(b)

    	self.updateTime(dt)
    
        solve(A1, u_n1.vector(), b)#, 'gmres')

    # -------------------------------------- #

        # Galerkin variational problem
        F = (1.5*u - 2.0*u_n1 + 0.5*u_n0)*v*dx
        F += dt*(mu*dot(grad(v), grad(u))*dx + v*dot(velocity, grad(u))*dx + sigma*v*u*dx - f*v*dx)
        
        self.u_n0, self.u_n1 = u_n0, u_n1 # save initial conditions for model run

        # Create bilinear and linear forms
        self.a1 = lhs(F)
        self.L1 = rhs(F)


    def setupSim(self, nx, dt):
        """ 
        Sets all variables, formulations, and other simulation parameters. 
        
        """
        # Create progress bar
        progress = Progress('Time-stepping')
        set_log_level(PROGRESS)

        self.createOutput(self, nx)
        self.t = 0.0 # Start Time

        self.modelSetup(nx)
        u_ = self.u_

        def boundary(x, on_boundary):
                return on_boundary
            
        self.opSetup(dt)
        
        self.progress.update(self.t / self.T)

        # Outputting files
        # ue = interpolate(self.u_exact, Q)

        # Save t = 0.0 + dt
        # self.save_uexact << (ue, float(t))

        u_.rename('u','u')
        self.save_ubar << (u_, float(t))

        while t - T + dt < DOLFIN_EPS:
            # Step 1 Solve on Coarse Grid

        self.updateTime(dt)
            # Assemble matrices
            A1 = assemble(self.a1)
            b = assemble(self.L1)
    #         bc = DirichletBC(Q, u_D, boundary)
    #         bc.apply(A1)
    #         bc.apply(b)

            solve(A1, self.u_.vector(), b)#, 'gmres')
            self.progress.update(self.t / self.T)

            # Save solution
            u_.rename('u','u')
            self.save_ubar << (u_, float(t))
            
            # Update initial conditions
            u_n0.assign(u_n1)
            u_n1.assign(u_)

        # self.save_ubar << (u_, float(t))
        # L2, H1 = compute_errors(u_exact, u_, t, mesh)
        # maxval, minval = compute_extrema(u_, t)

        # print nx, L2, H1,maxval,minval
        # return str(L2)+','+str(H1)+','+str(maxval)+','+str(minval)


	# def compute_errors(self, u_e, u, t, mesh):
	#     L2n = errornorm(u_e, u, norm_type='L2', degree_rise=3, mesh=mesh)
	#     H1n = errornorm(u_e, u, norm_type='H1', degree_rise=3, mesh=mesh)
	#     return L2n, H1n

	# def compute_extrema(self, u, t):
	#     maxval = np.amax(u.vector().get_local())
	#     minval = np.amin(u.vector().get_local())
	#     return maxval, minval

    def runSwirlSim(self,nx,dt):
        self.velocity = Expression(('cos(t)','sin(t)'), degree = self.degree, t = 0)
        self.mu = 0.5
        self.sigma = 0.1

        self.source_f = Expression('exp(-(pow(x[0]-0.5,2)+pow(x[1]-0.5,2))/pow(0.07,2))', degree = self.degree)



