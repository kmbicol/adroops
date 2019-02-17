## Program Details
# Time Discretization: BDF2

from dolfin import *
import numpy as np
import sympy as sym
import csv
import matplotlib.pyplot as plt


class SimADR(object):
    '''
    Common base class for all ADR simulations
    Attributes:
        method:
    
    '''
    
    ## Parameters for all simulations:
    basefolder = 'ADR-results/'
    degree = 2              # degree of finite element (Expressions will be degree+1)
    gridSize = [25, 50, 100, 200, 400] # nx
    timeSize = [0.1, 0.01, 0.001]       # dt
    NeedsDirBC = ['HeatSim', 'RisingHumpSim']
    saveEvery = 10
    
    def __init__(self, method):
        #self.simName = simName          # string: simulation name
        self.method = method              # string: method can be Galerk, SUPG, EFR
        # self.velocity = velocity        # Expression: advective velocity vector
        # self.sigma = sigma                # Scalar: reaction coefficient
        # self.mu = mu                  # Scalar: diffusivity coefficient
        # self.f_code = sourceFn          # Expression: source function
        # self.u_code = uExact            # Expression: exact solution or  boundary condition if no exact solution available

    def createOutput(self, nx, dt):
        ''' 
        Creates pvd output files for ParaView 
        nx: this should be within run function

        '''

        self.save_uexact = File(self.folder+str(dt)+"_Exact_u_"+str(nx)+".pvd") 
        self.save_ubar = File(self.folder+str(dt)+"_"+self.method+"_u_"+str(nx)+".pvd") 

    def modelSetup(self, nx, t):
        degree = self.degree
        u_code = self.u_code # when exact code is given; if not, treat as Dir boundary condition
        f_code = self.f_code
        t = self.t
        
        self.u_exact = Expression(u_code, degree = degree+1, t = t)
        self.f = Expression(f_code, degree = degree+1, t = t)

        mesh = UnitSquareMesh(nx,nx)
        self.h = CellDiameter(mesh)
        Q = FunctionSpace(mesh, "CG", degree)
        self.Q = Q
        self.mesh = mesh
        # Set up boundary condition
        self.u_D = Expression(self.u_exact.cppcode, degree = degree+1, t = t)

        # Test and trial functions
        self.u, self.v = TrialFunction(Q), TestFunction(Q)
        self.u_n0 = interpolate(self.u_D, Q)
        self.u_n1 = Function(Q)
        self.u_ = Function(Q)

    def setDirBC(self,A,b):
        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(self.Q, self.u_D, boundary)
        bc.apply(A)
        bc.apply(b)
        
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
        
        # Add in SUPG stabilization terms (Backward Euler)
        if self.method == 'SUPG':
            # based on paper's definition of residual and stabilization term
            Lt = -mu*div(grad(u)) + dot(velocity, grad(u)) + (sigma+1.0/dt)*u 
            ft = u_n0/dt + f
            r = ft - Lt
            vnorm = sqrt(dot(velocity, velocity))
            F -= dt*(self.h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx    
            
        # Create bilinear and linear forms
        a1 = lhs(F)
        L1 = rhs(F)

        # Assemble matrices
        A1 = assemble(a1)
        b = assemble(L1)
        
        # if example needs Dirchlet BC, put in list below
        if self.ex in self.NeedsDirBC:
            self.setDirBC(A1,b)

        self.updateTime(dt)
    
        solve(A1, u_n1.vector(), b)#, 'gmres')

    # -------------------------------------- #

        # Galerkin variational problem
        F = (1.5*u - 2.0*u_n1 + 0.5*u_n0)*v*dx
        F += dt*(mu*dot(grad(v), grad(u))*dx + v*dot(velocity, grad(u))*dx + sigma*v*u*dx - f*v*dx)
        
        # Add in SUPG stabilization terms (BDF2)
        if self.method == 'SUPG':
            # based on paper's definition of residual and stabilization term
            Lt = -mu*div(grad(u)) + dot(velocity, grad(u)) + (sigma+1.5/dt)*u 
            ft = 2.0*u_n1/dt - 0.5*u_n0/dt + f
            r = ft - Lt
            vnorm = sqrt(dot(velocity, velocity))
            F -= dt*(self.h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx    
        
        self.u_n0, self.u_n1 = u_n0, u_n1 # save initial conditions for model run

        # Create bilinear and linear forms
        self.a1 = lhs(F)
        self.L1 = rhs(F)


    def runSim(self, nx, dt):
        """ 
        Sets all variables, formulations, and other simulation parameters. 
        
        """
        # Create progress bar
        self.progress = Progress('Time-stepping')
        set_log_level(PROGRESS)

        self.createOutput(nx, dt)
        self.t = 0.0 # Start Time

        self.modelSetup(nx, self.t)
        u_ = self.u_
            
        self.opSetup(dt)
        
        self.progress.update(self.t / self.T)

        # Outputting files
        # ue = interpolate(self.u_exact, Q)

        # Save t = 0.0 + dt
        # self.save_uexact << (ue, float(t))

        # u_.rename('u','u')
        # self.save_ubar << (u_, float(self.t))
        
        it = 2
        while self.t - self.T + dt < DOLFIN_EPS:
            # Step 1 Solve on Coarse Grid
            self.updateTime(dt)
            
            # Assemble matrices
            A1 = assemble(self.a1)
            b = assemble(self.L1)
            
            if self.ex in self.NeedsDirBC:
                self.setDirBC(A1,b)

            solve(A1, self.u_.vector(), b)#, 'gmres')
            self.progress.update(self.t / self.T)

            # Save solution
            u_.rename('u','u')
            if it % self.saveEvery == 0:
                self.save_ubar << (u_, float(self.t))
                if self.saveExact == True: 

                    interp_uExact = interpolate(self.u_exact, self.Q)
                    interp_uExact.rename('Exact','Exact')
                    self.save_uexact << (interp_uExact, float(self.t))
                    self.compute_errors(u_)
                    self.compute_extrema(u_)
            
            # Update initial conditions
            self.u_n0.assign(self.u_n1)
            self.u_n1.assign(u_)
            it += 1
            
        # save last time step
        self.save_ubar << (u_, float(self.t))
        if self.saveExact == True: 
            interp_uExact = interpolate(self.u_exact, self.Q)
            interp_uExact.rename('Exact','Exact')
            self.save_uexact << (interp_uExact, float(self.t))        
    
    def loopAllSims(self):
        for dt in self.timeSize:
            for nx in self.gridSize:
                self.runSim(nx,dt)

        # self.save_ubar << (u_, float(t))
        # L2, H1 = compute_errors(u_exact, u_, t, mesh)
        # maxval, minval = compute_extrema(u_, t)


    def compute_errors(self, u):
        L2n = errornorm(self.u_exact, u, norm_type='L2', degree_rise=3, mesh=self.mesh)
        H1n = errornorm(self.u_exact, u, norm_type='H1', degree_rise=3, mesh=self.mesh)
        print L2n, H1n

    def compute_extrema(self, u):
        maxval = np.amax(u.vector().get_local())
        minval = np.amin(u.vector().get_local())
        print maxval, minval
        
    def FilterRelax:
        # i'm thinking that filter relax step can happen after the evolve step
        pass

##################################################################################
##################################################################################

                
class HeatSim(SimADR):
    # Heat Example from FEniCS documentation
    # DOESN'T WORK RIGHT NOW
    alpha = 3.0; beta = 1.2
    
    T = 6.0             # total simulation time
    velocity = Expression(('0.0','0.0'), degree = 2, t = 0)
    mu = Constant(-1.0)
    sigma = Constant(0.0)
    ex = 'HeatSim'
        
    saveExact = True
    
    # Exact solution 
    u_code = '1 + x[0]*x[0] + ' + str(alpha) + '*x[1]*x[1] + ' + str(beta) + '*t' 
    f_code = str(beta)+' - 2 - 2*'+str(alpha)

    degree = 1
    def __init__(self, simName, method):
        SimADR.__init__(self, simName, method)
        self.folder = self.basefolder + self.ex + '/'
        print(self.f_code)
        
    def testRun(self):
        self.runSim(nx = 8, dt = 0.1)
                
class SwirlSim(SimADR):
    # Modified Example from https://github.com/redbKIT/redbKIT
    # code works fine! 
    
    T = 2*np.pi             # total simulation time
    velocity = Expression(('cos(t)','sin(t)'), degree = SimADR.degree, t = 0)
    mu = 0.005
    sigma = 0.01
    ex = 'SwirlSim'
        
    saveExact = False
    
    u_code = '0.0' # No exact solution available
    f_code = 'exp(-(pow(x[0]-0.5,2)+pow(x[1]-0.5,2))/pow(0.07,2))'

    def __init__(self, simName, method):
        SimADR.__init__(self, simName, method)
        self.folder = self.basefolder + self.ex + '/'
                
                
class TwoSourceSim(SimADR):
    # Modified Example from https://github.com/redbKIT/redbKIT
    # Two sources
    # code works fine! 
    
    T = 2*np.pi             # total simulation time
    velocity = Expression(('cos(t)','sin(t)'), degree = SimADR.degree, t = 0)
    mu = 0.005
    sigma = 0.01
    ex = 'TwoSourceSim'
    
    saveExact = False
    
    u_code = '0.0' # No exact solution available
    f_code = 'exp(-(pow(x[0]-0.75,2)+pow(x[1]-0.75,2))/pow(0.07,2)) + exp(-(pow(x[0]-0.25,2)+pow(x[1]-0.25,2))/pow(0.07,2))'

    def __init__(self, simName, method):
        SimADR.__init__(self, simName, method)
        self.folder = self.basefolder + self.ex + '/'
        
    def run(self):
        for dt in self.timeSize[1:2]:
            for nx in self.gridSize[0:2]:
                self.setupSim(nx,dt)


class RisingHumpSim(SimADR):
    # Iliescu example
    # Works for sigma bigger than 10**(-5)
    
    T = 0.5             # total simulation time
    velocity = Expression(('2.0','3.0'), degree = SimADR.degree, t = 0)
    mu = 10**(-5)
    sigma = 1.0
    ex = 'RisingHumpSim'
    
    saveExact = True
    
    u_code = '16*(-0.318309886183791*x[0]*x[1]*(-x[0] + 1)*(-x[1] + 1)*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) + 0.5*x[0]*x[1]*(-x[0] + 1)*(-x[1] + 1))*sin(3.14159265358979*t) '
    f_code = '(-5.09295817894065e-6*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*((4000.0*x[0] - 2000.0)*(8000.0*x[0] - 4000.0) + (4000.0*x[1] - 2000.0)*(8000.0*x[1] - 4000.0))*(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0)*sin(3.14159265358979*t) + pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)*(0.318309886183791*atan(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0) - 0.5)*(-16.0*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) - 50.2654824574367*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*cos(3.14159265358979*t) - 48.0*x[0]*x[1]*(x[0] - 1)*sin(3.14159265358979*t) - 32.0*x[0]*x[1]*(x[1] - 1)*sin(3.14159265358979*t) - 48.0*x[0]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) + 3.2e-5*x[0]*(x[0] - 1)*sin(3.14159265358979*t) - 32.0*x[1]*(x[0] - 1)*(x[1] - 1)*sin(3.14159265358979*t) + 3.2e-5*x[1]*(x[1] - 1)*sin(3.14159265358979*t)) + (pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1)*(-10.1859163578813*x[0]*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1) - 15.278874536822*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 0.0407436654315252*x[0]*x[1]*(x[0] - 1)*(x[1] - 1) + 1.01859163578813e-5*x[0]*x[1]*(x[0] - 1)*(4000.0*x[1] - 2000.0) + 1.01859163578813e-5*x[0]*x[1]*(4000.0*x[0] - 2000.0)*(x[1] - 1) + 1.01859163578813e-5*x[0]*(x[0] - 1)*(x[1] - 1)*(4000.0*x[1] - 2000.0) + 1.01859163578813e-5*x[1]*(x[0] - 1)*(4000.0*x[0] - 2000.0)*(x[1] - 1))*sin(3.14159265358979*t))/pow(pow(2000.0*pow(x[0] - 0.5, 2) + 2000.0*pow(x[1] - 0.5, 2) - 125.0, 2) + 1, 2)'

    def __init__(self, method):
        SimADR.__init__(self, method)
        self.folder = self.basefolder + self.ex + '/'
        self.saveEvery = 25
        
