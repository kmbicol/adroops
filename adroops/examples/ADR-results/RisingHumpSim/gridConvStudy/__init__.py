# %load ../__init__.py
## Program Details
# Time Discretization: BDF2
import pickle
from time import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

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
    NeedsDirBC = ['HeatSim', 'RisingHumpSim']  # by default Neuman BC (free BC)
    saveEvery = 10
    
    def __init__(self, method):
        self.method = method              # String: method can be Galerk, SUPG, EFR
        self.init_time = time()
        
    def __str__(self):
        dt = datetime.fromtimestamp(self.init_time)
        return self.ex + ' created at {:%H:%M on %m-%d-%Y}'.format(dt)

    def createOutput(self, nx, dt):
        ''' 
        Creates pvd output files for ParaView 
        nx: this should be within run function

        '''
        if self.saveExact == True:
            self.save_uexact = File(self.folder+str(dt)+"_Exact_u_"+str(nx)+".pvd") 

        if self.method == 'EFR':
            methodname = self.method
            self.save_indicator = File(self.folder+str(dt)+"_"+methodname+"_N"+str(self.N)+"_delta"+str(self.delta)+"_chi_order"+str(self.chi_order)+"_ind_"+str(nx)+".pvd")
            self.save_ubar = File(self.folder+str(dt)+"_"+methodname+"_N"+str(self.N)+"_delta"+str(self.delta)+"_chi_order"+str(self.chi_order)+"_u_"+str(nx)+".pvd")
        else:
            methodname = self.method
            self.save_ubar = File(self.folder+str(dt)+"_"+methodname+"_u_"+str(nx)+".pvd")
        
    def modelSetup(self, nx):
        degree = self.degree
        u_code = self.u_code # when exact code is given; if not, treat as Dir boundary condition
        f_code = self.f_code
        t = self.t
        
        self.u_exact = Expression(u_code, degree = degree+1, t = t)
        self.f = Expression(f_code, degree = degree+1, t = t)

        mesh = UnitSquareMesh(nx,nx)
        Q = FunctionSpace(mesh, "CG", degree)
        self.h = np.sqrt(2.0)*(1.0/nx)
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

    def opSetup(self, nx, dt):
        # set test and trial functions and initial conditions
        u,v = self.u, self.v
        u_n0, u_n1 = self.u_n0, self.u_n1

        # model parameters    
        velocity, mu, sigma = self.velocity, self.mu, self.sigma # ADR
        f = self.f # source function


        # Galerkin: Backward Euler (to get u_n1 for BDF2)
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
        
        # if example needs Dirchlet BC (add new classes to above list)
        if self.ex in self.NeedsDirBC:
            self.setDirBC(A1,b)

        self.updateTime(dt)
    
        solve(A1, u_n1.vector(), b)#, 'gmres')

        # FilterRelax is activated IF AND ONLY IF method = 'EFR' 
        if self.method == 'EFR':
            self.FilterRelax(u_n1, nx, dt) # overwrite u_n1

        # Galerkin: BDF2
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
        self.t = 0.0 # Start Time

        
        # Create progress bar
        self.progress = Progress('Time-stepping')
        set_log_level(PROGRESS)


        self.modelSetup(nx)

        if self.method == 'EFR':
            # recall: self.method = ['EFR', N, chi_order]
            self.autosetEFRParam(nx, dt)

        self.createOutput(nx, dt)

        
        self.opSetup(nx, dt)
        
        self.progress.update(self.t / self.T)
        
        it = 2 # BDF2 uses two points for time discretization

        while self.t - self.T + dt < DOLFIN_EPS:

            self.updateTime(dt)
            
            # Assemble matrices
            A1 = assemble(self.a1)
            b = assemble(self.L1)
            
            if self.ex in self.NeedsDirBC:
                self.setDirBC(A1,b)

            solve(A1, self.u_.vector(), b)#, 'gmres')
            self.progress.update(self.t / self.T)

            # FilterRelax is activated IF AND ONLY IF method = 'EFR' 
            self.FilterRelax(self.u_, nx, dt) # overwrite u_

            # Save solution
            self.u_.rename('u','u')
            if it % self.saveEvery == 0:
                self.save_ubar << (self.u_, float(self.t))
                if self.saveExact == True: 

                    interp_uExact = interpolate(self.u_exact, self.Q)
                    interp_uExact.rename('Exact','Exact')
                    self.save_uexact << (interp_uExact, float(self.t))
                    self.computeErrors(self.u_)
                    self.computeExtrema(self.u_)
            
            # Update initial conditions
            self.u_n0.assign(self.u_n1)
            self.u_n1.assign(self.u_)
            it += 1
            
        # save last time step
        self.save_ubar << (self.u_, float(self.t))
        if self.saveExact == True: 
            interp_uExact = interpolate(self.u_exact, self.Q)
            interp_uExact.rename('Exact','Exact')
            self.save_uexact << (interp_uExact, float(self.t))

    
    def testGridConv(self):
        dt = 0.01
        self.folder += 'gridConvStudy/'
        self.L2norms = np.zeros(len(self.gridSize))
        self.H1norms = np.zeros(len(self.gridSize))
        self.maxs = np.zeros(len(self.gridSize))
        self.mins = np.zeros(len(self.gridSize))
        index = 0
        for nx in self.gridSize:
            self.runSim(nx,dt)
            # errors computed at end simulation time T
            print self.T
            self.L2norms[index], self.H1norms[index] = self.computeErrors(self.u_)
            self.maxs[index], self.mins[index] = self.computeExtrema(self.u_)
            index += 1
        self.plotErr(self.L2norms, self.H1norms)
        self.plotExtrema(self.maxs,'max')
        self.plotExtrema(self.mins,'min')
            
    def testChi(self):
        self.folder += 'chiStudy/'
        for N in [0,1,2,3]:
            print('N = '+str(N))
            self.N = N
            for chi_order in [0,1,2]:
                print('chi_order = '+str(chi_order))
                self.chi_order = chi_order
                self.runSim(nx = 100,dt = 0.01)

    def computeErrors(self, u):
        # only works if you have the exact solution

        L2n = errornorm(self.u_exact, u, norm_type='L2', degree_rise=3, mesh=self.mesh)
        H1n = errornorm(self.u_exact, u, norm_type='H1', degree_rise=3, mesh=self.mesh)
        return L2n, H1n
    
    def computeExtrema(self, u):
        # only works if you have the exact solution

        maxval = np.amax(u.vector().get_local())
        minval = np.amin(u.vector().get_local())
        return maxval, minval
    
    def openSim(self, pathto):
        # pathto: .dat file
        with open(pathto+'.dat', 'rb') as f:
            reloadSim = pickle.load(f)

        return reloadSim
    def howToSaveSim(self):
        print(""" Execute the following code after running simulation:
        with open(MYCLASS.folder+'.dat', 'wb') as simInfo:\n
        \t pickle.dump(MYCLASS, simInfo)
        """)
##################################################################################
##################################################################################

## EVOLVE-FILTER-RELAX (EFR) ALGORITHM
# Below are the functions that go into implementing EFR

    def autosetEFRParam(self, nx, dt):
        # set EFR parameters based on nx
        self.delta = 1.0/nx
        # already stored: self.N, self.chi_order
        ## according to APR derivation
        # Dr. Q, what h do we use? the big diameter or shortest edge?
        # BQV paper states "mesh size of h" so let's assume h = mesh diameter

        # alpha: coefficient in front of u at t^(n+1)
        # eta: finest mesh (kolmogorov scale)

        alpha = 2.0 # BDF2
        ## How to set eta? BQ paper, p. 3 says that a mesh is considered refined enough when the local Peclet number (see p. 3 of our paper, revised version) is < 1
        
        W = VectorFunctionSpace(self.mesh, "CG", self.degree)
        b = interpolate(self.velocity, W)
        eta = 2*self.mu/norm(b.vector(),'linf')
        self.eta = eta
        chi_order = self.chi_order
        # uses approximation grad_h = 1/h
        if chi_order == 1:
            self.chi = (self.mu/(alpha*self.delta**2))*(float(self.h)/eta - 1)*dt
        # uses approximation laplace_h = 1/h^2
        elif chi_order == 2:
            self.chi = (self.mu/(alpha*self.delta**2))*(float(self.h)**2/eta**2 - 1)*dt
        else: # default sets chi according to nx
            chi_dict = {'25': 1, '50': 0.5, '100': .25, '200': 1.0/16., '400': 1.0/256.}
            self.chi = chi_dict.get(str(nx))

    def setEFRParam(self, delta, N, chi):
        # manually set EFR parameters
        self.delta = delta
        self.N = N
        self.chi = chi

    def FilterRelax(self, u_, nx, dt):
        if self.method == 'EFR':
            delta = self.delta
            N = self.N
            chi = self.chi
            u_n1int = u_
            
            ## Main function

            # filter step
            u_n1bar, ind = self.filterStep(u_n1int, delta, N)
            ind.rename('Indicator Function','Indicator Function')
            self.save_indicator << (ind, float(self.t))

            # relaxation step
            self.u_.assign(Constant(1 - chi)*u_n1int + Constant(chi)*u_n1bar)
            self.u_.rename('u','u')


    def filterEqn(self, u_, delta):
        # helper function
        # implements linear Helmholtz filter equation

        Q = self.Q

        u_tilde = TrialFunction(Q)
        v = TestFunction(Q)

        F_Hfilter = v*u_tilde*dx - v*u_*dx + delta*delta*dot(grad(v), grad(u_tilde))*dx  

        a_Hfilter = lhs(F_Hfilter)
        L_Hfilter = rhs(F_Hfilter)
        
        A_Hfilter = assemble(a_Hfilter)
        b_Hfilter = assemble(L_Hfilter)
        
        # if example needs Dirchlet BC (add new classes to above list)
        if self.ex in self.NeedsDirBC:
            self.setDirBC(A_Hfilter,b_Hfilter)

        solver = LUSolver(A_Hfilter)
        u_tilde = Function(Q)
        solver.solve(u_tilde.vector(), b_Hfilter)
        
        return u_tilde

    def filterStep(self, u_, delta, N):
        degree = self.degree
        Q = self.Q

    # ask Dr. Q if degree of below expressions should be degree+1 or just degree (???)

        # helper function
        if N == 0:
            u_tilde0 = self.filterEqn(u_, delta)
            DF = Expression('u_tilde0', degree = degree, u_tilde0 = u_tilde0)

        if N == 1:
            u_tilde0 = self.filterEqn(u_, delta)
            u_tilde1 = self.filterEqn(u_tilde0, delta)
            DF = Expression('2.0*u_tilde0 - u_tilde1', degree = degree, u_tilde0 = u_tilde0, u_tilde1 = u_tilde1)

        if N == 2:
            u_tilde1 = self.filterEqn(u_, delta)
            u_tilde2 = self.filterEqn(u_tilde1, delta)
            u_tilde3 = self.filterEqn(u_tilde2, delta)
            DF = Expression('3.0*u_tilde1 - 3.0*u_tilde2 + u_tilde3', degree = degree, u_tilde1 = u_tilde1, u_tilde2 = u_tilde2, u_tilde3 = u_tilde3)

        if N == 3:
            u_tilde1 = self.filterEqn(u_, delta)
            u_tilde2 = self.filterEqn(u_tilde1, delta)
            u_tilde3 = self.filterEqn(u_tilde2, delta)
            u_tilde4 = self.filterEqn(u_tilde3, delta)
            DF = Expression('3.0*u_tilde1 - 3.0*u_tilde2 + u_tilde3 + u_tilde1 - 3.0*u_tilde2 + 3.0*u_tilde3 - u_tilde4', degree = degree, u_tilde1 = u_tilde1, u_tilde2 = u_tilde2, u_tilde3 = u_tilde3, u_tilde4 = u_tilde4)

        # Compute the indicator function
        indicator = Expression('sqrt((a-b)*(a-b))', degree = degree, a = u_, b = DF)
        indicator = interpolate(indicator, Q)
        max_ind = np.amax(indicator.vector().get_local())

        if max_ind < 1:
            max_ind = 1.0

        indicator = Expression('a/b', degree = 2, a = indicator, b = max_ind)
        indicator = interpolate(indicator, Q)

        # Apply the filter with indicator function
        u_bar = TrialFunction(Q)
        v = TestFunction(Q)
        F_filter = v*u_bar*dx + delta*delta*dot(grad(v), indicator*grad(u_bar))*dx - v*u_*dx 

        a_filter = lhs(F_filter)
        L_filter = rhs(F_filter)

        A_filter = assemble(a_filter)
        b_filter = assemble(L_filter)
        
        # if example needs Dirchlet BC (add new classes to above list)
        if self.ex in self.NeedsDirBC:
            self.setDirBC(A_filter,b_filter)
        
        solver = LUSolver(A_filter)
        u_bar = Function(Q)
        solver.solve(u_bar.vector(), b_filter)
        return u_bar, indicator

##################################################################################
##################################################################################

## VISUALIZATION
# plotErr -- plots L2, H1 errors
# plotExtrema -- plots max and min values of solution

    def plotErr(self, L2, H1):

        font = {'family' : 'serif',
                'weight' : 'bold',
                'size'   : 22}

        plt.rc('font', **font)
        plt.rc('text', usetex=True)

        fig = plt.figure()
        plt.semilogy([0,1,2,3,4], L2,'ro-',label = '$L^2$ error')
        plt.semilogy([0,1,2,3,4], H1,'bx--',label = '$H^1$ error')
        r = 10**(4.0)
        b = 10**(-2.0)
        plt.semilogy([0,1,2,3,4],[r, r/2**2, r/4**2, r/8**2, r/16**2],'k:',label = '$o(h^2)$')
        plt.semilogy([0,1,2,3,4],[b, b/2**3, b/4**3, b/8**3, b/16**3],'k--',label = '$o(h^3)$')

        plt.xlabel('Refinement Level, $\ell$')
        plt.ylabel('Error')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax = fig.gca()
        ax.set_xticks(np.arange(0, 5, 1.0))
        ax.set_yticks(10**np.arange(-6, 5, 2.0))

        plt.grid()
        plt.show()
        fig.savefig('Extrema for '+self.ex+':'+self.method+'.png')


    def plotExtrema(self, Extr, Type):
        font = {'family' : 'serif',
                'weight' : 'bold',
                'size'   : 22}

        plt.rc('font', **font)
        plt.rc('text', usetex=True)
        fig = plt.figure() #figsize=(12, 6)

        plt.plot([0,1,2,3,4], Extr,'ro-',label = 'E')
        plt.xlabel('Refinement Level, $\ell$')
        plt.ylabel('$'+Type+'(u_h)$')
        plt.legend(loc='best', borderaxespad=0.)
        plt.grid()
        ax = fig.gca()
        ax.set_xticks(np.arange(0, 5, 1.0))
        ax.set_yticks(np.arange(np.min(Extr), np.max(Extr), 0.2))
        plt.show()

        fig.savefig('Extrema for '+self.ex+':'+self.method+'.png')


##################################################################################
##################################################################################

## SIMULATION EXAMPLES

# Each of the following classes inherits properties from SimADR and they differ by their model parameters
# List of simulation examples:
# - RisingHumpSim: working
# - HeatSim: not working
# - SwirlSim: working
# - TwoSourcesSim: working

class RisingHumpSim(SimADR):
    # Iliescu example from BQ paper
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
        self.saveEvery = 10
                
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
    def __init__(self, method):
        SimADR.__init__(self, method)
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

    def __init__(self, method):
        SimADR.__init__(self, method)
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

    def __init__(self, method):
        SimADR.__init__(self, method)
        self.folder = self.basefolder + self.ex + '/'
        
    def run(self):
        for dt in self.timeSize[1:2]:
            for nx in self.gridSize[0:2]:
                self.setupSim(nx,dt)

class BlankSim(SimADR):
    # How to define NEW example simulation from scratch

    def __init__(self, method, ex, T, velocityExpr, mu, sigma, saveExact, sourceFn, uExact):

        SimADR.__init__(self,method)    # String: method can be Galerk, SUPG, EFR
        self.T = T                      # Scalar: total simulation time, 2*np.pi
        self.velocity = Expression(velocityExpr, degree = SimADR.degree, t = 0)        # Expression: advective velocity vector, Expression(('cos(t)','sin(t)'), degree = SimADR.degree, t = 0)
        self.mu = mu                    # Scalar: reaction coefficient, 0.005
        self.sigma = sigma              # Scalar: diffusivity coefficient, 0.01
        self.ex = ex                    # String: name of directory for pvd files, 'SubfolderSimName'
            
        self.saveExact = False               # Boolean: False if you have no exact solution, True if you have exact solution
        
        self.f_code = sourceFn          # Expression: source function
        self.u_code = uExact            # Expression: exact solution or  boundary condition if no exact solution available

        # String: pathway to simulation pvd files
        self.folder = self.basefolder + self.ex + '/'
        
        
