#!/usr/bin/env python

import numpy as np
from scipy.integrate import odeint, ode, solve_bvp
import matplotlib.pyplot as plt
import itertools
import os

class constants:
    def __init__(self):
        """
        Initialize global constants
        """
        self.rH = 0.18 #day^-1
        #rT = np.log(2)/2 #day^-1
        self.rE = 0.215 #day^-1
        self.bH = 1.0*10**(-9) #cell^-1
        self.bT = 1.02*10**(-9) #cell^-1
        self.bE = 1.0*10**(-7) #cell^-1
        """
        bH += bH*0.1
        bT += bT*0.1
        bE += bE*0.1
        """
        self.sigma = 10**4
        #Original Alpha Values
        self.alphaHT = 4.8*10**(-10) #cell^-1 day^-1
        self.alphaHE = 1.0*10**(-9)
        self.alphaIT = 2.8*10**(-9) #cell^-1 day^-1
        self.betaIT = 3.2*10**(-8)

        self.alphaTH = 1.1*10**(-10)
        self.alphaTI = 1.101*10**(-7)#5.2*10**(-8)
        self.alphaTE = 3.7*10**(-9)
        self.alphaIE = 1.0*10**(-12)
        self.alphaEH = 3.0*10**(-9)


        self.rhoE0 = 922.4#7379#11.7#89
        self.deltaI = 0.007#6.12*10**(-2) #day^-1

        self.K = 1e5
        self.G = 0.8

        #Therapy terms
        self.eT = 0.0691
        self.eI = 0.0497
        #self.eI = 0
        #self.eI = -self.eT/2
        self.eH = self.eT/2
        self.eE = self.eT/2
        self.xiT = 0.89#0.28
        self.xiE = 0.2811
        
        self.tauHalf = 2.
        self.oldCost = 10
        self.minCost = 10
        self.maxIter = 200
        self.maxGuess = 200

        self.t0 = 0
        self.tF = 250
        self.rT = np.log(2)/(self.tauHalf)
        self.tauXi = 19.6 #days
        self.tauE = 1.075#30./24. #days
        self.omegaXi = 10*self.tauXi #days
        self.omegaE = 10*self.tauE #days
        self.lamE = np.log(2)/self.tauE
        self.lamXi = np.log(2)/self.tauXi
        self.lamOE = np.log(2)/self.omegaE
        self.lamOXi = np.log(2)/self.omegaXi


        self.k1 = 5e6# 33514986.30406721 #5e4 * 1e2
        self.k2 = 5e2 * 1e1

        """
        initialize data parameters
        """
        self.t = np.linspace(int(self.t0),int(self.tF),int(1e5))
        self.tD1 = np.zeros(len(self.t))
        self.tD2 = np.zeros(len(self.t))

        self.u1 = np.zeros(len(self.t))
        self.u2 = np.zeros(len(self.t))

        self.decayU1 = self.u1.copy()
        self.decayU2 = self.u2.copy()

        self.xWork = np.zeros((len(self.t), 4))
        self.xLoc = self.xWork.copy()
        self.xGlob = self.xLoc.copy()

        self.yWork = np.zeros((len(self.t), 4))
        self.yLoc = self.yWork.copy()
        self.yGlob = self.yLoc.copy()

        self.costLoc = 1e52
        self.costGlob = 1e52
        self.cost = 1e52

        self.u1Loc = np.zeros(len(self.t))
        self.u2Loc = np.zeros(len(self.t))
        self.u1Glob = np.zeros(len(self.t))
        self.u2Glob = np.zeros(len(self.t))

        self.decayU1Work = np.zeros(len(self.t))
        self.decayU2Work = np.zeros(len(self.t))
        self.decayU1Loc = self.u1Loc.copy()
        self.decayU2Loc = self.u2Loc.copy()
        self.decayU1Glob = self.decayU1Loc.copy()
        self.decayU2Glob = self.decayU2Loc.copy()

        self.phi1Work = np.zeros(len(self.t))
        self.phi2Work = np.zeros(len(self.t))
        self.phi1Loc = np.zeros(len(self.t))
        self.phi2Loc = np.zeros(len(self.t))
        self.phi1Glob = np.zeros(len(self.t))
        self.phi2Glob = np.zeros(len(self.t))
        

        self.tD1Glob = np.zeros(len(self.t))
        self.tD2Glob = np.zeros(len(self.t))


class systemVariables(constants):
    def __init__(self):
        super().__init__()

        """
        initial values of state equations
        """
        self.initX = [1.0*10**(7), 2.5*10**6, 1.2*10**7, 9.55*10**7]
        """
        set the first row of the solution space to the initial values
        """
        self.xWork[0,:] = self.initX 

        """
        final values of adjoint equations
        """
        self.initY = [0, 0, 0, 0]

        """
        set the first row of the solution space to final values
        """
        self.yWork[0,:] = self.initY 

        """
        set control-specific parameters here
        """
        self.u1Multiplier = 1./max(1/(1 - (self.tauE/self.omegaE)) * (np.exp(-self.lamOE*(self.t)) - np.exp(-self.lamE*(self.t))))
        self.u2Multiplier = 1./max(1/(1 - (self.tauXi/self.omegaXi)) * (np.exp(-self.lamOXi*(self.t)) - np.exp(-self.lamXi*(self.t))))

        self.tds1 = []
        self.tds2 = []

        self.tdso1 = []
        self.tdso2 = []

    def stateEqCalc(self,t,y):

        """
        Find index of where time array equals time value
        """
        idx = (np.abs(self.t - t)).argmin()

        """
        Extract dose time
        """
        tD_u1 = self.tD1[idx]
        tD_u2 = self.tD2[idx]

        """
        Set controls to respective values
        """
        du1 = self.decayU1[idx]
        du2 = self.decayU2[idx]
 

        dydt = []
        """
        System of State Equations
        """
        dydt += [self.rH*y[0]*(1-self.bH*y[0]) - self.alphaHT*y[0]*y[2] - self.eH*y[0]*du1]
        dydt += [self.sigma + self.alphaIT*y[1]*y[2]*(1 - self.betaIT*y[2]) - self.deltaI*y[1] + self.eI*y[1]*du1]
        dydt += [self.rT*y[2]*(1-y[2]/(self.K + self.G*y[3])) - self.alphaTI*y[1]*y[2] - self.eT*y[2]*du1]
        dydt += [self.rE*y[3]*(1 - self.bE*y[3]) + (1 - self.xiE*du2)*self.rhoE0*y[2] - self.eE*y[3]*du1]
        return np.array(dydt)

    def adjointEqCalc(self,t,y):
        """
        Find index of where time array equals time value moving backwards
        """
        idx = (np.abs(self.t - t)).argmin()

        """
        Extract dose time from index
        """
        tD_u1 = self.tD1[idx]
        tD_u2 = self.tD2[idx]

        """
        Set controls to respective values
        """
        du1 = self.decayU1[idx]
        du2 = self.decayU2[idx]

        """
        Extract state equation values at that time
        """
        states = []
        for i in range(4):
            states += [self.xWork[idx][i]]
        states = np.array(states)


        dydt = []
        """
        Adjoint Equations
        """
        dydt += [-y[0]*self.rH + 2*y[0]*self.rH*self.bH*states[0] + y[0]*self.alphaHT*states[2] + y[0]*self.eH*du1]
        dydt += [-y[1]*states[2]*self.alphaIT + self.alphaIT*self.betaIT*states[2]*states[2]*y[1] + self.deltaI*y[1] - y[1]*self.eI*du1 + self.alphaTI*states[2]*y[2]]
        dydt += [-1 -self.alphaIT*states[1]*y[1] + 2*self.alphaIT*self.betaIT*states[1]*states[2]*y[1] - self.rT*y[2] + (2*self.rT*states[2]*y[2])/(self.K + self.G*states[3]) + self.alphaTI*states[1]*y[2] + self.eT*y[2]*du1 - (1 - self.xiE*du2)*self.rhoE0*y[3]]
        dydt += [-(self.rT*states[2]*states[2]*y[2])/(self.K + self.G*states[3])**2 - self.rE*y[3] + 2*self.rE*self.bE*states[3]*y[3] + self.eE*du1*y[3]]
        dydt = np.array(dydt)
        dydt[abs(dydt) == float('inf')] = 0
        dydt[np.isnan(dydt)] = 0
        tmp = np.zeros(4)
        for i in range(4):
            if dydt[i] != 0 and abs(dydt[i]) < 1e-9:
                tmp[i] = np.sign(dydt[i])*1e-9
            elif abs(dydt[i]) > 1e9:
                dydt[i] = 1e9
            else:
                tmp[i] = dydt[i]
        dydt = tmp
        del tmp

        return dydt

    def calcCost(self):
        self.cost = np.trapz(self.xWork[:,2] + self.k1*self.decayU1Work + self.k2*self.decayU2Work,x=self.t)

    def guessControls(self, iteration=0):
        """
        Randomly pick 7 discrete dosing times and set u1, u2 to 1
        """
        self.u1[np.random.randint(0,len(self.t)-1,7)] = 1
        oldU1 = self.u1.copy()

        self.u2[np.random.randint(0,len(self.t)-1,7)] = 1
        oldU2 = self.u2.copy()


        """
        Pick sn, qm values based on u1, u2
        """
        ind = np.where(self.u1 != 0)[0]
        tau1 = self.t[ind[0]]
        self.u1[ind[0]:]=1
        
        ind = np.where(self.u2 != 0)[0]
        tau2 = self.t[ind[0]]
        self.u2[ind[0]:]=1

        for i in range(len(self.t)):
            self.tD1[i] = tau1
            self.tD2[i] = tau2
            if oldU1[i] == 1:
                tau1 = self.t[i]
            if oldU2[i] == 1:
                tau2 = self.t[i]
        
        self.decayU1Work = self.u1*(np.exp(-self.lamOE*(self.t - self.tD1)) - np.exp(-self.lamE*(self.t - self.tD1)))
        self.decayU2Work = self.u2*(np.exp(-self.lamOXi*(self.t - self.tD2)) - np.exp(-self.lamXi*(self.t - self.tD2)))

        """
        Normalize
        """
        self.decayU1Work /= max(self.decayU1Work)
        self.decayU2Work /= max(self.decayU2Work)
        

        """
        Plot guesses
        """
        fig = plt.figure(figsize=(13.69,9.27))
        plt.plot(self.t,self.decayU1Work,color='red',label='drug excretion')
        plt.xlabel('Time [d]')
        plt.ylabel('Relative Drug Effect')
        fig.savefig('u1guess_iter{}.png'.format(iteration))

        fig = plt.figure(figsize=(13.69,9.27))
        plt.plot(self.t,self.decayU2Work,color='red',label='drug excretion')
        plt.xlabel('Time [d]')
        plt.ylabel('Relative Drug Effect')
        fig.savefig('u2guess_iter{}.png'.format(iteration))
        plt.close('all')

    def updateControl(self):

        self.tds1 = []
        self.tds2 = []

        self.tdso1 = []
        self.tdso2 = []

        """
        Initialize arrays for control values and delivery time values
        """
        u1 = np.zeros(len(self.t))
        u2 = np.zeros(len(self.t))

        tDList1 = np.zeros(len(self.t))
        tDList2 = np.zeros(len(self.t))

        tD1 = 0
        tD1old = 0
        tD2 = 0
        tD2old = 0

        decayU1 = np.zeros(len(self.t))
        decayU2 = np.zeros(len(self.t))

        u1old = 0
        u2old = 0

        phi1Saving = np.zeros(len(self.t))
        phi2Saving = np.zeros(len(self.t))

        multiplier = 0
        multiplier2 = 0

        """
        Solve for switching functions and delivery time values
        """
        for i in range(len(self.t)):
            tDList1[i] = tD1
            tDList2[i] = tD2
            '''
            phi1
            '''
            phi1 = self.k1 - self.eH*self.xWork[i,0]*self.yWork[i,0] + self.eI*self.xWork[i,1]*self.yWork[i,1] - self.eT*self.xWork[i,2]*self.yWork[i,2] - self.eE*self.xWork[i,3]*self.yWork[i,3]
            phi1Saving[i] = phi1
            '''
            phi2
            '''
            phi2 = self.k2 - self.xiE*self.rhoE0*self.xWork[i,2]*self.yWork[i,3]
            phi2Saving[i] = phi2
            
            if phi1 < 0:
                """
                3 scenarios possible: 
                   1. u1 rising
                   2. u1 = self.u1Multiplier
                   3. u1 falling
                """
                if u1old < 1/(1 - (self.tauE/self.omegaE)) * (np.exp(-self.lamOE*(self.t[i] - tD1)) - np.exp(-self.lamE*(self.t[i] - tD1))):
                    """
                    update u1old only
                    """
                    multiplier = 1
                    u1old = 1/(1 - (self.tauE/self.omegaE)) * (np.exp(-self.lamOE*(self.t[i] - tD1)) - np.exp(-self.lamE*(self.t[i] - tD1)))
                else:
                    """
                    update u1old and tD1
                    """
                    multiplier = 0
                    u1[i] = u1old + 1/(1 - (self.tauE/self.omegaE)) * (np.exp(-self.lamOE*(self.t[i] - tD1)) - np.exp(-self.lamE*(self.t[i] - tD1))) / self.u1Multiplier
                    tD1old = tD1
                    self.tds1 += [self.t[i]]
                    self.tdso1 += [u1[i]]
                    tD1 = self.t[i]
            """
            Must ensure that control is smooth
            """    
            if i > 0 and tD1 != tDList1[i]:
                u1[i] = u1old 
            elif np.any(phi1Saving < 0):
                u1current = 1/(1 - (self.tauE/self.omegaE)) * (np.exp(-self.lamOE*(self.t[i] - tD1)) - np.exp(-self.lamE*(self.t[i] - tD1)))
                if u1old > u1current and u1current > 1/(1 - (self.tauE/self.omegaE)) * (np.exp(-self.lamOE*(self.t[i-1] - tD1)) - np.exp(-self.lamE*(self.t[i-1] - tD1))):
                    u1[i] = u1old + multiplier * u1current / self.u1Multiplier
                    self.tds1 += [self.t[i]]
                    self.tdso1 += [u1[i]]
                else:
                    u1[i] = u1current 
                    u1old = u1current

            decayU1[i] = u1[i]
            try:
                if self.t[i] == self.tds1[-1]:
                    if self.tdso1[-1] != decayU1[i]:
                        self.tdso1[-1] = decayU1[i]
            except IndexError:
                pass

            if phi2 < 0:
                """
                3 scenarios possible: 
                   1. u1 rising
                   2. u1 = self.u1Multiplier
                   3. u1 falling
                """
                if u2old < 1/(1 - (self.tauXi/self.omegaXi)) * (np.exp(-self.lamOXi*(self.t[i] - tD2)) - np.exp(-self.lamXi*(self.t[i] - tD2))):
                    """
                    update u2old only
                    """
                    multiplier2 = 1
                    u2old = 1/(1 - (self.tauXi/self.omegaXi)) * (np.exp(-self.lamOXi*(self.t[i] - tD2)) - np.exp(-self.lamXi*(self.t[i] - tD2)))
                    if len(self.tdso2) == 0:
                        self.tdso2 += [u2[i]]
                        self.tds2 += [self.t[i]]
                elif multiplier2 == 1:
                    """
                    update tD2old and tD2
                    """
                    multiplier2 = 0
                    self.tds2 += [tD2]
                    self.tdso2 += [u2[i]]
                    u2[i] = u2old #+ u2old * (np.exp(-self.lamOXi*(self.t[i] - tD2)) - np.exp(-self.lamXi*(self.t[i] - tD2))) / self.u2Multiplier
                    tD2old = tD2
                    tD2 = self.t[i]
                    u2old = u2[i-1]
            """
            Must ensure that control is smooth
            """    
            if i > 0 and tD2 != tDList2[i]:
                u2[i] = u2old
            elif np.any(phi2Saving < 0):
                u2current = 1/(1 - (self.tauXi/self.omegaXi)) * (np.exp(-self.lamOXi*(self.t[i] - tD2)) - np.exp(-self.lamXi*(self.t[i] - tD2)))
                if u2old > u2current and u2current > 1/(1 - (self.tauXi/self.omegaXi)) * (np.exp(-self.lamOXi*(self.t[i-1] - tD2)) - np.exp(-self.lamXi*(self.t[i-1] - tD2))):
                    if u2[i-1] != 1./self.u2Multiplier:
                        u2[i] = u2old + u2old * u2current / (self.u2Multiplier * 1/(1 - (self.tauXi/self.omegaXi)) )
                    else:
                        u2[i] = 1./self.u2Multiplier
                        self.tdso2 += [u2[i]]
                        self.tds2 += [self.t[i]]
                    if multiplier2 == 0:
                        self.tdso2 += [u2[i]]
                        self.tds2 += [self.t[i]]
                        multiplier2 = 2
                else:
                    u2[i] = u2current 
                    # u2old = u2current
            if self.u2Multiplier*u2[i] > 1.0 and phi2 < 0:
                u2[i] = 1.0/self.u2Multiplier
                self.tdso2 += [u2[i]]
                self.tds2 += [self.t[i]]
                u2old = u2[i]
            decayU2[i] = u2[i]
            try:
                if self.t[i] == self.tds2[-1]:
                    if self.tdso2[-1] != decayU2[i]:
                        self.tdso2[-1] = decayU2[i]
            except IndexError:
                pass

        self.tD1 = tDList1
        del tDList1

        self.tD2 = tDList2
        del tDList2

        self.decayU1Work = self.u1Multiplier*decayU1
        self.decayU2Work = self.u2Multiplier*decayU2

        self.tdso1 = self.u1Multiplier*np.array(self.tdso1).copy()
        self.tdso2 = self.u2Multiplier*np.array(self.tdso2).copy()

        if max(self.decayU2Work) > 1:
            self.tdso2 /= max(self.decayU2Work)
            self.decayU2Work /= max(self.decayU2Work)
 
        if max(self.decayU1Work) > 1:
            self.tdso1 /= max(self.decayU1Work)
            self.decayU1Work /= max(self.decayU1Work)

        
        self.decayU1 = self.decayU1Work.copy()
        self.decayU2 = self.decayU2Work.copy()

        self.phi1Work = phi1Saving.copy()
        del phi1Saving

        self.phi2Work = phi2Saving.copy()
        del phi2Saving



class initializeSystem(systemVariables):
    def __init__(self):
        super().__init__()
        
        self.tdso1Loc = self.tdso1.copy()
        self.tds1Loc = self.tds1.copy()
        
        self.tdso2Loc = self.tdso2.copy()
        self.tds2Loc = self.tds2.copy()

        self.tdso1Glob = self.tdso1.copy()
        self.tds1Glob = self.tds1.copy()
        
        self.tdso2Glob = self.tdso2.copy()
        self.tds2Glob = self.tds2.copy()

        self.greatDeluge = []
        self.iterats = []
        self.tumorMins = []
        self.greatDelugeLocal = []

    def clear(self):
        self.xWork = np.zeros((len(self.t), 4))
        self.u1 = np.zeros(len(self.t))
        self.u2 = np.zeros(len(self.t))
        self.yWork = np.zeros((len(self.t), 4))
        self.greatDelugeLocal = []

    def reinitialize(self):
        self.xWork[0,:] = self.initX
        self.yWork[0,:] = self.initY 
        
    def updateLocals(self):
        self.u1Loc = self.u1.copy()
        self.u2Loc = self.u2.copy()

        self.tD1Loc = self.tD1.copy()
        self.tD2Loc = self.tD2.copy()

        self.decayU1Loc = self.decayU1Work.copy()
        self.decayU2Loc = self.decayU2Work.copy()

        self.xLoc = self.xWork.copy()
        self.yLoc = self.yWork.copy()

        self.tds1Loc = self.tds1.copy()
        self.tdso1Loc = self.tdso1.copy()

        self.tds2Loc = self.tds2.copy()
        self.tdso2Loc = self.tdso2.copy()

        self.phi1Loc = self.phi1Work.copy()
        self.phi2Loc = self.phi2Work.copy()

        self.costLoc = self.cost

    def updateGlobals(self):
        self.u1Glob = self.u1Loc.copy()
        self.u2Glob = self.u2Loc.copy()

        self.tD1Glob = self.tD1Loc.copy()
        self.tD2Glob = self.tD2Loc.copy()

        self.decayU1Glob = self.decayU1Loc.copy()
        self.decayU2Glob = self.decayU2Loc.copy()

        self.xGlob = self.xLoc.copy()
        self.yGlob = self.yLoc.copy()

        self.tds1Glob = self.tds1Loc.copy()
        self.tdso1Glob = self.tdso1Loc.copy()

        self.tds2Glob = self.tds2Loc.copy()
        self.tdso2Glob = self.tdso2Loc.copy()

        self.phi1Glob = self.phi1Loc.copy()
        self.phi2Glob = self.phi2Loc.copy()

        self.costGlob = self.costLoc


    def solveODE(self, x0, t0, aTol=1e-5, rTol=1e-5, equation=None):
        """
        lsoda was chosen because, although Runge-Kutta works, it is not ideal for stiff
        problems. lsoda automatically switches between the Adams method for
        non-stiff problems and a backward differentiation formulas (BDF) method for stiff
        problems. This makes it more robust moving forward.
        """

        """
        Grab current working system of equations
        """
        if equation == self.stateEqCalc:
            x = self.xWork.copy()
            t = self.t.copy()
        elif equation == self.adjointEqCalc:
            x = self.yWork.copy()
            t = self.t.copy()
            t = t[::-1]

        m = 10 #power of iteration
        N = 2**m #number of steps
        """
        atol : float or sequence absolute tolerance for solution
        rtol : float or sequence relative tolerance for solution
        """

        """
        Calculate ode and integrate steps
        """
        #dop853, dopri5, lsoda, vode
        z = ode(equation).set_integrator("lsoda",atol = aTol,rtol = [aTol, aTol, aTol, aTol],nsteps=N,max_step=4*1e5)
        z.set_initial_value(x0, t0)
        for i in range(1, t.size):
            a = z.integrate(t[i])
            x[i, :] = a # get one more value, add it to the array
        del z

        """
        Update working system of equations
        """
        if equation == self.stateEqCalc:
            self.xWork = x.copy()
        elif equation == self.adjointEqCalc:
            self.yWork = x.copy()
        del x

    def plotLocals(self, iteration=0):
        ind = np.where(self.greatDeluge == min(self.greatDeluge))[0][0]

        fig, ax1 = plt.subplots(figsize=(22,10))
        ax2 = ax1.twinx()
        ax1.scatter(self.iterats[ind], self.greatDeluge[ind], s=300, facecolors='none', edgecolors='blue',label='Globally Optimal Solution')
        ax1.scatter(self.iterats, self.greatDeluge,s=200,color='black',marker='+',label='Locally Optimal Cost')
        ax2.scatter(self.iterats, self.tumorMins,s=200,color='red',marker='x',label='Locally Optimal Tumor Count')
        ax1.set_xlabel('Iteration Number',fontsize=25)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.tick_params(axis='both', which='minor', labelsize=20)
        ax1.set_ylabel('Cost',fontsize=25)
        ax2.set_ylabel('Final Tumor Count [Num. of Cells]',fontsize=25)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(axis='both', which='minor', labelsize=20)
        ax2.set_ylim([min(self.tumorMins)-0.05*min(self.tumorMins)-0.1,max(self.tumorMins)+0.05*max(self.tumorMins)+1])
        plt.xticks(fontsize=20)
        fig.legend(bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes,fontsize=20)
        fig.savefig('localCostSoln_iter{}.png'.format(iteration))


        fig = plt.figure(figsize=(13.69,9.27))
        plt.ticklabel_format(axis='y', style='sci')
        plt.rc('font', **{'size':'18'})
        plt.subplot(2,2,1)
        plt.plot(self.t,self.xLoc[:,0],color='blue',label = 'Host')
        plt.xticks(fontsize = 18)
        plt.xlabel('(a)',fontsize=18)
        plt.rc('font', **{'size':'18'})
        plt.yticks(fontsize = 18)
        plt.title('Host Cells')
        plt.rc('font', **{'size':'18'})
        plt.subplot(2,2,2)
        plt.plot(self.t,self.xLoc[:,1],color='purple',label = 'Immune')
        plt.xticks(fontsize = 18)
        plt.xlabel('(b)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.ticklabel_format(axis='y', style='sci',scilimits=(-2,2))
        plt.rc('font', **{'size':'18'})
        plt.title('Immune Cells')
        plt.subplot(2,2,3)
        plt.plot(self.t,self.xLoc[:,2], color='green',label = 'Tumor')
        plt.xticks(fontsize = 18)
        plt.xlabel('(c)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.rc('font', **{'size':'18'})
        plt.title('Tumor Cells',fontsize = 18)
        plt.subplot(2,2,4)
        plt.plot(self.t,self.xLoc[:,3], color='red', label = 'Endothelial')
        plt.xticks(fontsize = 18)
        plt.xlabel('(d)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.rc('font', **{'size':'18'})
        plt.title('Endothelial Cells')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
        fig.text(0.5, 0.04, 'Time [d]', ha='center', fontsize=20)
        fig.text(0.04, 0.5, 'No. of Cells', va='center', rotation='vertical',fontsize = 20)
        fig.savefig('localState_iter{}.png'.format(iteration))



        """
        Plot controls
        """
        fig = plt.figure(figsize=(13.69,9.27))
        plt.scatter(self.tds1Loc,self.tdso1Loc,color='black')
        plt.plot(self.t,self.decayU1Loc,color='red',label='drug excretion')
        plt.ylim([-0.05,1.05])
        plt.xlabel('Time [d]',fontsize=25)
        plt.ylabel('Relative Drug Effect',fontsize=25)
        fig.savefig('local_u1_iter{}.png'.format(iteration))

        fig = plt.figure(figsize=(13.69,9.27))
        plt.scatter(self.tds2Loc,self.tdso2Loc,color='black')
        plt.plot(self.t,self.decayU2Loc,color='red',label='drug excretion')
        plt.xlabel('Time [d]',fontsize=25)
        plt.ylabel('Relative Drug Effect',fontsize=25)
        plt.ylim([-0.05,1.05])
        fig.savefig('local_u2_iter{}.png'.format(iteration))

        fig = plt.figure(figsize=(13.69,9.27))
        plt.scatter(self.tds1,self.tdso1,color='black')
        plt.plot(self.t,self.decayU1,color='red',label='drug excretion')
        plt.ylim([-0.05,1.05])
        plt.xlabel('Time [d]',fontsize=25)
        plt.ylabel('Relative Drug Effect',fontsize=25)
        fig.savefig('local_state_u1_iter{}.png'.format(iteration))

        fig = plt.figure(figsize=(13.69,9.27))
        plt.scatter(self.tds2,self.tdso2,color='black')
        plt.plot(self.t,self.decayU2,color='red',label='drug excretion')
        plt.xlabel('Time [d]',fontsize=25)
        plt.ylabel('Relative Drug Effect',fontsize=25)
        plt.ylim([-0.05,1.05])
        fig.savefig('local_state_u2_iter{}.png'.format(iteration))

        fig = plt.figure(figsize=(13.69,9.27))
        plt.ticklabel_format(axis='y', style='sci')
        plt.rc('font', **{'size':'18'})
        plt.subplot(2,2,1)
        plt.plot(self.t,self.yLoc[:,0],color='blue',label = r'$\lambda_1$')
        plt.xticks(fontsize = 18)
        plt.xlabel('(a)',fontsize=18)
        plt.rc('font', **{'size':'18'})
        plt.yticks(fontsize = 18)
        plt.title(r'$\lambda_1$')
        plt.rc('font', **{'size':'18'})
        plt.subplot(2,2,2)

        plt.plot(self.t,self.yLoc[:,1],color='purple',label = r'$\lambda_2$')
        plt.xticks(fontsize = 18)
        plt.xlabel('(b)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.ticklabel_format(axis='y', style='sci',scilimits=(-2,2))
        plt.rc('font', **{'size':'18'})
        plt.title(r'$\lambda_2$')
        plt.subplot(2,2,3)

        plt.plot(self.t,self.yLoc[:,2], color='green',label = r'$\lambda_3$')
        plt.xticks(fontsize = 18)
        plt.xlabel('(c)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.rc('font', **{'size':'18'})
        plt.title(r'$\lambda_3$',fontsize = 18)
        plt.subplot(2,2,4)

        plt.plot(self.t,self.yLoc[:,3], color='red', label = r'$\lambda_4$')
        plt.xticks(fontsize = 18)
        plt.xlabel('(d)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.rc('font', **{'size':'18'})
        plt.title(r'$\lambda_4$')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
        fig.text(0.5, 0.04, 'Time [d]', ha='center', fontsize=20)
        fig.text(0.04, 0.5, 'No. of Cells', va='center', rotation='vertical',fontsize = 20)
        fig.savefig('localAdjoint1_iter{}.png'.format(iteration))
        plt.close('all')




        zero_crossings = np.where(np.diff(np.sign(self.phi1Work)))[0]

        yMin = np.min(self.phi1Work) + 0.1*np.min(self.phi1Work)
        yMax = np.max(self.phi1Work) + 30*np.max(self.phi1Work)

        fig = plt.figure(figsize=(13.69,9.27))
        plt.plot(self.t,self.phi1Work,color='black',label='drug excretion')
        if self.phi1Work[0] < 0:
            plt.vlines(self.t[0],yMin,yMax,colors='blue',linestyles='dashed')
        for hh in range(len(zero_crossings)):
            if self.phi1Work[zero_crossings[hh]-1] < 0:
                plt.vlines(self.t[zero_crossings[hh]],yMin,yMax,colors='red',linestyles='dashed')
            elif self.phi1Work[zero_crossings[hh]-1] > 0:
                plt.vlines(self.t[zero_crossings[hh]],yMin,yMax,colors='blue',linestyles='dashed')
        plt.xlabel('Time [d]')
        plt.ylabel(r'$\phi_1$(t)')
        fig.savefig('local_phi1New_iter{}.png'.format(iteration))


        """
        phi2
        """
        zero_crossings = np.where(np.diff(np.sign(self.phi2Work)))[0]

        yMin = np.min(self.phi2Work) + 0.1*np.min(self.phi2Work)
        yMax = np.max(self.phi2Work) + 30*np.max(self.phi2Work)

        fig = plt.figure(figsize=(13.69,9.27))
        plt.plot(self.t,self.phi2Work,color='black',label='drug excretion')
        if self.phi2Work[0] < 0:
            plt.vlines(self.t[0],yMin,yMax,colors='blue',linestyles='dashed')
        for hh in range(len(zero_crossings)):
            if self.phi2Work[zero_crossings[hh]-1] < 0:
                plt.vlines(self.t[zero_crossings[hh]],yMin,yMax,colors='red',linestyles='dashed')
            elif self.phi2Work[zero_crossings[hh]-1] > 0:
                plt.vlines(self.t[zero_crossings[hh]],yMin,yMax,colors='blue',linestyles='dashed')
        plt.xlabel('Time [d]')
        plt.ylabel(r'$\phi_2$(t)')
        fig.savefig('local_phi2New_iter{}.png'.format(iteration))
        plt.close('all')

    def plotGlobals(self, iteration=0):
        """
        Save current controls
        """
        fig = plt.figure(figsize=(13.69,9.27))
        plt.scatter(self.tds1Glob,self.tdso1Glob,c='k')
        plt.plot(self.t,self.decayU1Glob,color='red',label='drug excretion')
        plt.xlabel('Time [d]')
        plt.ylabel('Relative Drug Effect')
        fig.savefig('global_u1_iter{}.png'.format(iteration))

        fig = plt.figure(figsize=(13.69,9.27))
        plt.scatter(self.tds2Glob,self.tdso2Glob,c='k')
        plt.plot(self.t,self.decayU2Glob,color='red',label='drug excretion')
        plt.xlabel('Time [d]')
        plt.ylabel('Relative Drug Effect')
        fig.savefig('global_u2_iter{}.png'.format(iteration))


        ind = np.where(self.greatDeluge == min(self.greatDeluge))[0][0]

        fig, ax1 = plt.subplots(figsize=(22,10))
        ax2 = ax1.twinx()
        ax1.scatter(self.iterats[ind], self.greatDeluge[ind], s=300, facecolors='none', edgecolors='blue',label='Globally Optimal Solution')
        ax1.scatter(self.iterats, self.greatDeluge,s=200,color='black',marker='+',label='Locally Optimal Cost')
        ax2.scatter(self.iterats, self.tumorMins,s=200,color='red',marker='x',label='Locally Optimal Tumor Count')
        ax1.set_xlabel('Iteration Number',fontsize=25)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.tick_params(axis='both', which='minor', labelsize=20)
        ax1.set_ylabel('Cost',fontsize=25)
        ax2.set_ylabel('Final Tumor Count [Num. of Cells]',fontsize=25)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(axis='both', which='minor', labelsize=20)
        ax2.set_ylim([min(self.tumorMins)-0.05*min(self.tumorMins)-0.1,max(self.tumorMins)+0.05*max(self.tumorMins)+1])
        plt.xticks(fontsize=20)
        fig.legend(bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes,fontsize=20)
        fig.savefig('globalCostSoln_iter{}.png'.format(iteration))



        fig = plt.figure(figsize=(13.69,9.27))
        plt.ticklabel_format(axis='y', style='sci')
        plt.rc('font', **{'size':'18'})
        plt.subplot(2,2,1)
        plt.plot(self.t,self.xGlob[:,0],color='blue',label = 'Host')
        plt.xticks(fontsize = 18)
        plt.xlabel('(a)',fontsize=18)
        plt.rc('font', **{'size':'18'})
        plt.yticks(fontsize = 18)
        plt.title('Host Cells')
        plt.rc('font', **{'size':'18'})
        plt.subplot(2,2,2)
        plt.plot(self.t,self.xGlob[:,1],color='purple',label = 'Immune')
        plt.xticks(fontsize = 18)
        plt.xlabel('(b)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.ticklabel_format(axis='y', style='sci',scilimits=(-2,2))
        plt.rc('font', **{'size':'18'})
        plt.title('Immune Cells')
        plt.subplot(2,2,3)
        plt.plot(self.t,self.xGlob[:,2], color='green',label = 'Tumor')
        plt.xticks(fontsize = 18)
        plt.xlabel('(c)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.rc('font', **{'size':'18'})
        plt.title('Tumor Cells',fontsize = 18)
        plt.subplot(2,2,4)
        plt.plot(self.t,self.xGlob[:,3], color='red', label = 'Endothelial')
        plt.xticks(fontsize = 18)
        plt.xlabel('(d)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.rc('font', **{'size':'18'})
        plt.title('Endothelial Cells')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
        fig.text(0.5, 0.04, 'Time [d]', ha='center', fontsize=20)
        fig.text(0.04, 0.5, 'No. of Cells', va='center', rotation='vertical',fontsize = 20)
        fig.savefig('globalState_iter{}.png'.format(iteration))
        plt.close('all')





        fig = plt.figure(figsize=(13.69,9.27))
        plt.ticklabel_format(axis='y', style='sci')
        plt.rc('font', **{'size':'18'})
        plt.subplot(2,2,1)
        plt.plot(self.t,self.yGlob[:,0],color='blue',label = r'$\lambda_1$')
        plt.xticks(fontsize = 18)
        plt.xlabel('(a)',fontsize=18)
        plt.rc('font', **{'size':'18'})
        plt.yticks(fontsize = 18)
        plt.title(r'$\lambda_1$')
        plt.rc('font', **{'size':'18'})
        plt.subplot(2,2,2)

        plt.plot(self.t,self.yGlob[:,1],color='purple',label = r'$\lambda_2$')
        plt.xticks(fontsize = 18)
        plt.xlabel('(b)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.ticklabel_format(axis='y', style='sci',scilimits=(-2,2))
        plt.rc('font', **{'size':'18'})
        plt.title(r'$\lambda_2$')
        plt.subplot(2,2,3)

        plt.plot(self.t,self.yGlob[:,2], color='green',label = r'$\lambda_3$')
        plt.xticks(fontsize = 18)
        plt.xlabel('(c)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.rc('font', **{'size':'18'})
        plt.title(r'$\lambda_3$',fontsize = 18)
        plt.subplot(2,2,4)

        plt.plot(self.t,self.yGlob[:,3], color='red', label = r'$\lambda_4$')
        plt.xticks(fontsize = 18)
        plt.xlabel('(d)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.rc('font', **{'size':'18'})
        plt.title(r'$\lambda_4$')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
        fig.text(0.5, 0.04, 'Time [d]', ha='center', fontsize=20)
        fig.text(0.04, 0.5, 'No. of Cells', va='center', rotation='vertical',fontsize = 20)
        fig.savefig('globalAdjoint_iter{}.png'.format(iteration))
        plt.close('all')

    def finishOptimization(self):
        print('Optimization finished!')
        print("Plotting globally optimal information...")

        self.iterats = np.array(self.iterats)
        self.greatDeluge = np.array(self.greatDeluge)
        self.tumorMins = np.array(self.tumorMins)

        ind = np.where(self.greatDeluge == min(self.greatDeluge))[0][0]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.scatter(self.iterats[ind], self.greatDeluge[ind], s=300, facecolors='none', edgecolors='blue',label='Globally Optimal Solution')
        ax1.scatter(self.iterats, self.greatDeluge,s=200,color='black',marker='+',label='Locally Optimal Cost')
        ax2.scatter(self.iterats, self.tumorMins,s=200,color='red',marker='x',label='Locally Optimal Tumor Count')
        ax1.set_xlabel('Iteration Number',fontsize=25)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.tick_params(axis='both', which='minor', labelsize=20)
        ax1.set_ylabel('Cost',fontsize=25)
        ax2.set_ylabel('Final Tumor Count [Num. of Cells]',fontsize=25)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(axis='both', which='minor', labelsize=20)
        ax2.set_ylim([min(self.tumorMins)-0.05*min(self.tumorMins)-0.1,max(self.tumorMins)+0.05*max(self.tumorMins)+1])
        plt.xticks(fontsize=20)
        fig.legend(loc="upper right",fontsize=20)

        ind = np.where(self.greatDeluge == min(self.greatDeluge))[0]

        """
        Plot controls
        """
        plt.figure()
        plt.scatter(self.tds1Glob,self.tdso1Glob,color='black',label='dosing time')
        plt.plot(self.t,self.decayU1Glob,color='red',label='drug excretion')
        plt.xlabel('Time [d]',fontsize=25)
        plt.ylabel('Relative Drug Effect',fontsize=25)
        plt.ylim([-0.05,1.05])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.figure()
        plt.scatter(self.tds2Glob,self.tdso2Glob,color='black',label='dosing time')
        plt.plot(self.t,self.decayU2Glob,color='red',label='drug excretion')
        plt.xlabel('Time [d]',fontsize=25)
        plt.ylabel('Relative Drug Effect',fontsize=25)
        plt.ylim([-0.05,1.05])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)

        """
        Plot phi1
        """
        zero_crossings = np.where(np.diff(np.sign(self.phi1Glob)))[0]

        yMin = np.min(self.phi1Glob) + 0.1*np.min(self.phi1Glob)
        yMax = np.max(self.phi1Glob) + 30*np.max(self.phi1Glob)

        fig = plt.figure(figsize=(13.69,9.27))
        plt.plot(self.t,self.phi1Glob,color='black',label='drug excretion')
        for hh in range(len(zero_crossings)):
            plt.vlines(self.t[zero_crossings[hh]],yMin,yMax,colors='blue',linestyles='dashed')
        plt.xlabel('Time [d]')
        plt.ylabel(r'$\phi_1$(t)')

        """
        Plot phi2
        """
        zero_crossings = np.where(np.diff(np.sign(self.phi2Glob)))[0]

        yMin = np.min(self.phi2Glob) + 0.1*np.min(self.phi2Glob)
        yMax = np.max(self.phi2Glob) + 30*np.max(self.phi2Glob)

        fig = plt.figure(figsize=(13.69,9.27))
        plt.plot(self.t,self.phi2Glob,color='black',label='drug excretion')
        for hh in range(len(zero_crossings)):
            plt.vlines(self.t[zero_crossings[hh]],yMin,yMax,colors='blue',linestyles='dashed')
        plt.xlabel('Time [d]')
        plt.ylabel(r'$\phi_2$(t)')

        """
        Plot state equations
        """
        fig = plt.figure(figsize=(13.69,9.27))
        plt.ticklabel_format(axis='y', style='sci')
        plt.rc('font', **{'size':'18'})
        plt.subplot(2,2,1)
        plt.plot(self.t,self.xGlob[:,0],color='blue',label = 'Host')
        plt.xticks(fontsize = 18)
        plt.xlabel('(a)',fontsize=18)
        plt.rc('font', **{'size':'18'})
        plt.yticks(fontsize = 18)
        plt.title('Host Cells')
        plt.rc('font', **{'size':'18'})
        plt.subplot(2,2,2)
        plt.plot(self.t,self.xGlob[:,1],color='purple',label = 'Immune')
        plt.xticks(fontsize = 18)
        plt.xlabel('(b)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.ticklabel_format(axis='y', style='sci',scilimits=(-2,2))
        plt.rc('font', **{'size':'18'})
        plt.title('Immune Cells')
        plt.subplot(2,2,3)
        plt.plot(self.t,self.xGlob[:,2], color='green',label = 'Tumor')
        plt.xticks(fontsize = 18)
        plt.xlabel('(c)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.rc('font', **{'size':'18'})
        plt.title('Tumor Cells',fontsize = 18)
        plt.subplot(2,2,4)
        plt.plot(self.t,self.xGlob[:,3], color='red', label = 'Endothelial')
        plt.xticks(fontsize = 18)
        plt.xlabel('(d)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.rc('font', **{'size':'18'})
        plt.title('Endothelial Cells')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
        fig.text(0.5, 0.04, 'Time [d]', ha='center', fontsize=20)
        fig.text(0.04, 0.5, 'No. of Cells', va='center', rotation='vertical',fontsize = 20)

        """
        Plot adjoint equations
        """

        fig = plt.figure(figsize=(13.69,9.27))
        plt.ticklabel_format(axis='y', style='sci')
        plt.rc('font', **{'size':'18'})
        plt.subplot(2,2,1)
        plt.plot(self.t,self.yGlob[:,0],color='blue',label = r'$\lambda_1$')
        plt.xticks(fontsize = 18)
        plt.xlabel('(a)',fontsize=18)
        plt.rc('font', **{'size':'18'})
        plt.yticks(fontsize = 18)
        plt.title(r'$\lambda_1$')
        plt.rc('font', **{'size':'18'})
        plt.subplot(2,2,2)

        plt.plot(self.t,self.yGlob[:,1],color='purple',label = r'$\lambda_2$')
        plt.xticks(fontsize = 18)
        plt.xlabel('(b)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.ticklabel_format(axis='y', style='sci',scilimits=(-2,2))
        plt.rc('font', **{'size':'18'})
        plt.title(r'$\lambda_2$')
        plt.subplot(2,2,3)

        plt.plot(self.t,self.yGlob[:,2], color='green',label = r'$\lambda_3$')
        plt.xticks(fontsize = 18)
        plt.xlabel('(c)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.rc('font', **{'size':'18'})
        plt.title(r'$\lambda_3$',fontsize = 18)
        plt.subplot(2,2,4)

        plt.plot(self.t,self.yGlob[:,3], color='red', label = r'$\lambda_4$')
        plt.xticks(fontsize = 18)
        plt.xlabel('(d)',fontsize=18)
        plt.yticks(fontsize = 18)
        plt.rc('font', **{'size':'18'})
        plt.title(r'$\lambda_4$')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
        fig.text(0.5, 0.04, 'Time [d]', ha='center', fontsize=20)
        fig.text(0.04, 0.5, 'No. of Cells', va='center', rotation='vertical',fontsize = 20)


        print('Relevant information...')
        print('tumor sum: ', np.trapz(self.xGlob[:,2],x=self.t))
        print('u1 sum: ', np.trapz(self.decayU1Glob,x=self.t))
        print('u2 sum: ', np.trapz(self.decayU2Glob,x=self.t))
        print('Tf = ', self.xGlob[-1,2])

        plt.show()






