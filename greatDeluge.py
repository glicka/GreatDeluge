#!/usr/bin/env python

import numpy as np
import os

from greatdeluge import initializeSystem

def main():
    """
    Comment out these lines if changing chemo effect on immune
    system
    """
    # try:
    #     os.mkdir('GreatDelugeEInorm')
    #     os.chdir('GreatDelugeEInorm')
    # except:
    #     os.chdir('GreatDelugeEInorm')

    """
    Initialize system
    """
    system = initializeSystem()
    """
    To adjust chemotherapy effect on immune system, uncomment
    the following relevant lines:
    """
    # system.eI = -system.eT/2  # chemotherapy HURTS immune system
    # try:
    #     os.mkdir('GreatDelugeEIneg')
    #     os.chdir('GreatDelugeEIneg')
    # except:
    #     os.chdir('GreatDelugeEIneg')
    system.eI = 0             # chemotherapy has NO EFFECT on immune system
    try:
        os.mkdir('GreatDelugeEI0')
        os.chdir('GreatDelugeEI0')
    except:
        os.chdir('GreatDelugeEI0')

    tracker = 0
    iteration = 0
    for guess in range(system.maxGuess):
        """
        Initialize U1, U2, Sn, Qm here.
        """
        trackerLocal = 0
        tracker += 1
        if (tracker)%10 == 0:
            break
        """
        Check if local minimum is less than estimated global minimum
        """
        if guess != 0:
            """
            10) Clear old control
            """
            system.clear()

        system.guessControls(iteration=iteration)

        if guess == 0:
            system.updateLocals()
            system.updateGlobals()

        if system.costLoc < system.costGlob or guess == 0:
            tracker = 0
            if guess != 0:

                print('New minimum found!')
                print('previous final T count = ', system.xGlob[-1,2])
                print('new final T count = ', system.xLoc[-1,2])
                print('previous min cost = ', system.costGlob)
                print('new min cost = ', system.costLoc)

                """
                8) Save new estimated global minimum solutions
                """
                system.updateGlobals()

                """
                For some reason, a weird value is put in tdso2Glob not present in tdso2Loc so to correct for that
                we remove the outlier value
                """
                ind = np.where(system.tdso2Glob <= 1)[0]

                system.tds2Glob = np.array(system.tds2Glob)[ind]
                system.tdso2Glob = np.array(system.tdso2Glob)[ind]

                """
                9) Save figures of globally optimal solutions
                """
                system.plotGlobals(iteration=iteration)



        for iters in range(system.maxIter):
            if iters == 0:
                system.costLoc = 1e57
            trackerLocal += 1
            if (trackerLocal)%10 == 0:
                break

            if iters != 0:
                """
                7) Set initial and final conditions
                """
                system.reinitialize()

            """
            1) start with assumed control u and move forward
            """
            system.solveODE(system.initX, system.t0, equation=system.stateEqCalc)
            system.xWork[system.xWork < 0.5] = 1e-50

            """
            2) solve adjoint equations moving backward in time
            """
            system.solveODE(system.initY, system.tF, equation=system.adjointEqCalc)

            """
            3) reverse adjoint equations so that they move forward in time
            """
            system.yWork = system.yWork[::-1]
            system.yWork[np.isinf(system.yWork)] = 0

            """
            4) verify correct convergence of adjoint equations
            """
            if not (system.yWork[-1,:] == system.initY).all():
                print(system.yWork[-1,:])
                print(system.yWork[0,:])
                raise RuntimeError("Adjoint integration failed!")

            """
            5) calculate cost function and fill in minimization lists
            """
            system.calcCost()
            iteration += 1
            
            print('iteration ',iteration)
            print('cost = ', system.cost)

            if iters > 0 and system.cost < system.costLoc:
                system.greatDeluge += [system.cost]#/(1e11)]
                system.greatDelugeLocal += [system.cost]
                system.iterats += [iteration]
                system.tumorMins += [system.xWork[-1,2]]#*1./(1e6)]
                print('local min found!')
                print('old cost = ', system.costLoc)
                print('new cost = ', system.cost)
                print('old T = ', system.xLoc[-1,2])
                print('new T = ', system.xWork[-1,2])
                """
                Reset tracker
                """
                trackerLocal = 0
                
                """
                Save local minimum solutions
                """
                system.updateLocals()

                """
                Save plots of locally optimal solutions
                """
                system.plotLocals(iteration=iteration)

            """
            6) update controls
            """
            system.updateControl()

    """
    11) plot final results
    """
    system.finishOptimization()

if __name__ == "__main__":
    main()