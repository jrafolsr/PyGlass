# Copyright (c) 2018 Joan Ràfols Ribé
# OrcID: https://orcid.org/0000-0003-1256-149X

import numpy as np
from matplotlib import pyplot as plt

class Glass(object):
    """This class creates a glass object for which you can define a thermal history through the different thermal treatments: ramp (either heating or cooling) and annealing. After defining its history, you can then compute the fictive temperature and normalized heat capacity through the TNM model, with an evolution for the relaxation time that can follow the traditional tnm model or the Adam-Gibbs-Vogel model. More to be implemented."""
    
    def __init__(self, parameters = None, model = None):
        if model is None:
            self.set_model('tnm')
        else:
            self.set_model(model)
        self.dTstep = 0.5
        # Initalize all vector to 1-D and no value
        self.dt = np.zeros((0,))
        self.dT = np.zeros((0,))
        self.Tsystem = np.zeros((0,))
        self.Tfictive = np.zeros((0,))
        self.tau = np.zeros((0,))       
        self.currentT = None
        # Initalize a 2-D array to save the index of each treatment (first dimension) and type (second dimension)
        self.treatments = [[],[]]
        self.index_lasttt = 0

        if parameters is None:
            # Default parameters as example
            self.set_parameters([-125, 0.5,1.0, 40638])
        else:
            self.set_parameters(parameters)       
        
    def set_dTstep(self, step):
        """Defines the overall temperature resolution for the calculations"""
        self.dTstep = step
    def set_parameters(self,parameters):
        """Changes the set of parameters p, which needs to be a 4-element list"""
        self.p = parameters
        print('Info: Using the parameters {}'.format(self.p))
    def set_model(self,model):
        """Changes the model"""
        self.model = model
        print('Info: your glass will obey the {}  relaxation model'.format(self.model))
    def ramp(self, rate, T1, T0 = None, step = None):
        """ This method implements a temperature ramp to the created glass. It needs the rate in K/min for the first argument (the sign indicting either a cooling or a heating) and the setpoint temperature (T1).
If its the first thermal treatment in requires also the initial temperature (T0), otherwise it complains. If it's not he first treatment, the initial temperature its the current one.        
        Optional arguments:
            - step: defines the temperature step for this ramp. Otherwise it takes the global one set by the self.dTstep.
        """
        q = rate / abs(rate)
        if q > 0 : print("Info: preparing  heating scan at {:.1g} K/min".format(abs(rate)))
        else: print("Info: preparing  cooling scan at {:.1g} K/min".format(abs(rate)))
        
        if step is None: step = self.dTstep
            
        if self.currentT is None:
            if T0 is None:
                raise ValueError('Temperature limit missing!')
            else:
                if (T1-T0)/(abs(T1-T0)) !=  rate / abs(rate): raise ValueError('Temperature limits incoherent!')
                self.currentT = T0
        if self.currentT < T1 and q < 0: 
            raise ValueError('Glass is now at {:.1f} K, cannot be cooled to {:.1f}'.format(self.currentT, T1))
        elif self.currentT > T1 and q > 0:
            raise ValueError('Glass is now at {:.1f} K, cannot be heated to {:.1f}'.format(self.currentT, T1))
        else:
            if len(self.treatments[0]) == 0:
                # If it's the first scan, I add the initial and final point
                Nsteps = int(abs(self.currentT-T1)/step) + 1
                rampT = np.linspace(self.currentT, T1, Nsteps)
            else:
                # If it's an initiated treatment, I don't add the initial point as it alrady have it
                Nsteps = int(abs(self.currentT-T1)/step)
                rampT = np.linspace(self.currentT + step, T1, Nsteps)
            
            self.dt = np.hstack((self.dt, np.ones((Nsteps,))* step / abs(rate) * 60))
            self.dT = np.hstack((self.dT, np.ones((Nsteps,))*step * q))
            self.Tsystem = np.hstack((self.Tsystem, rampT))

            
            self.treatments[0].append((self.index_lasttt,self.index_lasttt + Nsteps))
            
            if q > 0: 
                self.treatments[1].append('heating of {:.0f} K to {:.0f} K at {:.1e} K/min'.format(self.currentT,T1,abs(rate)))
            else: 
                self.treatments[1].append('cooling of {:.0f} K to {:.0f} K at {:.1e} K/min'.format(self.currentT,T1,abs(rate)))
            
            self.currentT = self.Tsystem[-1]
            self.index_lasttt = self.index_lasttt + Nsteps
    def annealing(self, t0, t1, Nsteps = None):
        """Calculates and adds to the thermal treatment the steps followed by an annealing specified by                             t0 and t1, which produced a equally logspaced vector form 10^t0 to 10^t1 with a number of Nsteps.
        Optional arguments:
             - Nsteps = None: if not specified, by default it takes 5 steps for each  input decade."""
        if Nsteps is None:
            Nsteps = int((abs(t0) + abs(t1)) * 5) + 1
        
        self.dt = np.hstack((self.dt,np.logspace(t0,t1,Nsteps)))
        self.dT = np.hstack((self.dT, np.zeros((Nsteps,))))
        self.Tsystem = np.hstack((self.Tsystem, self.currentT * np.ones((Nsteps,))))
        
        self.treatments[0].append((self.index_lasttt,self.index_lasttt + Nsteps))
        self.treatments[1].append('annealing of {:.1g} s  at {:.0f} K'.format(10**t1, self.currentT))
        self.index_lasttt = self.index_lasttt + Nsteps
    
    def relaxation_time(self, Tsystem, Tfictive):
        """ Calculate the relaxation time using the specified parameters and model, at the moment the agv and tnm mdels are implemented """
        if self.model is 'tnm':
            return np.exp(self.p[0]+(self.p[1] * self.p[3] / Tsystem)+((1-self.p[1]) * self.p[3] / Tfictive))
        elif self.model is 'agv':
            return  10**self.p[3] * np.exp(self.p[0] * self.p[1] / ((Tsystem * (1 - self.p[1] /(Tfictive)))))
        elif self.model is 'test':
            print('Unused model, to be implemented')
            return 1.0
        else:
            raise ValueError('Model {} not implemented nor known\n'.format(self.model))
    def start_treatment(self, model = None, parameters = None):
        """ This function computes the thermal treatment that the user have already specidied using the methods ramp and annealing. The output variables are initialized every time start_treatment is called, so it can be used to compute the treatment with different relaxation time models or parameters."""
        if model is not None: 
            self.model = model
            print('Computing the thermal treatment assuming the {} model'.format(model))
        if parameters is not None:
            self.set_parameters(parameters)
            
        Ntotal = self.index_lasttt
        dttau = np.zeros((Ntotal,))
        
        self.tau = np.zeros((Ntotal,))
        self.Tfictive = np.zeros((Ntotal,))
        self.cpnorm = np.zeros((Ntotal,))
        
        # Initialize first values
        self.Tfictive[0] = self.Tsystem[0]
        self.tau[0] = self.relaxation_time(self.Tfictive[0], self.Tfictive[0])
        self.cpnorm[0] = 1.0 # Initialize the cp, assuming you've started from the liquid.
        Tfinal = self.Tfictive[0]
        
        for i in range(1,Ntotal):
            self.tau[i] = self.relaxation_time(self.Tsystem[i],  self.Tfictive[i-1])
            for j in range(1, i + 1):
                dttau[j] = dttau[j] + self.dt[i] / self.tau[i]
                Tfinal = Tfinal+(self.dT[j]*(1-np.exp(-(dttau[j]**self.p[2]))))
                self.Tfictive[i] = Tfinal
            Tfinal =  self.Tfictive[0]
            if self.dT[i] != 0:
                self.cpnorm[i] = ( self.Tfictive[i] -  self.Tfictive[i-1]) / self.dT[i]
            else:
                self.cpnorm[i] = 0

        self.time = np.cumsum(self.dt)
    def save_history(self, output = 'report.log',ttstep = None):
        """This method can be used to save a report of the whole thermal history or, if specified,
        a particular thermal treatment step. If the output filename is not specified it saves it
        as 'report.log' in the current folder."""
        if ttstep is None:
            imin = 0
            imax = self.index_lasttt
        else:
            imin = self.treatments[0][ttstep][0]
            imax = self.treatments[0][ttstep][1]
        data = np.zeros((imax-imin,5))
        data[:,0] = self.Tsystem[imin:imax]
        data[:,1] = self.cpnorm[imin:imax]
        data[:,2] = self.Tfictive[imin:imax]
        data[:,3] = self.time[imin:imax]
        data[:,4] = self.tau[imin:imax]
        header = self.__print_header__() + (5*'{:^16}\t').format('Tsystem', 'Cpnorm', 'Tfictive','Time','Tau') + '\n'
        header = header + (5*'{:^16}\t').format('K', 'norm', 'K','s','s')
        np.savetxt(output, data,header=header, fmt = '%16.10e', delimiter='\t')
    def plot_report(self):
        """ This method plots a basic report of the current computed thermal treatment """
        if self.currentT is None:
            raise ValueError('No data to plot yet, prepare the glass and run the "experiment"')
        else:
            plt.figure(figsize=(16,4))
            ax = plt.subplot(133)
            plt.title('Time evolution of fictive temperature')
            plt.xlabel('Time (s)')
            plt.ylabel('Temperature (K)')
            ax.semilogx(self.time,self.Tfictive, self.time,self.Tsystem)
            ax.legend(['Fictive temperarure','System temperature'])
            ax2 = plt.subplot(131)
            plt.title('Normalized heat capacity')
            plt.ylabel('Normalized $C_p$ (a.u.)')
            plt.xlabel('Temperature (K)')
            ax2.plot(self.Tsystem,self.cpnorm)
            ax3 = plt.subplot(132)
            plt.title('Temperature evolution of fictive temperature')
            plt.plot(self.Tsystem,self.Tfictive)
            plt.ylabel('Fictive temperarure (K)')
            plt.xlabel('Temperature (K)')
            plt.subplots_adjust(wspace=0.2)
            plt.show()
    def compute_cpe(self, cliquid, cglass, save = False, ttstep = None, output = 'report_cpe.log'):
        """This function 'de-normalizes' the cpnorm using the input and required polynomial functions, 
        either as numpy array or as a list of coefficients.
        
        Optional arguments:
            - save = False: If set to True it prints the values (Tsystem, cpe and Tfictive) in the
                            default logfile 'report_cpe.log'.
            - ttstep = None: if an integer is passed and represents an existing thermal treatment step
                            it only print such step (the last heating for instance).
            - output = 'report_cpe.log': sets the output file name, other than the default.
        """
        self.cpe = np.zeros((self.index_lasttt,))
        pliquid = np.poly1d(cliquid)
        pglass = np.poly1d(cglass)
        self.cpe = pglass(self.Tsystem) + (pliquid(self.Tsystem) - pglass(self.Tsystem))*self.cpnorm
        if save:
            if ttstep is None:
                imin = 0
                imax = self.index_lasttt
            else:
                imin = self.treatments[0][ttstep][0]
                imax = self.treatments[0][ttstep][1]
            data = np.zeros((imax-imin,3))
            data[:,0] = self.Tsystem[imin:imax]
            data[:,1] = self.cpe[imin:imax]
            data[:,2] = self.Tfictive[imin:imax]
            header = self.__print_header__() +\
            '\n pliquid used was: {}\n pglass used was: {}'.format(np.poly1d(cliquid),np.poly1d(cglass))
            header = self.__print_header__() + (3*'{:^16}').format('Tsystem', 'Cpe', 'Tfictive') + '\n'
            header = header + (3*'{:^16}').format('K', 'J/gK', 'K')
            np.savetxt(output, data,header=header, fmt = '%16.10e', delimiter='\t')       
            
    def __print_header__(self):
        """ This method returns the header with all the thermal history of the current
        thermal treatment together with the model and current parameters that can 
        be used later as the headerline of an output file. """
        header = 'model:  {:3}\n'.format(self.model)
        header = header + 'parameters {}\n'.format(self.p)
        for tt in self.treatments[1]:
            header = header + tt + '\n'
        return header
    
if __name__ is '__main__':
    # Very quick example of usage of the Glass class
    tpd = Glass(model = 'tnm',parameters=[-125, 0.7, 0.8, 42638])
    tpd.ramp(-10,270,380) # We cool the glass at 10 K/min t0 270 K sarting from 380 K
    tpd.annealing(-2,5,500) # We perform an annealing of 100000 s using 500 steps
    tpd.ramp(10,380) # We heat our sample at 10 K/min up to 380 K again
    tpd.start_treatment()
    tpd.plot_report()