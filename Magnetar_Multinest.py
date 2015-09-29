import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
import argparse
from pylab import *
import pymultinest
import sys
import random
import os


try:if not os.path.exists("chains"): os.mkdir("chains")
except: print 'chains/ folder exists'

### FUNCTIONS ###

def my_prior(cube,ndims,nparam):
        for i in range(ndims):
                cube[i] = cube[i]*(dim_range[i,1] - dim_range[i,0]) + dim_range[i,0]	### Mapping prior cube onto M,z1,z2,T0,L0,td ranges       

def my_log_likelihood(cube,ndims,nparam):
        L = Magnetar(t,cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6])
        ### chi2 = np.sum(((L-L_data)**2) / L_data_error**2)    # chi square

	chi2 = np.sum(np.log10(L_data / L)**2 / np.log10((L_data+L_data_error)/L)**2)   ### Logarithmic chisquare

        return -chi2 / 2

def A(z,td,T0,z2):
	y=td/(z2*T0)
	return (np.exp(0.5*z**2)*z)/(1+y*z)**2

def Magnetar(t,M,z1,z2,L0,T0,td,a1):
	x=t/td
	f = np.zeros(len(x)) 
	for i in range(len(x)):
		f[i] = integrator(x[i],td,T0,z2)	
	g=np.exp(-0.5*x**2)
	k=M*t**-a1
	h=L0*(1+(t/T0))**-2
	return k+h+((L0*z1)/z2)*g*f

def integrator(td, T0, z2 , upper_limit , lower_limit=0):
        if upper_limit == lower_limit: return 0,0
	n = 50
	while True:
		dz_1 = upper_limit / n	
		dz_2 = upper_limit/ n / 10
		z_array_1 = np.linspace(upper_limit/2/(n+1)  ,  upper_limit-upper_limit/2/(n+1),n)
		z_array_2 = np.linspace(upper_limit/2/(10*n+1)  ,  upper_limit-upper_limit/2/(10*n+1),10*n)

		A_array_1 = A(z_array_1 , td, T0, z2)
		A_array_2 = A(z_array_2 , td, T0, z2)

		f_out_1 = np.sum( A_array_1 * dz_1)
		f_out_2 = np.sum( A_array_2 * dz_2)
		if (np.abs(f_out_1/f_out_2 - 1) > 0.001) or np.isinf(f_out_1) or np.isinf(f_out_2):
			n *= 10			
		else: return f_out_2





####################
### FITTING DATA ###
####################


### LOAD DATA FILE (time (d), lum (erg/s), lum err (erg/s))
in_data = np.loadtxt('data.txt')	#load data
t = in_data[:,0]			#time (days)
L_data = in_data[:,1]			#luminosity (erg/s)
L_data_error = in_data[:,2]		#luminosity error (erg/s)

### Fitting data with MultiNest (via PyMultiNest)
M_range = np.array([1e+45,1e+55])
z1_range = np.array([1,1e+3])
z2_range = np.array([0.01,1.5])
L0_range = np.array([1e+40,1e+50])
T0_range = np.array([10,1e+5])
td_range = np.array([1,30])
a1_range = np.array([0.1,3])
dim_range = np.array([M_range , z1_range, z2_range, L0_range, T0_range, td_range, a1_range])

#########################
### Running MultiNest ###
#########################

n_params = 7
livePoints = 1000   # Increase for a more well defined posterior, decrease for speed
pymultinest.run(my_log_likelihood, my_prior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'model', n_live_points = livePoints,evidence_tolerance=0.5)

###################################
### Reading in MultiNest output ###
###################################

open_stats = open('chains/1-stats.dat','r')
read_stats = open_stats.read()
open_stats.close()

number_of_modes = int(read_stats.split('Total Modes Found:')[1].split()[0])
print 'Posterior has %d modes'%number_of_modes
stats_gaussian_M = np.zeros([number_of_modes,2])
stats_gaussian_z1 = np.zeros([number_of_modes,2])
stats_gaussian_z2 = np.zeros([number_of_modes,2])
stats_gaussian_L0 = np.zeros([number_of_modes,2])
stats_gaussian_T0 = np.zeros([number_of_modes,2])
stats_gaussian_td = np.zeros([number_of_modes,2])
stats_gaussian_a1 = np.zeros([number_of_modes,2])

stats_bestfit_M = np.zeros(number_of_modes)
stats_bestfit_z1 = np.zeros(number_of_modes)
stats_bestfit_z2 = np.zeros(number_of_modes)
stats_bestfit_L0 = np.zeros(number_of_modes)
stats_bestfit_T0 = np.zeros(number_of_modes)
stats_bestfit_td = np.zeros(number_of_modes)
stats_bestfit_a1 = np.zeros(number_of_modes)

for i_modes in range(number_of_modes):
        stats_gaussian_in = read_stats.split('Dim No.')[1 + 3*i_modes].split('\n')
        stats_bestfit_in = read_stats.split('Dim No.')[2 + 3*i_modes].split('\n')
        stats_gaussian_M[i_modes] = stats_gaussian_in[1].split()[1:]
        stats_gaussian_z1[i_modes] = stats_gaussian_in[2].split()[1:]
        stats_gaussian_z2[i_modes] = stats_gaussian_in[3].split()[1:]
        stats_gaussian_L0[i_modes] = stats_gaussian_in[4].split()[1:]
        stats_gaussian_T0[i_modes] = stats_gaussian_in[5].split()[1:]
        stats_gaussian_td[i_modes] = stats_gaussian_in[6].split()[1:]
        stats_gaussian_a1[i_modes] = stats_gaussian_in[7].split()[1:]
        stats_bestfit_M[i_modes] = stats_bestfit_in[1].split()[1]
        stats_bestfit_z1[i_modes] = stats_bestfit_in[2].split()[1]
        stats_bestfit_z2[i_modes] = stats_bestfit_in[3].split()[1]
        stats_bestfit_L0[i_modes] = stats_bestfit_in[4].split()[1]
        stats_bestfit_T0[i_modes] = stats_bestfit_in[5].split()[1]
        stats_bestfit_td[i_modes] = stats_bestfit_in[6].split()[1]
        stats_bestfit_a1[i_modes] = stats_bestfit_in[7].split()[1]
        print 'Mode %d:'%(i_modes+1)
        print '   M bestfit = %s'%(stats_bestfit_M[i_modes])
        print '   z1 bestfit = %s\n\n'%(stats_bestfit_z1[i_modes])
        print '   z2 bestfit = %s\n\n'%(stats_bestfit_z2[i_modes])
        print '   L0 bestfit = %s\n\n'%(stats_bestfit_L0[i_modes])
        print '   T0 bestfit = %s\n\n'%(stats_bestfit_T0[i_modes])
        print '   td bestfit = %s\n\n'%(stats_bestfit_td[i_modes])
        print '   a1 bestfit = %s\n\n'%(stats_bestfit_a1[i_modes])
        

if number_of_modes > 1:
        try: plot_mode = input('Posterior has %d modes. Choose which to plot [%s]: '%(number_of_modes,'/'.join(map(str,range(1,number_of_modes+1))))) - 1
        except:
                print 'Bad input! Now exiting.'
                raise SystemExit(0)
else: plot_mode = 0

### Print values

# B-field

B = np.sqrt(4.2 / (stats_bestfit_L0[i_modes] * 1e-49 * (stats_bestfit_T0[i_modes] * 1e-3))**2)

print '\n\n***************************************'
print 'B-field = ', '%0.2f' %B, 'x 10**15 G'

# Initial spin period

P = np.sqrt(2.05 / (stats_bestfit_L0[i_modes] * 1e-49 * stats_bestfit_T0[i_modes] * 1e-3))

print 'Initial spin period = ', '%0.2f' %P, 'ms'
print '***************************************\n\n'

### PLOT RESULTS


t_plot = np.linspace(0,100,300)						#linspace for smooth Arnett plot
L_fit = Magnetar(t_plot,stats_bestfit_M[plot_mode],stats_bestfit_z1[plot_mode], stats_bestfit_z2[plot_mode], stats_bestfit_L0[plot_mode], stats_bestfit_T0[plot_mode], stats_bestfit_td[plot_mode], stats_bestfit_a1[plot_mode]) 	

plt.figure()
plt.loglog(t_plot,L_fit, color='blue')	
plt.errorbar(t,L_data, yerr=L_data_error, fmt='o', color='blue')	#Fitted data (incl. errorbars)
plt.legend(['Magnetar (fit)','GRB (data)'])				#Arnett model (best-fit)

#plt.xlim([0,100])
#plt.ylim([0,3e+43])
plt.ylabel('$L$ (erg/s)')
plt.xlabel('$t-t_{0}$ (d)')
plt.title('Magnetar model')


### VISUALS ###

plt.savefig('Magnetar_Multinest.pdf')
plt.show()

