# Anderson_Project

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import pylab as plot
import random as ran 
from numpy import linalg as LA
from scipy import linalg as LAg
import os
#Going to write a class which evolves Rho using the linbland equation method. 

class Density_Matrix_Evolution: 

	def __init__(self, dim = 200, rel_t = 0.1, t_step = 0.01, total_steps = 10000, nom_plots = 6, gama=1e-5 ):
		self.__dim = dim # dimensions of the crystal 
		self.__rel_t = rel_t # the fraction between the random energies and the transmission factor t. 
		self.__t_step = t_step # time step 
		self.__total_steps = total_steps # total number of steps 
		self.__nom_plots = nom_plots # number of p
		self.__gama = gama

	def dim(self):
		return self.__dim

	def rel_t(self):
		return self.__rel_t

	def t_step(self):
		return self.__t_step

	def total_steps(self):
		return self.__total_steps

	def nom_plots(self):
		return self.__nom_plots

	def gama(self):
		return self.__gama

	def Hamiltonian(self): # function that makes the hamiltonian for the anderson localisation.
		E_k = [] # list random numbers along the diagonals.
		for i in range(self.__dim):
			random = ran.randrange(-100,100,1)
			E_k.append(random*float(self.__rel_t/100))
		Hamil = np.zeros([self.__dim,self.__dim], dtype=complex)
		for i in range(self.__dim): # fill the matrix with the correct values.
			for j in range(self.__dim):
				if i == j:
					Hamil[i][j]+= E_k[i]
				elif i == j+1:
					Hamil[i][j]= float(1)
				elif i+1 == j:
					Hamil[i][j]= float(1)
				else:
					Hamil[i][j] = float(0)
		print('Hamiltonian calculated')
		return(Hamil)

	def GS_Density_Matrix(self): # function to create the initial density matrix for a central groundstate.
		Hamil = self.Hamiltonian()
		#DIAGONALISING THE HAMILTONIAN: 
		w,v = LA.eig(Hamil) # w is a list of the eigenvalues, and v is an array of the corresponding eigenvectors.
		# FINDING THE GROUND STATE:
		for i in range(self.__dim):
			if w[i] == min(w):
				index = i
		gs_vec = v[:,index]
		gs_sq =[]
		Rho_0 = np.zeros([self.__dim,self.__dim], dtype =complex) + np.outer(gs_vec, gs_vec) # calculating the denisty matrix.
		argMax = np.argmax(np.diag(np.real(Rho_0))) # finding the central index
		if int(self.__dim*0.4) < argMax < int(self.__dim*0.6): # checking the gs is central (to best observe dynamics)
			print('Rho calculated')
			return(Rho_0, v, w, Hamil)
		else:
			return(self.GS_Density_Matrix())

	def Rho_Evol_and_Plot(self): #function to evolve and to store the density matricies.
		if not os.path.exists('Plots'): # This directory will store all of the Matricies i need for analysis. This will make all other anaylsis forms much easier. 
				os.makedirs('Plots')
		if not os.path.exists('Plots/Rho_Evolution'): # This directory will store all of the Matricies i need for analysis. This will make all other anaylsis forms much easier. 
				os.makedirs('Plots/Rho_Evolution')
		Results = self.GS_Density_Matrix()
		w =  Results[2]
		Hamil =  Results[3]
		v = Results[1]
		Rho_0 = Results[0]
		S_cross = np.transpose(v)
		Mat = np.matmul(S_cross, Hamil)
		Diag = np.diagflat(w)
		evol = np.dot(np.dot(v,LAg.expm(-1j*Diag*self.__t_step)),v.transpose().conjugate())
		evol_t = evol.transpose().conjugate()
		dephase = np.zeros([self.__dim,self.__dim], dtype=complex)
		for i in range(self.__dim): # fill the matrix with the correct values.
			for j in range(self.__dim):
			      dephase[i][j] = np.exp(-1*self.__gama * (i-j)**2 * self.__t_step )
		#Now evolving and storing:	      
		print('Matrix Multiplication and Plotting Underway')
		plt.figure()
		for i in range (self.__total_steps):
			if i%(self.__total_steps/(self.__nom_plots)) ==0:
				plt.plot(range(self.__dim), np.diag(np.real(Rho_0)), label= i) #taking real aprt to avoid type errors, but diagonal should be real anyway.
			Rho_0 =  dephase * np.dot(np.dot(evol,Rho_0),evol_t)
		print('Rho Evolved, Plot completed')
		#now doing the plotting: 	
		plt.ylabel(r'$\rho_{{i}{i}} $', size =15)
		plt.xlabel(r'$site \ number $', size =15)
		plt.legend(title = r'$Time \ Step $')
		plt.savefig('Plots/Rho_Evolution/Evolution_dim_%s_rel_t_%s_t_step_%s_total_steps_%s_gama_%s.pdf'%(self.__dim, self.__rel_t, self.__t_step, self.__total_steps, self.__gama))
		plt.show()	

	def Tilt_Hamiltonian(self, sigma=0.5): # H = sum(alpha*k*|k><k|), the sigma of the distribution is an argument of the function.
		E_k = [] # list random numbers with a gaussian distribution 
		alpha = np.random.normal(scale=sigma)
		for i in range(self.__dim):
			E_k.append(alpha *i)
		Hamil = np.zeros([self.__dim,self.__dim], dtype=complex)
		for i in range(self.__dim): # fill the matrix with the correct values.
			for j in range(self.__dim):
				if i == j:
					Hamil[i][j]+= E_k[i]
				else:
					Hamil[i][j] = 0.
		print('Tilt Hamiltonian calculated')
		return(Hamil)

	def Simple_Density_Matrix(self): #defining a density matrix with every element=1.


	def Averaging_Method_1(self, Density_Matrix = 'simple', Hamiltonian = 'Tilt_Hamiltonian'): 
	# averaging method based on evolving rho for one timestep with various alphas then taking an average.
	# Then the process is repeated with that average being used as the new initial state.
	# This is going to be done initially with the tilt hamiltonian. 

