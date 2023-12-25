"Mathematical Modelling of Photonic-Phononic Scattering Suppression in Nanophoxonic Waveguide for MWIR frequency range"
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 23:24:05 2022

@author: AnuragSharma
"""

from __future__ import print_function
#from Utilities.SaveAnimation import Video
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from IPython.display import HTML
import time
import pandas as pd
import pickle
#Get GPU Information
print("CUDA is available: {}".format(torch.cuda.is_available()))
print("CUDA Device Count: {}".format(torch.cuda.device_count()))
print("CUDA Device Name: {}".format(torch.cuda.get_device_name(0)))
from qutip import *   # importing the quantum computing package by Stanford folks

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image

# System Parameters
#-----------------------------------
Nm = 25   # Number of mech states
# truncation of photon Hilbert space (if num of phonons participating in the problem exceeds this value, then the code doesn't work!!!)

# eta*Omega=etaPrime, Gamma_0=xi*omega, Omega=omega/xi
omega = 3.3
xi = 45
u0 = np.sqrt(1/(2*1*(omega/xi)))  #  u0=sqrt(hbar/(2*M*Omega))
psi = 0.707*(np.exp(1j*np.pi/4)*coherent(10,-2.0j)+np.exp(-1j*np.pi/4)*coherent(10,2j))
# Operators
#----------
# we recall that in qutip sigmam()=[[0. 0.], [1. 0.]]; we define [1,0,....,0] to be the lowest energy state

id1 = qeye(3)
id2 = tensor(qeye(Nm), qeye(Nm))
id = tensor(id1, id2)

M1 = Qobj(np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 0]]))
M2 = Qobj(np.array([[0, 1,  1],
                    [1, 0, 1],
                    [1, 1, 0]]))
#M3 = Qobj(np.array([[0, 1, -1],
                  #  [-1, 0, 1],
                  #  [1, -1, 0]]))
# M3 is the analytically modelled thermal density matrix
M3 = Qobj(np.array([[0.363+1j*0.166, 0, 0],
                    [0, 0.357+(-1j)*0.013, 0],
                    [0, 0, 0.280-01j*0.153]]))
M1 = tensor(M1, id2)
M2 = tensor(M2, id2)
M3 = tensor(M3, id2)

ax = tensor(id1, tensor(destroy(Nm), qeye(Nm)))
ay = tensor(id1, tensor(qeye(Nm), destroy(Nm)))

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))
axes[0, 0].set_xlabel(r"$\eta^{\prime}$")
#axes[0].set_xlim(left=-1.5, right=0.75)
axes[0, 0].set_ylabel(r"atom 1 momentum")

axes[0, 1].set_xlabel(r"$\eta^{\prime}$")
axes[0, 1].set_ylabel(r"atom 2 momentum")

axes[1, 0].set_xlabel(r"$\eta^{\prime}$")
axes[1, 0].set_ylabel(r"atom 3 momentum")

axes[1, 1].set_xlabel(r"$\eta^{\prime}$")
axes[1, 1].set_ylabel(r"# of phonons in GS")

axes[0, 2].set_xlabel(r"$\eta^{\prime}$")
axes[0, 2].set_ylabel(r"$|<GS|\hat{a}_x|GS>|$")
axes[1, 2].set_xlabel(r"$\eta^{\prime}$")
axes[1, 2].set_ylabel(r"$|<GS|\hat{a}_y|GS>|$")

etaPrime = 0
etaPrime_num = 150
etaPrime_array = np.zeros(etaPrime_num)
num_of_energies = 3

atom1_momentum = np.zeros(etaPrime_num)
atom2_momentum = np.zeros(etaPrime_num)
atom3_momentum = np.zeros(etaPrime_num)

num_of_phonons_in_GS = np.zeros(etaPrime_num)
annih_oper_x_ev_of_phonons_in_GS = np.zeros(etaPrime_num)
annih_oper_y_ev_of_phonons_in_GS = np.zeros(etaPrime_num)
energies_result = np.zeros((etaPrime_num, num_of_energies))

for i in range(etaPrime_num):
    etaPrime_array[i] = etaPrime
    print(etaPrime)
    Heff = (omega/xi)*ax.dag()*ax + (omega/xi)*ay.dag()*ay + 2*(etaPrime**2*xi/omega)/3*id - np.sqrt(3)*(xi*omega)/2*M1 - \
           etaPrime/np.sqrt(6)*(1j*ax-1j*ax.dag())*M2 + etaPrime/np.sqrt(6)*(ay-ay.dag())*M3
    eigenenergies, eigenstates = Heff.eigenstates()

    p1 = 1j*(np.sqrt(3)*(ax.dag()-ax)+3*(ay-ay.dag()))/6
    p2 = 1j*(np.sqrt(3)*(ax.dag()-ax)-3*(ay-ay.dag()))/6
    p3 = 1j*np.sqrt(2)*(ax-ax.dag())/np.sqrt(6)
    atom1_momentum[i] = expect(p1, eigenstates[1])
    atom2_momentum[i] = expect(p2, eigenstates[1])
    atom3_momentum[i] = expect(p3, eigenstates[1])
    num_of_phonons_in_GS[i] = expect(ax.dag()*ax+ay.dag()*ay, eigenstates[1])
    annih_oper_x_ev_of_phonons_in_GS[i] = np.abs(expect(ax, eigenstates[1]))
    annih_oper_y_ev_of_phonons_in_GS[i] = np.abs(expect(ay, eigenstates[1]))
    energies_result[i, :] = eigenenergies[:num_of_energies]
    etaPrime += 0.01

axes[0, 0].plot(etaPrime_array, atom1_momentum, '.')
axes[0, 1].plot(etaPrime_array, atom2_momentum, '.')
axes[1, 0].plot(etaPrime_array, atom3_momentum, '.')
axes[1, 1].plot(etaPrime_array, num_of_phonons_in_GS, '.')
axes[0, 2].plot(etaPrime_array, annih_oper_x_ev_of_phonons_in_GS, '.')
axes[1, 2].plot(etaPrime_array, annih_oper_y_ev_of_phonons_in_GS, '.')

figEnergies = plt.figure(2)
plt.plot(etaPrime_array, energies_result, '.')
plt.show()

