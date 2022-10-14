# Program:
#   This program include most function I defined to simulate traffic model
# History:
# 2020/07/30
# ________________________________________________________________________
# import module

import pyfftw
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import pandas as pd
import os
from numpy import linalg as LA
from functools import reduce
import operator
#________________________________________________________________

# Function of creat Gaussian distribution with upper bound and lower bound


def get_truncated_normal(mean=1, sd=0.01, low=0.5, upp=1.5):
    # Usage:
    # W = get_turncated_norma( sd = ? )
    # w = W(totalcarn)
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

# _______________________________________________________________

# Function of Fourier Transformation

# Description:
#   a and b are the parameter that use for Fourier transformation. c and d are
#   the parameter that use for inverse FOurier transformation.


def fft_function(split, function_name, type_name='f'):
    a = pyfftw.empty_aligned(split, dtype='complex128')
    b = pyfftw.empty_aligned(split, dtype='complex128')
    c = pyfftw.empty_aligned(split, dtype='complex128')
    d = pyfftw.empty_aligned(split, dtype='complex128')
    fftw_object = pyfftw.FFTW(a, b)
    ifftw_object = pyfftw.FFTW(c, d, direction='FFTW_BACKWARD')
    if type_name == 'i':
        return ifftw_object(function_name)
    else:
        return fftw_object(function_name)

# _______________________________________________________________
# Function of generating fixed solution

# Description:
#   Step 1: Calaulate the constraint \delta x_i * w_i = A ,and  \sum \delta x_i
#           = length. Then we can get all \delta x_i for all cars.
#   Step 2: Use \delta x_i to get initial position of all cars


def fixed_solution(length, totalcarn, w):

    # Step 1

    denominator = 0
    for i in w:
        denominator += 1/i
    A = length / denominator
    delta_x_i = A/w

    # Step 2

    initial_position = [0]
    for i in range(0, len(delta_x_i)-1):
        initial_position.append(initial_position[len(initial_position)-1]+delta_x_i[i])
    return np.array(initial_position)

# _______________________________________________________________

# Function of creat initial condition

# Description:
#   Step 1: Generate the fix solution position and velocity
#   Step 2: Use the parameter of that fixed point to generate perturbation
#           of the eigenfuncion


def xinit(amplitude, alpha, length, totalcarn, w, init='cos'):
    # Step 1
    x = fixed_solution(length, totalcarn, w)
    delta = []
    for i in range(0, len(x)):
        if i == len(x)-1:
            delta.append(x[0]+length-x[i])
        else:
            delta.append(x[i+1]-x[i])
    v = np.tanh(w*np.array(delta)-2)+np.tanh(2)

    # Step 2
    fprime = 1/np.cosh(w*np.array(delta)-2)**2
    f = fprime[0]
    # initdx = np.zeros(totalcarn)
    # initdv = np.zeros(totalcarn)
    if init == 'cos':
        initdx = np.cos(np.linspace(0, 2*np.pi, totalcarn, endpoint=False))*amplitude
        initdv = -np.sin(np.linspace(0, 2*np.pi, totalcarn, endpoint=False))*amplitude
    elif init == 'sin':
        initdx = np.sin(np.linspace(0, 2*np.pi, totalcarn, endpoint=False))*amplitude
        initdv = np.cos(np.linspace(0, 2*np.pi, totalcarn, endpoint=False))*amplitude

    # initdx[0] = amplitude
    return [x, v, f, initdv, initdx]

# _______________________________________________________________



# Function of creat w with specific Fourier mode distribution
#
# Description:
#   we let only first and second Fourier mode have value
# 
# Note:
#   currently we don't use this since we use random field


def creatw(size, mean, sigma):
    beta = np.random.sample(size//2+1)
    a = np.sqrt(2*np.log(1/(1-beta)))
    a = a*sigma*np.sqrt(size/2)
    a[0] = np.sqrt(2)*a[0]
    a[-1] = np.sqrt(2)*a[-1]
    fftamplitude = np.append(a, np.flipud(a[1:-1]))
    phase = 2*np.pi*np.random.sample(size//2-1)
    phase = np.append(np.append(np.append(np.array([0]), phase), np.array([0])), -1*np.flipud(phase))
    fftw = fftamplitude * np.exp(1j*phase)
    w = np.real(fft_function(size, fftw, 'i'))
    w = w - np.mean(w) + mean
    return w

# _______________________________________________________________

# Function of returning deltax
#
# Description:
#   Calculate delta x


def deltaxfunction(length, function_name):
    delta = np.diff(function_name)
    delta = np.append(delta, function_name[0]+length-function_name[-1])
    # print(len(delta))
    return delta


def deltavfunction(function_name):
    delta = np.diff(function_name)
    delta = np.append(delta, function_name[0]-function_name[-1])
    return delta


def findroot(w, initiallamb):
    lamb = initiallamb 
    while True:
        f = -w-lamb
        multiply = reduce(operator.mul, f)
        multiply += -1*reduce(operator.mul, w)
        dermul = 0
        for i in range(0, len(f)):
            thistimemul = -1
            for j in range(0, len(f)):
                if i != j:
                    thistimemul = thistimemul*f[j]
            dermul += thistimemul
        delta = -multiply/dermul
        lamb += delta
        if abs(delta) < 5*10**(-17):
            break
    return lamb

def eigenvalue(w, totalcarn):
    fftw = fft_function(totalcarn, w)/len(fft_function(totalcarn, w))
    total_array = []
    for k in range(-int(0.5*totalcarn), int(0.5*totalcarn)):
        B_k = []
        for l in range(-int(0.5*totalcarn), int(0.5*totalcarn)):
            if l != 0 and k != 0:
                alpha_l = 2*np.pi*l/totalcarn 
                B_k.append(fftw[k-l]*(np.exp(1j*alpha_l)-1))
        if B_k != []:
            total_array.append(B_k)
    total_array = np.array(total_array)
    w, v = LA.eig(total_array)
    return [w, v]


def find_boundary(eigenvalue, f):
    if type(eigenvalue) == list or type(eigenvalue) == np.ndarray:
        a = np.max(-f*(np.imag(np.delete(eigenvalue, np.where(eigenvalue == 0)))**2)/np.real(np.delete(eigenvalue, np.where(eigenvalue == 0))))
    else:
        a = -f * (np.imag(eigenvalue)**2)/np.real(eigenvalue)
    # print(-f*(np.imag(np.delete(eigenvalue, np.where(eigenvalue == 0)))**2)/np.real(np.delete(eigenvalue, np.where(eigenvalue == 0))))
    return a


def velocity_eigenvalue(w, g, totalcarn, sense, f, eqdx, lamb, f2 ):
    fftw = fft_function(totalcarn, w)/len(fft_function(totalcarn, w))
    fftg = fft_function(totalcarn, g)/len(fft_function(totalcarn, g))
    A = eqdx
    total_array = []
    for k in range(0, totalcarn-1):
        B_k = np.full(2*totalcarn-2, 0, dtype='complex128')
        B_k[totalcarn-1+k] = 1
        total_array.append(B_k)
    # print(total_array)

    for k in range(-int(0.5*totalcarn), int(0.5*totalcarn)):
        B_k = []
        for l in range(-int(0.5*totalcarn), int(0.5*totalcarn)):
            if k != 0 and l != 0:
                alpha_l = 2*np.pi*l/totalcarn 
                B_k.append(sense*f*fftw[k-l]*(np.exp(1j*alpha_l)-1))
        for l in range(-int(0.5*totalcarn), int(0.5*totalcarn)):
            if k != 0 and l != 0:
                alpha_l = 2*np.pi*l/totalcarn 
                B_k.append(lamb*f2*np.exp(-1*A)*fftg[k-l]*(np.exp(1j*alpha_l)-1))
        if B_k != []:
            B_k = np.array(B_k)
            if k >0:
                B_k[int(1.5*totalcarn)+k-2] -= sense
            else:
                B_k[int(1.5*totalcarn)+k-1] -= sense
            B_k = B_k.astype('complex128')
            total_array.append(B_k)
    total_array = np.array(total_array)
    w, v = LA.eig(total_array)
    return [w, v]


def velocity_find_boundary(w, h, totalcarn, f, eqdx, lamb, f2):
    change_time = 0
    a = 1
    delta = 0.01
    previous = 'nothing'
    while change_time <= 5:
        rate = max(np.real(velocity_eigenvalue(w, h, totalcarn, a, f, eqdx, lamb, f2)[0][1:]))
        # print(rate)
        # rate = max(np.real(eigenvalue(w, totalcarn)[0][1:]))
        # print(rate)
        # print(a, z_plus, z_minus)
        # exit()
        # print(rate, change_time)
        # print(delta,a)
        if previous == 'nothing':
            if rate > 0:
                a += delta
                previous = 'unstable'
            else:
                a -= delta
                previous = 'stable'
        else:
            if rate > 0:
                if previous == 'unstable':
                    a += delta
                elif previous == 'stable':
                    delta = delta/2
                    a += delta
                    change_time += 1
                    previous = 'unstable'
            elif rate <= 0:
                if previous == 'stable':
                    a -= delta
                elif previous == 'unstable':
                    delta = delta/2
                    a -= delta
                    change_time += 1
                    previous = 'stable'
        if rate == 0:
            break
    return a


def draw_gaussian(w):
    denote = [w[0]]
    time = [1]
    for i in range(1, len(w)):
        check = True
        for j in range(0, len(denote)):
            if abs(w[i]-denote[j]) <= 0.05:
                time[j] += 1
                check = False
                break 
        if check:
            denote.append(w[i])
            time.append(1)
    return [denote, time]




# # _______________________________________________________________

# def xinittest(amplitude, alpha, length, totalcarn, w, sense):
    # # Step 1
    # x = fixed_solution(length, totalcarn, w)
    # delta = []
    # for i in range(0, len(x)):
        # if i == len(x)-1:
            # delta.append(x[0]+length-x[i])
        # else:
            # delta.append(x[i+1]-x[i])

    # v = np.tanh(w*np.array(delta)-2)+np.tanh(2)

    # # Step 2
    # fprime = 1/np.cosh(w*np.array(delta)-2)**2
    # f = fprime[0]
    # [eig, eigenfunction] = eigenvalue(w, totalcarn)
    # z = (-sense+np.sqrt(sense**2+4*sense*f*eig))/2
    # realeigenfunction = LA.inv(eigenfunction)
    # perturbation = np.zeros(totalcarn)
    # perturbation[0] = amplitude
    # perturbation = fft_function(totalcarn, perturbation)/totalcarn
    # perturbation = np.flipud(perturbation[1:])
    # # print(realeigenfunction)

    # vperturbeineigenspace = np.dot(realeigenfunction, perturbation)
    # vperturbeineigenspace = z * vperturbeineigenspace 
    # print(z)
    # print(vperturbeineigenspace)
    # print('eig', eigenfunction)
    # vperturbe = np.dot(eigenfunction, vperturbeineigenspace)
    # print(vperturbe)
    # vperturbe = np.flipud(vperturbe)
    # vperturbe = np.append([0], vperturbe)
    # # vperturbe[len(vperturbe)//2] = np.real(vperturbe[len(vperturbe)//2])
    # # print(vperturbe)
    # initdv = fft_function(totalcarn, vperturbe, 'i')
    # # print(initdv)
    # exit()
    
    # initdx = np.cos(np.linspace(0, 2*np.pi, totalcarn, endpoint=False))*amplitude

    # # initdv = np.zeros(totalcarn)
    # initdv = -np.sin(np.linspace(0, 2*np.pi, totalcarn, endpoint=False))*amplitude
    # # initdx[0] = amplitude
    # return [x, v, f, initdv, initdx]

