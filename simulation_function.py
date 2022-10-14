# Program:
# 	This program has three kinds of simulation way to simulate the traffic jam
# History:
# 20200524
# _______________________________________________________________

import function as myf
import numpy as np


# _______________________________________________________________
# Function of evolution function
#
# Description:
#   Use real "nonlinear"  equation to evolve the system
#   Step1: Use delta x to calculate acceleration of each car
#   Step2: Use Euler method to evolve the system
#   Step3: Record the data of first mode, and hystersis data


def animate_real_nonlinear(w, originx, x, v, length, split,  sense, simulationtime, RK4=False):

    # Step 1 
    #
    # =================================================

    deltax = myf.deltaxfunction(length, x)
    xa = sense * (np.tanh(w*deltax-2)+np.tanh(2) - v)
    dt = np.min(deltax/v)*0.01

    # Step 2
    #
    # ================================================

    if RK4:
        k1x = v
        k1v = sense * (np.tanh(w*deltax-2)+np.tanh(2) - v)
        k2x = v + (dt * k1v * 0.5)
        k2v = sense * (np.tanh(w*myf.deltaxfunction(length, x+(0.5*dt*k1x))-2)+np.tanh(2) - (v+(0.5*dt*k1v)))
        k3x = v + (dt * k2v * 0.5)
        k3v = sense * (np.tanh(w*myf.deltaxfunction(length, x+(0.5*dt*k2x))-2)+np.tanh(2) - (v+(0.5*dt*k2v)))
        k4x = v + (dt * k3v)
        k4v = sense * (np.tanh(w*myf.deltaxfunction(length, x+(dt*k3x))-2)+np.tanh(2) - (v+(dt*k3v)))
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        v += dt*(k1v+2*k2v+2*k3v+k4v)/6
        v = np.where(v < 0, 0, v)
        y = x - originx

    else:
        x += v * dt
        v += xa * dt
        v = np.where(v < 0, 0, v)
        y = x - originx

    # Step 3
    #
    # ==================================================

    return [x, v, y, deltax, dt]

# _______________________________________________________________
# Function of evolution function
#
# Description:
#   Use real "nonlinear"  equation to evolve the system
#   Step1: Use delta x to calculate acceleration of each car
#   Step2: Use Euler method to evolve the system
#   Step3: Record the data of first mode, and hystersis data


def animate_velocity(w, h, originx, x, v, length, split,  sense, simulationtime, RK4=False):

    # Step 1 
    #
    # =================================================

    deltax = myf.deltaxfunction(length, x)
    deltav = myf.deltavfunction(v)
    dt = np.min(deltax/v)*0.01

    # Step 2
    #
    # ================================================

    if RK4:
        k1x = v
        k1v = sense * (np.tanh(w*deltax-2)+np.tanh(2) - v) + 3*sense*h*np.exp(-1*deltax)*deltav*sense*(np.tanh(deltav*100)+1)/2
        k2x = v + (dt * k1v * 0.5)
        k2v = sense * (np.tanh(w*myf.deltaxfunction(length, x+(0.5*dt*k1x))-2)+np.tanh(2) - (v+(0.5*dt*k1v))) + 3*sense*h*np.exp(-1*myf.deltaxfunction(length, x +(0.5*dt*k1x)))*myf.deltavfunction(v + (dt * k1v * 0.5))*sense*(np.tanh(100*myf.deltavfunction(v + (dt * k1v * 0.5)))+1)/2
        k3x = v + (dt * k2v * 0.5)
        k3v = sense * (np.tanh(w*myf.deltaxfunction(length, x+(0.5*dt*k2x))-2)+np.tanh(2) - (v+(0.5*dt*k2v))) + 3*sense*h*np.exp(-1*myf.deltaxfunction(length, x +(0.5*dt*k2x)))*myf.deltavfunction(v + (dt * k2v * 0.5))*sense*(np.tanh(100*myf.deltavfunction(v + (dt * k2v * 0.5)))+1)/2
        k4x = v + (dt * k3v)
        k4v = sense * (np.tanh(w*myf.deltaxfunction(length, x+(dt*k3x))-2)+np.tanh(2) - (v+(dt*k3v))) + 3*sense*h*np.exp(-1*myf.deltaxfunction(length, x +(dt*k3x)))*myf.deltavfunction(v + (dt * k3v))*sense*(np.tanh(100*myf.deltavfunction(v + (dt * k3v)))+1)/2
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        v += dt*(k1v+2*k2v+2*k3v+k4v)/6
        v = np.where(v < 0, 0, v)
        y = x - originx

    # if RK4:
        # k1x = v
        # k1v = sense * (np.tanh(w*deltax-2)+np.tanh(2) - v) + h*deltav*sense*(np.tanh(deltav*100)+1)/6
        # k2x = v + (dt * k1v * 0.5)
        # k2v = sense * (np.tanh(w*myf.deltaxfunction(length, x+(0.5*dt*k1x))-2)+np.tanh(2) - (v+(0.5*dt*k1v))) + h*myf.deltavfunction(v + (dt * k1v * 0.5))*sense*(np.tanh(100*myf.deltavfunction(v + (dt * k1v * 0.5)))+1)/6
        # k3x = v + (dt * k2v * 0.5)
        # k3v = sense * (np.tanh(w*myf.deltaxfunction(length, x+(0.5*dt*k2x))-2)+np.tanh(2) - (v+(0.5*dt*k2v))) + h*myf.deltavfunction(v + (dt * k2v * 0.5))*sense*(np.tanh(100*myf.deltavfunction(v + (dt * k2v * 0.5)))+1)/6
        # k4x = v + (dt * k3v)
        # k4v = sense * (np.tanh(w*myf.deltaxfunction(length, x+(dt*k3x))-2)+np.tanh(2) - (v+(dt*k3v))) + h*myf.deltavfunction(v + (dt * k3v))*sense*(np.tanh(100*myf.deltavfunction(v + (dt * k3v)))+1)/6
        # x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        # v += dt*(k1v+2*k2v+2*k3v+k4v)/6
        # v = np.where(v < 0, 0, v)
        # y = x - originx
    else:
        x += v * dt
        v += xa * dt
        v = np.where(v < 0, 0, v)
        y = x - originx

    # Step 3
    #
    # ==================================================

    return [x, v, y, deltax, dt]
# _______________________________________________________________

# Function of evolution function
#
# Description:
#   Use real equation to evolve the system
#   Step1: Use delta x to calculate acceleration of each car
#   Step2: Use Euler method to evolve the system
#   Step3: Record the data of first mode

# def animate_real(sense):
    # global y
    # global ydot
    # global f
    # global initvalue
    # global w
    # global dt 
    # global originx
    # global x
    # global v
    # global length
    # global hystv
    # global hystdeltax

    # # Step 1 
    # #
    # # =================================================

    # deltay = []
    # for i in range(0, len(y)):
        # if i == len(y)-1:
            # deltay.append(y[0]-y[i])
        # else:
            # deltay.append(y[i+1]-y[i])
    # deltay = np.array(deltay)
    # ya = sense * f * w * deltay - sense * ydot 

    # # Step 2
    # #
    # # ================================================

    # y += ydot * dt
    # ydot += ya * dt

    # realv = originv+ydot
    # for i in range(0, len(realv)):
        # if realv[i] < 0:
            # realv[i] = 0
    # ydot = realv - originv

    # # Step 3
    # if totaltime % 0.01 <= 0.0001:
        # yplot.append(np.log10(abs(fft_function(y)[1])/len(fft_function(y))))
        # hystdeltax = deltax+deltay
        # hystv = originv + ydot


# _______________________________________________________________

# Function of evolution function
#
# Description:
#   Use all Fourier mode to evolve the system
#   Step1: Use Fourier mode to calculate acceleration of each mode
#   Step2: Use Euler method to evolve the system
#   Step3: Record the data of first mode

# def animate_Fourier(sense, simulation_range):
    # global ffty
    # global fftydot
    # global f
    # global initvalue
    # global w
    # global dt 
    # global totaltime

    # # Step 1
    # fftw = fft_function(w)/len(fft_function(w))
    # ffty_len = simulation_range
    # if ffty_len != int(totalcarn/2):
        # left_bound = - ffty_len
        # right_bound = ffty_len+1
    # else:
        # left_bound = -ffty_len
        # right_bound = ffty_len
    # wbound = int(totalcarn/2)
    # ya = np.zeros(totalcarn, dtype='complex128')
    # for left_y_mode in range(left_bound, right_bound):
        # yai = 0
        # for right_y_mode in range(left_bound, right_bound):
            # if int(left_y_mode - right_y_mode) in range(-wbound, wbound):
                # alpha_l = (np.pi*2*right_y_mode)/totalcarn
                # yai += ffty[right_y_mode]*fftw[left_y_mode - right_y_mode]*sense*f*(np.exp(1j*alpha_l)-1)
                # alpha_l = (np.pi*2*right_y_mode)/totalcarn
                # yai += ffty[right_y_mode]*fftw[left_y_mode - right_y_mode + totalcarn]*sense*f*(np.exp(1j*alpha_l)-1)
            # elif int(left_y_mode - right_y_mode - totalcarn) in range(-wbound, wbound):
                # alpha_l = (np.pi*2*right_y_mode)/totalcarn
                # yai += ffty[right_y_mode]*fftw[left_y_mode - right_y_mode - totalcarn]*sense*f*(np.exp(1j*alpha_l)-1)

        # yai -= sense*fftydot[left_y_mode]
        # ya[left_y_mode] = yai

    # # Step 2
    # ffty += fftydot*dt 
    # fftydot += ya*dt

    # # Step 3
    # if totaltime % 0.01 <= 0.001:
        # yplot.append(np.log10(abs(ffty[1])))
