# Program:
# 	This program use fourier mode to simulate the traffic jam
# History:
# 20200524
# _______________________________________________________________

# import module

import numpy as np
import matplotlib.pyplot as plt
import pyfftw
from scipy.stats import truncnorm
import pandas as pd
from numpy import linalg as LA
import os
import shutil
import function as myf
import simulation_function as myf_simulate
import shutil
# _______________________________________________________________

# Declare variable

split = 2**5
length = split*5
radius = length/(2*np.pi)
amplitude = 0.001
totalcarn = split
alpha = 2*np.pi/totalcarn
deltetime = 20000
uptime = 3000000
unstable_delta_coefficient = 0.02
fig, ax = plt.subplots()
an = np.linspace(0, 2*np.pi, totalcarn)

if os.path.isfile('./pd_data_final.csv'):
    read_final_data = pd.read_csv('./pd_data_final.csv')
    read_final = pd.DataFrame(read_final_data)
    read_final.columns = ['0', 'sigma', 'w2', 'a', 'initialsense', 'perturbation']
    final_sense = read_final_data['a']
else:
    os.system('python3 ./creat_final.py')
    read_final_data = pd.read_csv('./pd_data_final.csv')
    read_final = pd.DataFrame(read_final_data)
    read_final.columns = ['0', 'sigma', 'w2', 'a', 'initialsense', 'perturbation']
    final_sense = read_final_data['a']

# _______________________________________________________________

# Function 

# Main Program

# Step: Call data from numerical.csv file
#
#   current we use random field, thus we don't needto call data
# ==========================================

# read_numerical_data = pd.read_csv('./pd_data_numerical.csv')
# read_numerical_data.columns = [ '0', 'sigma', 'w2', 'a' ]
# numerical_sigma = read_numerical_data['sigma']
# numerical_w2 = read_numerical_data['w2']
# numerical_sense = read_numerical_data['a']

# ==========================================

# Step: Call data from final.csv file



simulation_sigma = np.linspace(0, 0.22, 16)

# Step: Call data from final color.csv file

wmean = 1

# Step3: Use while loop to evolve

while len(final_sense) < 100:

    """
    Description:
        record sigma show which coefficient we have done. 
        Initial set zero since we font' want to simulate at phase boundary.
    Step: 
      (1) setup the coefficient we use in this simulation: simulation_coefficient
      (2) setup the coefficient we use in differential technique: difftech_coefficient
      (3) creat w
      (4) prepare the final list for denote result: denoteslope, denotesense 

    """
    record_coefficient = [0]
    is_near_fixed_point = False

    simulation_coefficient = 0
    w = myf.creatw(totalcarn , 1, simulation_sigma[simulation_coefficient])
    fftw = myf.fft_function(split, w)/len(w)
    initvalue = myf.xinit(amplitude, alpha, length, totalcarn, w)
    initvalue2 = myf.xinit(amplitude, alpha, length, totalcarn, w)
    f = initvalue[2]
    [eigenvalue, eigenfunction] = myf.eigenvalue(w, totalcarn)
    initial_sense = myf.find_boundary(eigenvalue, f)
    np.save('wfile', w)

    
    varw = np.delete(fftw, 0)
    varw = abs(varw)**2
    varw = np.sum(varw)
    denotesense = []
    denoteslope = []
    # Perturbation
    number = np.arange(int(0.5*totalcarn)-1, -int(0.5*totalcarn)-1,-1)
    number = np.delete(number, np.where(number == 1))
    sumfftw = np.delete(fftw, 0)
    summation = (abs(sumfftw)**2)*(np.exp(1j*alpha)-1)*np.divide((np.exp(1j*2*np.pi*number/totalcarn)-1), (np.mean(w)*(np.exp(1j*alpha)-np.exp(1j*2*np.pi*number/totalcarn))))
    calculate_perturbation = np.sum(summation)
    perturbation = calculate_perturbation + np.mean(w)*(np.exp(1j*alpha)-1)
    A = f*((np.imag(perturbation))**2)/(-np.real(perturbation))

    # ===================================================
    # Step:
    #   (1) create notefile to let you know which condition the program simulate
    #   (2) create initial condition to simulate

    coefficient = 0.005
    delta_coefficient = coefficient/2
    how_much_time_reach_another_phase = 0
    previous_type = 'I do nothing'
    doing_time = 0
    testtime = 0
    while True:
        print(coefficient)
        try:
            os.makedirs('./{}'.format(coefficient))
        except:
            shutil.rmtree('./{}'.format(coefficient))
            os.makedirs('./{}'.format(coefficient))
        record_coefficient.append(coefficient)
        savetime = 0
        sense = initial_sense + coefficient
        z = (-sense+np.sqrt(sense**2+4*sense*f*eigenvalue))/2
        print(z[np.where(np.real(z)==max(np.real(z)))])
        notefile = open("nowcoefficient.txt", "w+")
        notefile.write("var ={} \nw2={} \nsense={} \ncheckvar={} \nsigma={}".format(np.var(w), abs(fftw[2]), sense, varw, np.sqrt(varw)))
        notefile.close()
        originx = initvalue2[0]
        originv = initvalue2[1]
        """
        Since the real fourier space should be the linear superpositon of eigen vector.
        We add two eigenvector and shuffle to the correct order(0,1,2,...,-1).
        The initial condition will be set on th eigen vector.
        """
        # Set eigenvector
        eigendirection1 = []
        eigendirection2 = []
        eigneindex1 = np.where(np.real(z)==max(np.real(z)))[0][0]
        finaleigenvalue = (z[np.where(np.real(z)==max(np.real(z)))])[0]
        for index in range(0, len(eigenvalue)):
            if index != eigneindex1:
                if abs(np.real(z[index])-np.real(finaleigenvalue))<10**(-8):
                    eigneindex2 = index
        for index in range(0, len(eigenfunction)):
            eigendirection1.append(eigenfunction[index][eigneindex1])
            eigendirection2.append(eigenfunction[index][eigneindex2])

        eigendirection1 = np.array(eigendirection1)
        eigendirection2 = np.array(eigendirection2)
        eigendirection1 = np.roll(eigendirection1, totalcarn//2-1)
        eigendirection2 = np.roll(eigendirection2, totalcarn//2-1)
        # Set initial condition
        firsteigenvector = amplitude*np.append(np.array([0]), eigendirection1 + eigendirection2)
        initx = myf.fft_function(totalcarn, (firsteigenvector)*totalcarn, 'i')
        initx = np.real((initx))
        vfirsteigenvector = amplitude*np.append(np.array([0]), finaleigenvalue*eigendirection1 + np.conjugate(finaleigenvalue)*eigendirection2)
        initv = myf.fft_function(totalcarn, (vfirsteigenvector)*totalcarn, 'i')
        initv = np.real((initv))
        x = initvalue[0]+initx 
        v = initvalue[1]+initv
        y = initx

        totaltime = 0
        time = []
        denotex = []
        denoteflowtime = []
        yplot = []
        yplotheory = []
        tmax = []
        tmin = []
        meanvalue = []
        meanvaluetime = []
        simulationtime = 0
        firsttime = True
        plottime = 0
        theoryy = myf.fft_function(totalcarn, totalcarn*amplitude*np.append(np.array([0]), eigendirection1*np.exp(finaleigenvalue*totaltime) + eigendirection2*np.exp(np.conjugate(finaleigenvalue)*totaltime)), 'i')
        yplot.append(np.real(myf.fft_function(split, y)[1])/len(myf.fft_function(split, y)))
        yplotheory.append(np.real(myf.fft_function(split, theoryy)[1]/len(myf.fft_function(split, y))))

        time.append(totaltime)
        ydot = initv

    # ===================================================

    # Step:
    #   (1) Run simulation until we contour ten maximun and minimun
    #   (2) Calculate slope according to the data we record
    #   (3) If there is no maxiun, we need to direct denote slope and the typ of stability

        # (1)
        while True:
            run_nonlinear = myf_simulate.animate_real_nonlinear(w, originx, x, v, length, split, sense, simulationtime, RK4=True)
            # run_linear = myf_simulate.animate_real_linear(w, y, ydot, length, split, sense, originv, f)
            [x, v, y, deltax, dt] = run_nonlinear
            # [y, ydot, dt] = run_linear
            totaltime += dt
            theoryy = myf.fft_function(totalcarn, totalcarn*amplitude*np.append(np.array([0]), eigendirection1*np.exp(finaleigenvalue*totaltime) + eigendirection2*np.exp(np.conjugate(finaleigenvalue)*totaltime)), 'i')
            # theoryv = myf.fft_function(totalcarn, totalcarn*amplitude*np.append(np.array([0]), finaleigenvalue*eigendirection1*np.exp(finaleigenvalue*totaltime) + np.conjugate(finaleigenvalue)*eigendirection2*np.exp(np.conjugate(finaleigenvalue)*totaltime)), 'i')
            # exit()
            
            if simulationtime % 100 == 0 and simulationtime >= 0:
                yplot.append(np.real(myf.fft_function(split, y)[1])/len(myf.fft_function(split, y)))
                yplotheory.append(np.real(myf.fft_function(split, theoryy)[1]/len(myf.fft_function(split, y))))
                time.append(totaltime)
            simulationtime += 1
            if totaltime//50000 > savetime:
                np.save('./{}/{}_x'.format(coefficient,totaltime), x)
                np.save('./{}/{}_v'.format(coefficient,totaltime), v)
                savetime += 1

            if len(time) >= deltetime:
                # if firsttime:
                #     record = {
                #                 'time' : time,
                #                 'y' : yplot 
                #             }
                #     data = pd.DataFrame(record)
                #     data.to_csv('./{}/data.csv'.format(coefficient))
                #     firsttime = False
                # else:
                #     record = {
                #                 'time' : time,
                #                 'y' : yplot 
                #             }
                #     data = pd.DataFrame(record)
                #     origindata = pd.DataFrame(pd.read_csv('./{}/data.csv'.format(coefficient)))
                #     finalrecord = pd.concat([origindata, data], axis=0, join='inner')
                #     finalrecord.to_csv('./{}/data.csv'.format(coefficient))

                for i in range(2, len(yplot)-1):
                    if yplot[i] > yplot[i+1] and yplot[i] > yplot[i-1] and yplot[i] > 0:
                        if time[i] not in meanvaluetime:
                            meanvalue.append(np.log(abs(yplot[i])))
                            meanvaluetime.append(time[i])
                # record = {
                    # 'time' : time,
                    # 'y' : yplot 
                # }
                # data = pd.DataFrame(record)
                # data.to_csv('./data.csv')
                # exit()
                # meanvaluetime.append(totaltime)
                # meanvalue.append(np.mean(yplot[-int(0.4*len(yplot)):]))
                # if len(meanvalue) != 0 and plottime < len(meanvalue):
                if True:
                    ax.cla()
                    ax.plot(time, yplot, 'r.')
                    ax.plot(time, yplotheory, 'b.')
                    fig.savefig('./{}/test{}.png'.format(coefficient, simulationtime))
                    ax.cla()
                    ax.plot(meanvaluetime, meanvalue, 'r.')
                    fig.savefig('./{}/mean{}.png'.format(coefficient, simulationtime))
                    plottime += 1
                    # exit()
                del time[0:deltetime//2]
                del yplot[0:deltetime//2]
                del yplotheory[0:deltetime//2]


                if np.log10(abs(myf.fft_function(split, y))[1]/len(myf.fft_function(split, y))) < -6:
                    if totaltime < uptime:
                        style = False
                        current_type = 'stable'
                        break
                    else:
                        style = True
                        slope = 0
                        for slopeindex in range(len(meanvalue)-10, len(meanvalue)-1):
                            slope += (meanvalue[slopeindex+1]-meanvalue[slopeindex])/(meanvaluetime[slopeindex+1]-meanvaluetime[slopeindex])
                            slope = slope/(len(meanvalue)-1)
                        break
                # elif np.log10(abs(myf.fft_function(split, y))[1]/len(myf.fft_function(split, y))) > originamplitude + 5:
                elif np.log10(abs(myf.fft_function(split, y))[1]/len(myf.fft_function(split, y))) > -1.2:
                    if totaltime < uptime:
                        style = False
                        current_type = 'unstable'
                        break
                    else:
                        style = True
                        slope = 0
                        for slopeindex in range(len(meanvalue)-10, len(meanvalue)-1):
                            slope += (meanvalue[slopeindex+1]-meanvalue[slopeindex])/(meanvaluetime[slopeindex+1]-meanvaluetime[slopeindex])
                            slope = slope/(len(meanvalue)-1)
                        break
            if len(meanvalue) > 10:
                style = True
                slope = np.diff(np.array(meanvalue)[-len(meanvalue)//2-1:])/np.diff(np.array(meanvaluetime)[-len(meanvaluetime)//2-1:])
                slope = np.mean(slope)
                break

        ax.cla()
        ax.plot(meanvaluetime, meanvalue, 'r.')

        print(slope)
    # ===================================================
    #  Step: Evaluate the stability of the system
    #      (1) If slope > 0 denote it as instable by 'rX'
    #      (2) If slope < 0 denote it as stable by 'bo'
    #
        # if len(yplotmax) or len(meanvalue) != 0:
        if style:
            if len(meanvalue) != 0:
                if slope < 0:
                    current_type = 'stable'

                elif slope > 0:
                    current_type = 'unstable'

            denoteslope.append(slope)
            denotesense.append(sense)
    # ===================================================
    #
    # Step:
    #   (1) Check whether we get the boundary
    #   (2) If not, change the coefficient

        if previous_type == 'stable' and current_type == 'stable':
            previous_type = current_type
            if np.all(abs(np.array(record_coefficient)-(coefficient-delta_coefficient)) > 0.9*delta_coefficient ):
                coefficient -= delta_coefficient
            else:
                if how_much_time_reach_another_phase < 4:
                    delta_coefficient = delta_coefficient/2
                    coefficient -= delta_coefficient
                else:
                    break
        elif previous_type == 'unstable' and current_type == 'unstable':
            previous_type = current_type
            if np.all(abs(np.array(record_coefficient)-(coefficient+delta_coefficient)) > 0.9*delta_coefficient ):
                coefficient += delta_coefficient
            else:
                if how_much_time_reach_another_phase < 4:
                    delta_coefficient = delta_coefficient/2
                    coefficient += delta_coefficient
                else:
                    break

        elif previous_type == 'stable' and current_type == 'unstable':
            is_near_fixed_point = True
            if how_much_time_reach_another_phase < 4:
                delta_coefficient = delta_coefficient/2
                previous_type = current_type
                coefficient += delta_coefficient
                how_much_time_reach_another_phase += 1
            else:
                break

        elif previous_type == 'unstable' and current_type == 'stable':
            if how_much_time_reach_another_phase < 4:
                delta_coefficient = delta_coefficient/2
                previous_type = current_type
                coefficient -= delta_coefficient
                how_much_time_reach_another_phase += 1
            else:
                break

        if previous_type == 'I do nothing':
            previous_type = current_type
            if current_type == 'stable':
                coefficient -= delta_coefficient
            elif current_type == 'unstable':
                coefficient += unstable_delta_coefficient

        testtime += 1
        if testtime >= 4:
            break

    # ===================================================
    #   Step: Calculate the slope and calculate the zero point
    #       (1) Calculate slope
    #       (2) Calaulate the zero point
    #       (3) Renew the final data
    senseslope = 0

    #   (1)

    for datasense in range(0, len(denotesense)-1):
        senseslope += (denoteslope[datasense+1]-denoteslope[datasense])/(denotesense[datasense+1]-denotesense[datasense])
    senseslope = senseslope/(len(denoteslope)-1)
    #   (2)

    b0 = 0
    for datasense in range(0, len(denotesense)):
        b0 += denoteslope[datasense]-senseslope * denotesense[datasense]
    b0 = b0 / len(denotesense)
    final_sense_cal = -b0/senseslope
    #   (3)

    add_finalsense_library = {
                    "sigma": [np.sqrt(np.var(w))],
                    "w2": [fftw[2]],
                    "a": [final_sense_cal],
                    "initialsense": [initial_sense],
                    "perturbation": [A]
                        }
    add_finalsense = pd.DataFrame(add_finalsense_library)
    final_sense_data = pd.concat([read_final, add_finalsense], axis=0, join='inner')
    final_sense_data.to_csv('./pd_data_final.csv')
    read_final_data = pd.read_csv('./pd_data_final.csv')
    read_final = pd.DataFrame(read_final_data)
    read_final.columns = ['0', 'sigma', 'w2', 'a','initialsense', 'perturbation']
    final_sense = read_final_data['a']


