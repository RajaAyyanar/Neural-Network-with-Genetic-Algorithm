# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:37:01 2018

@author: Raja Ayyanar
"""


import numpy as np

#Creating Datasets
no_of_training_data=1000
x1=np.random.uniform(low=0, high=1,size=(1,no_of_training_data))
x2=np.random.uniform(low=0, high=1, size=(1,no_of_training_data))
y=x1**2 - x2**2
train_datasets=np.zeros((3,no_of_training_data))
train_datasets[0,:]=x1
train_datasets[1,:]=x2
train_datasets[2,:]=y
train_datasets=np.transpose(train_datasets)
#del x1,x2,y


#Activation Function
def reluFunction(x):
    y=np.maximum(0,x)
    return y

def derivative_relu(x):
    y = (x >= 0).astype(int)
    return y
    

#layer -H1
no_of_inputs_h1=3
no_of_neurons_h1=4
w1=np.random.uniform(0,1,(no_of_inputs_h1,no_of_neurons_h1))
del_w1=np.zeros((no_of_inputs_h1,no_of_neurons_h1))
gamma_g_h1=0.8
gamma_m_h1=0.2

#layer -H2
no_of_inputs_h2=no_of_neurons_h1 + 1
no_of_neurons_h2=4
w2=np.random.uniform(0,1,(no_of_inputs_h2,no_of_neurons_h2))
del_w2=np.zeros((no_of_inputs_h2,no_of_neurons_h2))
gamma_g_h2=0.8
gamma_m_h2=0.2

#Layer - Output 
no_of_inputs_out=no_of_neurons_h2 + 1
no_of_neurons_out=1
no_of_outputs_out=no_of_neurons_out
w_out=np.random.uniform(0,1,(no_of_inputs_out,no_of_neurons_out))
del_w_out=np.zeros((no_of_inputs_out,no_of_neurons_out))
gamma_g_out=0.8
gamma_m_out=0.2

#inputs for H1
all_x= np.zeros((np.shape(train_datasets)))
all_x[:,0]=all_x[:,0]**0
all_x[:,1:3] = train_datasets[:,0:2]
target_y=train_datasets[:,2]


#Epocs for fitness
epocs=round(no_of_training_data / 2)

#Fitness Evaluation
def calculate_fitness(weight_elements):
    
    #weight_elements=weight_elements[0,:]
    w1=np.zeros((no_of_inputs_h1,no_of_neurons_h1))
    w2=np.zeros((no_of_inputs_h2,no_of_neurons_h2))
    w_out=np.zeros((no_of_inputs_out,no_of_neurons_out))
    
    n=np.size(weight_elements)
    n1,m1=np.shape(w1)
    n2,m2=np.shape(w2)
    n_out,m_out=np.shape(w_out)
    if n != n1*m1 +n2*m2 +n_out*m_out:
        print("Check your Matrix dimensions")
    
    #Converting Vector from DE to Matrix for Neural Network
    b=0
    for i in range(0,n1):
         for j in range(0,m1):
             w1[i,j]=weight_elements[b]
             b=b+1
             
    for i in range(0,n2):
         for j in range(0,m2):
             w2[i,j]=weight_elements[b]
             b=b+1
             
    for i in range(0,n_out):
         for j in range(0,m_out):
             w_out[i,j]=weight_elements[b]
             b=b+1
    
    
    ##Neural Network Training Part
    error_y=np.zeros(epocs);
    for ep in range(0,epocs):
        j=np.random.randint(0,np.size(target_y))
        one_x=np.array([all_x[j,:]]);
        a1=one_x @ w1
        d1=reluFunction(a1); #H1 layer Relu activation function
        
        bias1=np.ones((1,1))
        d1_w=np.concatenate((d1,bias1), axis=1)
        a2=d1_w @ w2
        d2=reluFunction(a2) #H2 layer activation
        
        bias2=np.ones((1,1))
        d2_w=np.concatenate((d2,bias2), axis=1)
        a_out=d2_w @ w_out
        d_out=a_out     #linear activation for output layer
        y_out=d_out
        
        error_y[ep]=y_out-target_y[j]

    fitness=np.mean(np.square(error_y))
    return fitness
    
        

    

########GENETIC ALGORITHM - PSO #########
import numpy as np
from random import randint
from random import random

PopulationSize=200;
Dimensions = np.size(w1)+ np.size(w2)+ np.size(w_out);
Xmin = -5.2;
Xmax = 5.2;
Vmin = -5.1;
Vmax = 5.1;
c1= 2;
c2= 2;
w = 0.5;
MaxIterations = 1000;

Positions_X= Xmin + (Xmax-Xmin)*np.random.uniform(0,1,(PopulationSize,Dimensions));
Velocities_V= Vmin + (Vmax-Vmin)*np.random.uniform(0,1,(PopulationSize,Dimensions));

PBestFitnesses=np.zeros(PopulationSize);
PBestPositions = Positions_X;
for Particle in range(0,PopulationSize):
    #PBestFitnesses[Particle]= sum(Positions_X[Particle,:]**2);
    PBestFitnesses[Particle]= calculate_fitness(Positions_X[Particle,:])
GBestFitness=min(PBestFitnesses)
GBestIndex=np.argmin(PBestFitnesses)
GBestPosition= PBestPositions[GBestIndex,:]
BestFitness=np.zeros(MaxIterations)
for Iteration in range(0,MaxIterations):
    w=0.9-0.8* Iteration/MaxIterations;
    for Particle in range(0,PopulationSize):
        Inertia = w*Velocities_V[Particle,:];
        CogAcc= -c1* random()*(Positions_X[Particle,:]+ PBestPositions[Particle,:])
        SoAcc= -c2* random()* (Positions_X[Particle,:]+ GBestPosition)
        Velocities_V[Particle,:]= Inertia + CogAcc + SoAcc
        CurrentParticleVelocity= Velocities_V[Particle,:]
    
        CurrentParticleVelocity[CurrentParticleVelocity > Vmax]= Vmax
        CurrentParticleVelocity[CurrentParticleVelocity < Vmin]= Vmin
        Velocities_V[Particle,:]= CurrentParticleVelocity

        CurrentPosition= Positions_X[Particle,:]
        NewPosition= CurrentPosition + Velocities_V[Particle,:]
        NewPosition[NewPosition>Xmax] = Xmax
        NewPosition[NewPosition<Xmin] =Xmin
        Positions_X[Particle,:]=NewPosition;

        #Newfitness = sum(NewPosition**2)
        Newfitness = calculate_fitness(NewPosition)
        if Newfitness < PBestFitnesses[Particle]:
            PBestFitnesses[Particle]=Newfitness;
            PBestPositions[Particle,:]=NewPosition;
        if Newfitness < GBestFitness:
            GBestFitness = Newfitness;
            GBestPosition = NewPosition;
            
    print('\n Iteration: ',Iteration, 'BestFitness: ',GBestFitness)
    BestFitness[Iteration]=GBestFitness;

print(GBestPosition)

        
#Tesing Part
x_test=np.random.uniform(-1, 1,1000)
x2_test=np.random.uniform(-1,1,1000)
y_test=x_test**2 - x2_test**2
x0_test=x_test**0
testdata=np.array([x0_test, x_test,x2_test, y_test])
testdata=testdata.transpose()

test_in=testdata[:,0:3]
e_test=np.zeros(max(np.shape(test_in)))
for test in range(0,max(np.shape(test_in))):
    one_x=np.array([test_in[test,:]])
    a1=one_x @ w1
    d1=reluFunction(a1); #H1 layer Relu activation function
    
    bias1=np.ones((1,1))
    d1_w=np.concatenate((d1,bias1), axis=1)
    a2=d1_w @ w2
    d2=reluFunction(a2) #H2 layer activation
    
    bias2=np.ones((1,1))
    d2_w=np.concatenate((d2,bias2), axis=1)
    a_out=d2_w @ w_out
    d_out=a_out     #linear activation for output layer
    y_out=d_out
    e_test[test]=y_out-y_test[test]










