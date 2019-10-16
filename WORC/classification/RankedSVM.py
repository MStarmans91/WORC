#!/usr/bin/env python

# Copyright 2016-2019 Biomedical Imaging Group Rotterdam, Departments of
# Medical Informatics and Radiology, Erasmus MC, Rotterdam, The Netherlands
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import division
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import fminbound
from scipy import linalg
import operator
import WORC.addexceptions as WORCexceptions


'''
This code is based on the original RankSVM Matlab Code from [1] and [2].
Only the multi-classification variant has been ported.


RanKSVM_train trains a multi-label ranking svm using the method described in
[1] and [2] and originally implemented in MATLAB.

[1] Elisseeff A, Weston J. Kernel methods for multi-labelled classfication
      and categorical regression problems. Technical Report,
      BIOwulf Technologies, 2001.
[2] Elisseeff A,Weston J. A kernel method for multi-labelled classification.
      In: Dietterich T G, Becker S, Ghahramani Z, eds. Advances in
      Neural Information Processing Systems 14, Cambridge,
      MA: MIT Press, 2002, 681-687.

Translated by Mumtaz Hussain Soomro (mumtazhussain.soomro@uniroma3.tk) in August 2018
'''


def neg_dual_func(Lambda, Alpha_old, Alpha_new, c_value, kernel,
                  num_training, num_class, Label, not_Label,
                  Label_size, size_alpha):

    # Local Variables: kernel, num_class, Alpha_new, index, i, num_training, k, size_alpha, m, c_value, Label, Alpha, Beta, n, not_Label, output, Lambda, Label_size, Alpha_old
    # Function calls: sum, zeros, neg_dual_func

    Alpha = Alpha_old+np.dot(Lambda, Alpha_new-Alpha_old)
    Beta = np.zeros(shape =(num_class, num_training))
    for k in range(num_class):
          for i in range(num_training):
              for m in range(Label_size[:,int(i)]):
                  for n in range(num_class-Label_size[:,int(i)]):
                      index = np.sum(size_alpha[:,0:i])+n

                      ak = np.array(c_value[k], dtype=int)
                      r1 = Label[int(i)]    ####this supports for only for multiclass
                                     ### if you want to work on multilabel then try this: r1 = Label[i]
                                     #################################################### r1 = r1[m]
                      c1 = not_Label[int(i)]
                      c1 = c1[n]
                      Beta[k,i] = Beta[k,i]+ak[int(r1),int(c1)]*Alpha[:,int(index)]


    output = 0
    for k in range(num_class):
        fg = np.dot(Beta[int(k),:], kernel.conj().T)
        fg = fg.conj().T
        gf = np.dot(Beta[int(k),:], fg)
        output = output + gf

    output = np.dot(0.5, output)
    output = output-np.sum(Alpha)
    return np.array([output])


def is_empty(any_structure):
    if any_structure:
        return False

    else:
        return True


def RankSVM_train_old(train_data, train_target, cost=1, lambda_tol=1e-6,
                  norm_tol=1e-4, max_iter=500, svm='Poly', gamma=0.05,
                  coefficient=0.05, degree=3):
    # NOTE: Only multilabel classification, not multiclass! Make a check.
    '''
         Weights,Bias,SVs = RankSVM_train(train_data,train_target,cost,lambda_tol,norm_tol,max_iter,svm,gamma,coefficient,degree)

      Description

         RankSVM_train takes,
             train_data   - An MxN array, the ith instance of training instance is stored in train_data[i,:]
             train_target - A QxM array, if the ith training instance belongs to the jth class, then train_target[j,i] equals +1, otherwise train_target(j,i) equals -1
              svm          - svm gives the type of svm used in training, which can take the value of 'RBF', 'Poly' or 'Linear'; svm.para gives the corresponding parameters used for the svm:
                             1) if svm is 'RBF', then gamma gives the value of gamma, where the kernel is exp(-Gamma*|x[i]-x[j]|^2)
                            2) if svm is 'Poly', then three values are used gamma, coefficient, and degree respectively, where the kernel is (gamma*<x[i],x[j]>+coefficient)^degree.
                            3) if svm is 'Linear', then svm is [].
             cost         - The value of 'C' used in the SVM, default=1
             lambda_tol   - The tolerance value for lambda described in the appendix of [1]; default value is 1e-6
             norm_tol     - The tolerance value for difference between alpha(p+1) and alpha(p) described in the appendix of [1]; default value is 1e-4
             max_iter     - The maximum number of iterations for RankSVM, default=500

         and returns,
              Weights          - The value for beta[ki] as described in the appendix of [1] is stored in Weights[k,i]
              Bias             - The value for b[i] as described in the appendix of [1] is stored in Bias[1,i]
              SVs              - The ith support vector is stored in SVs[:,i]


       For more details,please refer to [1] and [2].
    '''

    # RankedSVM only works for multilabel problems, not multiclass, so check
    # Whether patients have no class or multiple classes
    n_class = train_target.shape[0]
    n_object = train_target.shape[1]
    for i in range(0, n_object):
        if np.sum(train_target[:, i]) != -n_class + 2:
            raise WORCexceptions.WORCIOError('RankedSVM only works ' +
                                                   'for multilabel problems,' +
                                                   ' not multiclass. One or ' +
                                                   'more objects belong ' +
                                                   'either to no class or' +
                                                   ' multiple classes. ' +
                                                   'Please check your data' +
                                                   ' again.')

    num_training, tempvalue = np.shape(train_data)

    SVs = np.zeros(shape=(tempvalue,num_training))

    num_class, tempvalue = np.shape(train_target)
    lc = np.ones(shape=(1,num_class))

    target = np.zeros(shape=(num_class, tempvalue))
    for i in range(num_training):
        temp = train_target[:,int(i)]
        if np.logical_and(np.sum(temp) != num_class, np.sum(temp) != -num_class):
              #SVs =  (SVs, train_data[int(i),:].conj().T)
            SVs [:,i] =  train_data[int(i),:].conj().T
            target[:,i] = temp


    Dim, num_training = np.shape(SVs)
    Label = np.array(np.zeros(shape=(num_training,1)), dtype=float)
    not_Label = []
    Label_size = np.zeros(shape=(1,num_training))
    size_alpha = np.zeros(shape=(1,num_training), dtype=float)

    for i in range(num_training):
        temp1 = train_target[:,int(i)]
        Label_size[0,int(i)] = np.sum(temp1 == lc)
        lds = num_class-Label_size[0,int(i)]
        size_alpha[0,int(i)] = np.dot(lds, Label_size[0,int(i)])
        for j in range(num_class):
            if temp1[int(j)] == 1:
                 Label[int(i),0] = np.array([j])
            else:
                 not_Label.append((j))

    not_Label = np.reshape(not_Label, (num_training,num_class-1))

    kernel = np.zeros(shape =(num_training, num_training), dtype=float)

    if svm == 'RBF':
        for i in range(num_training):
            for j in range(num_training):
                kernel[int(i),int(j)] = np.exp(-gamma*(np.sum((SVs[:,i]-SVs[:,j])**2)))


    else:
        if svm == 'Poly':
            for i in range(num_training):
                for j in range(num_training):
                    ab= np.dot((np.array([SVs[:,int(j)]])),((np.array([SVs[:,int(i)]])).conj().T))
                    ab=gamma*ab
                    ab=ab+coefficient
                    ab=ab**degree
            #kernel[int(i),int(j)] = (gamma*(SVs[:,int(i)].conj().T)*SVs[:,int(j)]+coefficient)**degree
                    kernel[int(i),int(j)] = np.array([ab])
        else:
                for i in range(num_training):
                    for j in range(num_training):
                        kernel[int(i),int(j)] = np.dot((np.array([SVs[:,int(j)]])),((np.array([SVs[:,int(i)]])).conj().T))

    svm_used=svm;

    #Begin training phase

    #data initializing

    ak = np.sum(size_alpha, dtype=int)
    Alpha = np.zeros(shape=(1, ak))

    ####creating a cell c_value

    c_value = np.zeros((num_class,), dtype=np.object)

    for i in range(num_class):
        c_value[i] = np.zeros(shape=(num_class,num_class))

    for i in range(num_class):
        ak = c_value[i]
        ak[i,:]= np.ones(shape=(1,num_class))
        ak[:,i]= -np.ones(shape=(num_class,))
        c_value[i] = ak

    #print Label_size
     ### Find the Alpha value using Franke and Wolfe method [1]

    continuing = True
    iteration = 0

    while(continuing):

    #computing Beta
    #iteration=iteration+1;

    #disp(strcat('current iteration: ',num2str(iteration)))
        Beta = np.zeros(shape=(num_class,num_training))
        for k in range(num_class):
            for i in range(num_training):
                for m in range(Label_size[:,int(i)]):
                    for n in range(num_class-Label_size[:,int(i)]):
                    #index = np.sum(size_alpha[:,0:i])+(m-1)*(num_class-Label_size[i])+n
                        index = np.sum(size_alpha[:,0:i])+n

                        ak = np.array(c_value[k], dtype=int)
                        r1 = Label[int(i)]    ####this supports for only for multiclass
                                     ### if you want to work on multilabel then try this: r1 = Label[i]
                                     #################################################### r1 = r1[m]
                        c1 = not_Label[int(i)]
                        c1 = c1[n]
                        Beta[k,i] = Beta[k,i]+ak[int(r1),int(c1)]*Alpha[:,int(index)]

    ####computing gradient(ikl)

        inner = np.zeros(shape=(num_class,num_training))
        for k in range(num_class):
            for j in range(num_training):
                inner[k,j] = np.dot(Beta[k,:], kernel[:,j])

        gradient=[]

        for i in range(num_training):
            for m in range(Label_size[:,int(i)]):
                for n in range(num_class-Label_size[:,int(i)]):
                    r1 = Label[int(i)]    ####this supports only for multiclass
                                     ### if you want to work on multilabel then try this: r1 = Label[i]
                                     #################################################### r1 = r1[m]
                    c1 = not_Label[int(i)]
                    c1 = c1[n]
                    temp = inner[int(r1), int(i)]-inner[int(c1),int(i)]-1
            #gradient=np.array([gradient,temp])
                    gradient.append(float(temp))

        gradient = np.array(gradient, dtype=float)
        gradient = gradient.conj().T


    ###Find Alpha_new
        Aeq = np.zeros(shape=(num_class,np.sum(size_alpha, dtype=int)))
        for k in range(num_class):
            counter=0
            for i in range(num_training):
                for m in range (Label_size[:,int(i)]):
                    for n in range(num_class-Label_size[:,int(i)]):
                #counter+=1
                        r1 = Label[i]    ####this supports only for multiclass
                                     ### if you want to work on multilabel then try this: r1 = Label[i]
                                     #################################################### r1 = r1[m]
                        c1 = not_Label[int(i)]
                        c1 = c1[n]
                        ak = c_value[k]
                        Aeq[k,counter] = ak[int(r1),int(c1)]
                        counter+=1
    #print Aeq
        beq=np.zeros(shape=(num_class,))
        LB=np.zeros(shape=(np.sum(size_alpha, dtype=int),1))
        UB=np.zeros(shape=(np.sum(size_alpha, dtype=int),1))
        counter=0
        for i in range(num_training):
            for m in range(Label_size[:,int(i)]):
                for n in range(num_class-Label_size[:,int(i)]):
            #counter+=1
                    UB[counter,:]=cost/(size_alpha[:,i])
                    counter+=1
    #print UB
        cc = [LB.T, UB.T]
        cc =np.ravel(cc)
        bounds = np.reshape(cc, (2,np.sum(size_alpha, dtype=int)))
        bounds = bounds.T
        Alpha_new=linprog(gradient.conj().T,A_ub=None, b_ub=None, A_eq=Aeq, b_eq=beq.T,bounds=bounds)
        Alpha_new = Alpha_new.x
        Alpha_new = (np.array(Alpha_new)).conj().T

        Lambda =fminbound(neg_dual_func, 0.0, 1.0,args=  (Alpha,Alpha_new,c_value,kernel,num_training,num_class,Label,not_Label,Label_size,size_alpha))


    #print Lambda
    #Test convergence

        if np.logical_or(np.abs(Lambda)<=lambda_tol, np.dot(Lambda, np.sqrt(np.sum(((Alpha_new-Alpha)**2.))))<=norm_tol):
            continuing = False
            # np.disp('program terminated normally')
        else:
            if iteration >= max_iter:
                continuing = False

            else:
                Alpha = Alpha+np.dot(Lambda, Alpha_new-Alpha)

                iteration+=1


    Weights = Beta

    #Computing Bias

    Left = []
    Right = []
    for i in  range(num_training):
        for m in range(Label_size[:,int(i)]):
            for n in range(num_class-Label_size[:,int(i)]):
                index = np.sum(size_alpha[:,0:i])+n
                if np.logical_and(np.abs(Alpha[:,int(index)]) >= lambda_tol, np.abs(Alpha[:,int(index)]-cost/(size_alpha[:,i])) >= lambda_tol):
                    vector = np.zeros(shape=(1, num_class))
                    vector[0,int(Label[i])] = 1
                    c1 = not_Label[int(i)]
                    c1 = c1[n]
                    vector[0,int(c1)] = -1.
                    Left.append(vector)
                    Right.append(-gradient[int(index)])


    if is_empty(Left):
        Bias = np.sum(train_target.conj().T)
    else:
        bb = np.array([Right])
        ss1,ss2 = bb.shape
        aa = np.ravel(Left)
        aa = np.reshape(aa,(ss2,num_class))

        ##### Proper way to solve linear equation with non-square matrix
        Bias = np.linalg.lstsq(aa,bb.T,rcond = -1)[0]
        #Bias = Bias.T

    return Weights, Bias, SVs


def RankSVM_train(train_data, train_target, cost=1, lambda_tol=1e-6,
                  norm_tol=1e-4, max_iter=500, svm='Poly', gamma=0.05,
                  coefficient=0.05, degree=3):
    print('Training Ranked SVM ...')
    num_training, tempvalue = np.shape(train_data)

    SVs = np.zeros(shape=(tempvalue,num_training))
    num_class, tempvalue = np.shape(train_target)
    lc = np.ones(shape=(1,num_class))
    #print SVs.shape
    target = np.zeros(shape=(num_class, tempvalue))
    for i in range(num_training):
        temp = train_target[:,int(i)]
        if np.logical_and(np.sum(temp) != num_class, np.sum(temp) != -num_class):
              #SVs =  (SVs, train_data[int(i),:].conj().T)
            #print (train_data[int(i),:]).conj().T
            SVs [:,i] =  train_data[int(i),:].conj().T
            #SVs [i,:] =  train_data[int(i),:].T

            target[:,i] = temp

    #SVs = SVs.T
    Dim, num_training = np.shape(SVs)
    Label = np.array(np.zeros(shape=(num_training,1)), dtype=float)
    not_Label = []
    Label_size = np.zeros(shape=(1,num_training))
    size_alpha = np.zeros(shape=(1,num_training), dtype=float)

    for i in range(num_training):
        temp1 = train_target[:,int(i)]
        Label_size[0,int(i)] = np.sum(temp1 == lc)
        lds = num_class-Label_size[0,int(i)]
        size_alpha[0,int(i)] = np.dot(lds, Label_size[0,int(i)])
        for j in range(num_class):
            if temp1[int(j)] == 1:
                 Label[int(i),0] = np.array([j])
            else:
                 not_Label.append((j))
    not_Label = np.reshape(not_Label, (num_training,num_class-1))

    kernel = np.zeros(shape =(num_training, num_training), dtype=float)

    if svm == 'RBF':
        for i in range(num_training):
            for j in range(num_training):
                kernel[int(i),int(j)] = np.exp(-gamma*(np.sum((SVs[:,i]-SVs[:,j])**2)))


    else:
        if svm == 'Poly':
            for i in range(num_training):
                for j in range(num_training):
                    ab= np.dot((np.array([SVs[:,int(j)]])),((np.array([SVs[:,int(i)]])).conj().T))
                    ab=gamma*ab
                    ab=ab+coefficient
                    ab=ab**degree
            #kernel[int(i),int(j)] = (gamma*(SVs[:,int(i)].conj().T)*SVs[:,int(j)]+coefficient)**degree
                    kernel[int(i),int(j)] = np.array([ab])
        else:
                for i in range(num_training):
                    for j in range(num_training):
                        kernel[int(i),int(j)] = np.dot((np.array([SVs[:,int(j)]])),((np.array([SVs[:,int(i)]])).conj().T))

    svm_used=svm;

    #Begin training phase

    #data initializing

    ak = np.sum(size_alpha, dtype=int)
    Alpha = np.zeros(shape=(1, ak))

    ####creating a cell c_value

    c_value = np.zeros((num_class,), dtype=np.object)

    for i in range(num_class):
        c_value[i] = np.zeros(shape=(num_class,num_class))

    for i in range(num_class):
        ak = c_value[i]
        ak[i,:]= np.ones(shape=(1,num_class))
        ak[:,i]= -np.ones(shape=(num_class,))
        c_value[i] = ak

    #print Label_size
     ### Find the Alpha value using Franke and Wolfe method [1]

    continuing = True
    iteration = 0

    while(continuing):
        print('Iteration {}.').format(iteration)
        #computing Beta
        #iteration=iteration+1;

        #disp(strcat('current iteration: ',num2str(iteration)))
        Beta = np.zeros(shape=(num_class,num_training))
        for k in range(num_class):
            for i in range(num_training):
                for m in range(Label_size[:,int(i)]):
                    for n in range(num_class-Label_size[:,int(i)]):
                    #index = np.sum(size_alpha[:,0:i])+(m-1)*(num_class-Label_size[i])+n
                        index = np.sum(size_alpha[:,0:i])+n

                        ak = np.array(c_value[k], dtype=int)
                        r1 = Label[int(i)]    ####this supports for only for multiclass
                                     ### if you want to work on multilabel then try this: r1 = Label[i]
                                     #################################################### r1 = r1[m]
                        c1 = not_Label[int(i)]
                        c1 = c1[n]
                        Beta[k,i] = Beta[k,i]+ak[int(r1),int(c1)]*Alpha[:,int(index)]

        ####computing gradient(ikl)

        inner = np.zeros(shape=(num_class,num_training))
        for k in range(num_class):
            for j in range(num_training):
                inner[k,j] = np.dot(Beta[k,:], kernel[:,j])

        gradient=[]

        for i in range(num_training):
            for m in range(Label_size[:,int(i)]):
                for n in range(num_class-Label_size[:,int(i)]):
                    r1 = Label[int(i)]    ####this supports only for multiclass
                                     ### if you want to work on multilabel then try this: r1 = Label[i]
                                     #################################################### r1 = r1[m]
                    c1 = not_Label[int(i)]
                    c1 = c1[n]
                    temp = inner[int(r1), int(i)]-inner[int(c1),int(i)]-1
            #gradient=np.array([gradient,temp])
                    gradient.append(float(temp))

        gradient = np.array(gradient, dtype=float)
        gradient = gradient.conj().T


        ###Find Alpha_new

        Aeq = np.zeros(shape=(num_class,np.sum(size_alpha, dtype=int)))
        for k in range(num_class):
            counter=0
            for i in range(num_training):
                for m in range (Label_size[:,int(i)]):
                    for n in range(num_class-Label_size[:,int(i)]):
                #counter+=1
                        r1 = Label[i]    ####this supports only for multiclass
                                     ### if you want to work on multilabel then try this: r1 = Label[i]
                                     #################################################### r1 = r1[m]
                        c1 = not_Label[int(i)]
                        c1 = c1[n]
                        ak = c_value[k]
                        Aeq[k,counter] = ak[int(r1),int(c1)]
                        counter+=1
                        #print Aeq
        beq=np.zeros(shape=(num_class,))
        LB=np.zeros(shape=(np.sum(size_alpha, dtype=int),1))
        UB=np.zeros(shape=(np.sum(size_alpha, dtype=int),1))
        counter=0
        for i in range(num_training):
            for m in range(Label_size[:,int(i)]):
                for n in range(num_class-Label_size[:,int(i)]):
            #counter+=1
                    UB[counter,:]=cost/(size_alpha[:,i])
                    counter+=1
                    #print UB
        cc = [LB.T, UB.T]
        cc =np.ravel(cc)
        bounds = np.reshape(cc, (2,np.sum(size_alpha, dtype=int)))
        bounds = bounds.T
        try:
            Alpha_new = linprog(gradient.conj().T,A_ub=None, b_ub=None, A_eq=Aeq, b_eq=beq.T,bounds=bounds)
            Alpha_new = Alpha_new.x
            Alpha_new = (np.array(Alpha_new)).conj().T
        except IndexError:
            print('[WORC Warning] RankedSVM could not be fit to data. Returning zero classifier.')
            Alpha_new = Alpha

        Lambda =fminbound(neg_dual_func, 0.0, 1.0,args=  (Alpha,Alpha_new,c_value,kernel,num_training,num_class,Label,not_Label,Label_size,size_alpha))


        #print Lambda
        #Test convergence

        if np.logical_or(np.abs(Lambda)<=lambda_tol, np.dot(Lambda, np.sqrt(np.sum(((Alpha_new-Alpha)**2.))))<=norm_tol):
            continuing = False
            # np.disp('program terminated normally')
        else:
            if iteration >= max_iter:
                continuing = False

            else:
                Alpha = Alpha+np.dot(Lambda, Alpha_new-Alpha)

                iteration+=1

    Weights = Beta

    #Computing Bias
    Left = []
    Right = []
    for i in  range(num_training):
        for m in range(Label_size[:,int(i)]):
            for n in range(num_class-Label_size[:,int(i)]):
                index = np.sum(size_alpha[:,0:i])+n
                if np.logical_and(np.abs(Alpha[:,int(index)]) >= lambda_tol, np.abs(Alpha[:,int(index)]-cost/(size_alpha[:,i])) >= lambda_tol):
                    vector = np.zeros(shape=(1, num_class))
                    vector[0,int(Label[i])] = 1
                    c1 = not_Label[int(i)]
                    c1 = c1[n]
                    vector[0,int(c1)] = -1.
                    Left.append(vector)
                    Right.append(-gradient[int(index)])

    if is_empty(Left):
        Bias = np.sum(train_target.conj().T)
    else:
        bb = np.array([Right])
        ss1,ss2 = bb.shape
        aa = np.ravel(Left)
        aa = np.reshape(aa,(ss2,num_class))

        ##### Proper way to solve linear equation with non-square matrix
        Bias = np.linalg.lstsq(aa,bb.T,rcond = -1)[0]
        #Bias = Bias.T

    if Bias.shape == ():
        # Error in Alpha, create empty bias
        print('[WORC Warning] Error in Alpha, create empty bias.')
        Bias = np.zeros(shape=(num_class,1))

    return Weights,Bias,SVs


def RankSVM_test_original(test_data, test_target, Weights, Bias, SVs,
                 svm='Poly', gamma=0.05,
                 coefficient=0.05, degree=3):
    num_testing, tempvalue = np.shape(test_data)
    num_class, tempvalue = np.shape(test_target)
    tempvalue,num_training= np.shape(SVs)
    Label = np.array(np.zeros(shape=(num_testing,1)), dtype=float)
    not_Label = []
    Label_size = np.zeros(shape=(1,num_testing))
    size_alpha = np.zeros(shape=(1,num_training), dtype=float)
    lc = np.ones(shape=(1,num_class))
    for i in range(num_testing):
        temp = test_target[:,int(i)]
        Label_size[0,int(i)] = np.sum(temp == lc)
        lds = num_class-Label_size[0,int(i)]
        size_alpha[0,int(i)] = np.dot(lds, Label_size[0,int(i)])
        for j in range(num_class):
            if temp[int(j)] == 1:
                Label[int(i),0] = np.array([j])
            else:
                not_Label.append((j))

    not_Label = np.reshape(not_Label, (num_testing,num_class-1))

    kernel = np.zeros(shape =(num_testing, num_training), dtype=float)
    if svm == 'RBF':
        for i in range(num_testing):
            for j in range(num_training):
                kernel[int(i),int(j)] = np.exp(-gamma*(np.sum(((test_data[i,:].conj.T)-SVs[:,j])**2)))


    else:
        if svm == 'Poly':
            for i in range(num_testing):
                for j in range(num_training):
                    ab= np.dot((np.array([SVs[:,int(j)]])),(np.array([test_data[int(i),:]]).T))
                    ab=gamma*ab
                    ab=ab+coefficient
                    ab=ab**degree
            #kernel[int(i),int(j)] = (gamma*(SVs[:,int(i)].conj().T)*SVs[:,int(j)]+coefficient)**degree
                    kernel[int(i),int(j)] = np.array([ab])
        else:
             for i in range(num_testing):
                 for j in range(num_training):
                     kernel[int(i),int(j)] = np.dot((np.array([SVs[:,int(j)]])),(np.array([test_data[int(i),:]])))

    Outputs =np.zeros(shape=(num_class,num_testing))

    for i in range(num_testing):
        for k in range(num_class):
            temp = 0
            for j in range(num_training):

                temp=temp + np.dot(Weights[k,j],kernel[i,j])
                #temp = np.array([temp], dtype=int)
                #temp.append(int(temp))
            temp=temp+Bias[k]
            Outputs[k,i]=temp
    #hh = Outputs
    #mm, nn = np.shape(Outputs)

    for i in range(num_testing):       ########### this logic is only for 3 classes and can be modified for further classes

        idx,val = max(enumerate(Outputs[:,i]), key=operator.itemgetter(1))

        if idx == 0:
           Outputs[idx+1,i]=0
           Outputs[idx+2,i]=0
        if idx == 1:
           Outputs[idx+1,i]=0
           Outputs[idx-1,i]=0
        if idx == 2:
           Outputs[idx-1,i]=0
           Outputs[idx-2,i]=0

    Pre_Labels = np.zeros(shape = (num_class,num_testing))
    for i in range(num_testing):
        for k in range(num_class):

            if (abs(Outputs[k,i]) > 0):
                Pre_Labels[k,i]=1
            else:
             Pre_Labels[k,i]=-1

    return Outputs, Pre_Labels


def RankSVM_test(test_data, num_class, Weights, Bias, SVs,
                 svm='Poly', gamma=0.05,
                 coefficient=0.05, degree=3):
    num_testing, tempvalue = np.shape(test_data)
    # num_class, tempvalue = np.shape(test_target)
    tempvalue, num_training = np.shape(SVs)
    # Label = np.array(np.zeros(shape=(num_testing,1)), dtype=float)
    # not_Label = []
    # Label_size = np.zeros(shape=(1,num_testing))
    # size_alpha = np.zeros(shape=(1,num_training), dtype=float)
    # lc = np.ones(shape=(1,num_class))
    # for i in range(num_testing):
    #     temp = test_target[:,int(i)]
    #     Label_size[0,int(i)] = np.sum(temp == lc)
    #     lds = num_class-Label_size[0,int(i)]
    #     size_alpha[0,int(i)] = np.dot(lds, Label_size[0,int(i)])
    #     for j in range(num_class):
    #         if temp[int(j)] == 1:
    #             Label[int(i),0] = np.array([j])
    #         else:
    #             not_Label.append((j))
    #
    # not_Label = np.reshape(not_Label, (num_testing,num_class-1))

    kernel = np.zeros(shape =(num_testing, num_training), dtype=float)
    if svm == 'RBF':
        for i in range(num_testing):
            for j in range(num_training):
                kernel[int(i),int(j)] = np.exp(-gamma*(np.sum(((test_data[i,:].conj.T)-SVs[:,j])**2)))


    else:
        if svm == 'Poly':
            for i in range(num_testing):
                for j in range(num_training):
                    ab= np.dot((np.array([SVs[:,int(j)]])),(np.array([test_data[int(i),:]]).T))
                    ab=gamma*ab
                    ab=ab+coefficient
                    ab=ab**degree
            #kernel[int(i),int(j)] = (gamma*(SVs[:,int(i)].conj().T)*SVs[:,int(j)]+coefficient)**degree
                    kernel[int(i),int(j)] = np.array([ab])
        else:
             for i in range(num_testing):
                 for j in range(num_training):
                     kernel[int(i),int(j)] = np.dot((np.array([SVs[:,int(j)]])),(np.array([test_data[int(i),:]])))

    Probabilities = np.zeros(shape=(num_class,num_testing))

    for i in range(num_testing):
        for k in range(num_class):
            temp = 0
            for j in range(num_training):

                temp=temp + np.dot(Weights[k,j],kernel[i,j])
                #temp = np.array([temp], dtype=int)
                #temp.append(int(temp))
            temp=temp+Bias[k]
            Probabilities[k,i]=temp
    #hh = Outputs
    #mm, nn = np.shape(Outputs)

    # Class with maximum probability is predicted as label
    Predicted_Labels = np.zeros(Probabilities.shape)
    # Probabilities = np.asarray([np.argmax(Probabilities[:, i]) for i in range(num_testing)])
    for i in range(num_testing):
        idx = np.argmax(Probabilities[:, i])
        Predicted_Labels[idx, i] = 1

    # Transpose in order to be compatible with sklearn
    Probabilities = np.transpose(Probabilities)
    Predicted_Labels = np.transpose(Predicted_Labels)

    return Probabilities, Predicted_Labels
