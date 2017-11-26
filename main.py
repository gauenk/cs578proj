#!/usr/bin/env python3
import os,sys,re,math,operator,string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from ClassifierResult import *
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import make_classification

def k_fold_cross_validation(tr_data,tr_labels,split_number):

    # zero-one losses for naive bayes (nb) and support vector machine (svm)

    nb1 = ClassifierResult('Naive Bayes', [])
    svm1 = SVMClassifierResult('svm: c = 1.0', [], [])
    svm2 = SVMClassifierResult('svm: c = 0.75', [], [])
    svm3 = SVMClassifierResult('svm: c = 0.50', [], [])
    svm4 = SVMClassifierResult('svm: c = 0.25', [], [])

    # grab what size the chunks should be
    chunk = int(len(tr_data)/split_number)

    # set the start and end index
    te_start_index = 0
    te_end_index = chunk

    # set the start and end index
    tr_start_index = te_end_index
    tr_end_index = len(tr_data)


    for i in range(split_number):

        cv_tr_data,cv_tr_labels,cv_te_data,cv_te_labels\
            = split_data(tr_data,tr_labels,split_number,i)

        ## add bias terms -- do we need it?
        # cv_tr_data = add_bias_term(cv_tr_data)
        # cv_te_data = add_bias_term(cv_te_data)

        ## NAIVE DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION
        nb = MultinomialNB()
        nb.fit(cv_tr_data,cv_tr_labels)
        preds = nb.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)        
        nb1.zero_one_loss += [loss]
        
        ## SVM DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION
        svm = LinearSVC(penalty='l1',C=1.0,dual=False)
        svm.fit(cv_tr_data,cv_tr_labels)
        preds = svm.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)        
        svm1.zero_one_loss += [loss]
        svm1.params += [svm.coef_.ravel()]

        ## SVM_C1 DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION
        svm_c1 = LinearSVC(penalty='l1',C=0.75,dual=False)
        svm_c1.fit(cv_tr_data,cv_tr_labels)
        preds = svm_c1.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)        
        svm2.zero_one_loss += [loss]
        svm2.params += [svm.coef_.ravel()]

        ## SVM_C2 DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION
        svm_c2 = LinearSVC(penalty='l1',C=.50,dual=False)
        svm_c2.fit(cv_tr_data,cv_tr_labels)
        preds = svm_c2.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)        
        svm3.zero_one_loss += [loss]
        svm3.params += [svm.coef_.ravel()]

        ## SVM_C3 DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION
        svm_c3 = LinearSVC(penalty='l1',C=0.25,dual=False)
        svm_c3.fit(cv_tr_data,cv_tr_labels)
        preds = svm_c3.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)        
        svm4.zero_one_loss += [loss]
        svm4.params += [svm.coef_.ravel()]

    return nb1, svm1, svm2, svm3, svm4

def plot_experiment_losses(model_losses, xaxis_var, xlabel, cv_split_number, line_types, fn = None):
    print("-=-=-=-=-=-=-=-=-\nPLOTTING LOSS VALUES\n-=-=-=-=-=-=-=-=-")

    keys = model_losses.keys()
    print(keys)
    sorted(keys)
    print(keys)
    print(line_types)
    model_lines = {}
    i = 0
    for model in keys:
        if model_losses[model] is not None and len(model_losses[model]) != 0:
            mean = np.mean(model_losses[model],1)
            sderr = np.std(model_losses[model],1)/cv_split_number
            model_lines[model] = plt.errorbar(xaxis_var,mean,\
                    sderr,fmt=line_types[i],label=model)
            i += 1
    plt.legend(list(model_lines.values()),list(model_lines.keys()))
    plt.xlabel(xlabel)
    plt.ylabel("Zero-One Loss")
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn,bbox_inches='tight')
    plt.clf()
            
def plot_experiment_params(model_params, xaxis_var, cv_split_number, fn=None):
    print("-=-=-=-=-=-=-=-=-\nPLOTTING PARAMETER VALUES\n-=-=-=-=-=-=-=-=-")
    # Plot the highest training size params
    tr_size_idx = len(xaxis_var)-1
    for model in model_params.keys():
        plt.figure(figsize=(20,6))
        plt.ylabel("SVM Coefficient Value")
        plt.xlabel("Coefficient Index")
        data = np.array(model_params[model][tr_size_idx])
        plt.boxplot(x=data,sym="")
        if fn is None:
            plt.show()
        else:
            filename = fn.format(model.replace(' ', '_').replace('=', '').replace(':', '').replace('.', '_'))
            plt.savefig(filename, bbox_inches='tight')
        plt.clf()
            

    
def test_model():
    if len(sys.argv) != 4:
        print("System Usage: python main.py <trainingDataFilename> <testingDataFilename> <modelIdx>")
        print("modelIdx\n   1 : naive bayes \n   2 : svm\n")
        sys.exit()
    else:
        tr_data,tr_labels,cw = get_data(sys.argv[1])
        te_data,te_labels,_ = get_data(sys.argv[2])

        if sys.argv[3] == "1":
            model = MultinomialNB()
            post_fix = "NB"
        elif sys.argv[3] == "2":
            model = LinearSVC(penalty='l1',C=.1,dual=False)
            post_fix = "SVM"
        else:
            sys.exit("modelIdx must be in {1,2}")

        model.fit(tr_data,tr_labels)
        preds = model.predict(te_data)
        loss = zero_one_loss(preds,te_labels)
        print("ZERO-ONE-LOSS-" + post_fix +" {0:.4f}".format(round(float(loss),4)))

    
def experiment_1(data,labels,cv_split_number):
    ## NUMBER OF EXAMPLES
    tr_data = data
    tr_labels = labels
    tr_data_size = len(tr_data)
    nb_losses = []
    svm1_losses = []
    svm2_losses = []
    svm3_losses = []
    svm4_losses = []

    svm1_params = []
    svm2_params = []
    svm3_params = []
    svm4_params = []

    #data_balance = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0]
    #data_balance = [0.01,0.05,0.10]
    data_balance = [0.01,0.02,0.03]
    for i in data_balance:
        print('Processing: ' + str(i))
        # cross validation training data
        cv_tr_data = tr_data[:int(tr_data_size*i),:]
        # cross validation labels
        cv_tr_labels = tr_labels[:int(tr_data_size*i)]

        nb1, svm1, svm2, svm3, svm4 = k_fold_cross_validation(cv_tr_data, cv_tr_labels, cv_split_number)

        nb_losses += [nb1.zero_one_loss] # list of lists where each row is cv_split_number
        svm1_losses += [svm1.zero_one_loss]
        svm2_losses += [svm2.zero_one_loss]
        svm3_losses += [svm3.zero_one_loss]
        svm4_losses += [svm4.zero_one_loss]

        svm1_params += [svm1.params]
        svm2_params += [svm2.params]
        svm3_params += [svm3.params]
        svm4_params += [svm4.params]

    model_losses = {nb1.name: nb_losses,
                    svm1.name: svm1_losses,
                    svm2.name: svm2_losses,
                    svm3.name: svm3_losses,
                    svm4.name: svm4_losses,
    }
    model_params = {
        svm1.name: svm1_params,
        svm2.name: svm2_params,
        svm3.name: svm3_params,
        svm4.name: svm4_params,
    }
    write_losses(model_losses,data_balance,"exp1")

    return model_losses, model_params, np.array(data_balance)*tr_data_size

if __name__ == "__main__":

    if False:
        test_model()
    else:
        if len(sys.argv) < 3:
            print("Usage: python main.py <data_file> <exp_no>")
        exp_no = sys.argv[2]

        cv_split_number=10
        fast = False
        tr_data,tr_labels = get_data(sys.argv[1])

        no_examples = len(tr_data)
        #no_examples = 100
        if fast:
            no_examples = 50
        
        line_types = ['y--', 'b--', 'r--', 'k--', 'o--']

        if exp_no == "1" or exp_no == "-1":
            model_losses, model_params, cv_training_sizes = experiment_1(tr_data, tr_labels, cv_split_number)
            plot_experiment_losses(model_losses, cv_training_sizes, 'Cross Validation Sample Size', cv_split_number, line_types, './figures/exp_1_losses.png')
            plot_experiment_params(model_params, cv_training_sizes, cv_split_number, './figures/exp_1_params_{}.png')