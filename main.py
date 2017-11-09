#!/usr/bin/env python3
import os,sys,re,math,operator,string
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import make_classification


def k_fold_cross_validation(tr_data,tr_labels,split_number):

    # zero-one losses for naive bayes (nb) and support vector machine (svm)
    nb_zo_loss = []
    svm_zo_loss = []
    svm_c1_zo_loss = []
    svm_c2_zo_loss = []
    svm_c3_zo_loss = []
    svm_c1_params = []
    svm_c2_params = []
    svm_c3_params = []
    

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
        nb_zo_loss += [loss]
        
        ## SVM DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION
        svm = LinearSVC(penalty='l1',C=10,dual=False)
        svm.fit(cv_tr_data,cv_tr_labels)
        preds = svm.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)        
        svm_zo_loss += [loss]

        ## SVM_C1 DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION
        svm_c1 = LinearSVC(penalty='l1',C=1,dual=False)
        svm_c1.fit(cv_tr_data,cv_tr_labels)
        preds = svm_c1.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)        
        svm_c1_zo_loss += [loss]

        ## SVM_C2 DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION
        svm_c2 = LinearSVC(penalty='l1',C=.1,dual=False)
        svm_c2.fit(cv_tr_data,cv_tr_labels)
        preds = svm_c2.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)        
        svm_c2_zo_loss += [loss]

        ## SVM_C3 DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION
        svm_c3 = LinearSVC(penalty='l1',C=0.01,dual=False)
        svm_c3.fit(cv_tr_data,cv_tr_labels)
        preds = svm_c3.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)        
        svm_c3_zo_loss += [loss]

    print(svm_zo_loss,svm_c1_zo_loss,svm_c2_zo_loss,svm_c3_zo_loss)
    return np.array(nb_zo_loss),np.array(svm_zo_loss),np.array(svm_c1_zo_loss),np.array(svm_c2_zo_loss),np.array(svm_c3_zo_loss)

def plot_experiment(nb_losses,svm_losses,svm_c1_losses,svm_c2_losses,svm_c3_losses,xaxis_var,xlabel,cv_split_number,fn=None):
    model_losses = {"nb":[nb_losses,"b--"],"svm: c = 0.25":[svm_c1_losses,"o--"],"svm: c = 0.50":[svm_c2_losses,"r--"],"svm: c = 0.75":[svm_c3_losses,"k--"],"svm: c = 1.0":[svm_losses,"y--"]}
    #model_losses = {"svm: c = 0.25":[svm_c1_losses,"o--"],"svm: c = 0.50":[svm_c2_losses,"r--"],"svm: c = 0.75":[svm_c3_losses,"k--"],"svm: c = 1.0":[svm_losses,"y--"]}

    model_lines = {}
    for model in model_losses.keys():
        if model_losses[model][0] is not None and len(model_losses[model][0]) != 0:
            mean = np.mean(model_losses[model][0],1)
            sderr = np.std(model_losses[model][0],1)/cv_split_number
            model_lines[model] = plt.errorbar(xaxis_var,mean,\
                    sderr,fmt=model_losses[model][1],label=model)
    plt.legend(list(model_lines.values()),list(model_lines.keys()))
    plt.xlabel(xlabel)
    plt.ylabel("Zero-One Loss")
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn,bbox_inches='tight')

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
    svm_losses = []
    svm_c1_losses = []
    svm_c2_losses = []
    svm_c3_losses = []
    data_balance = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0]
    #data_balance = [0.01,0.05,0.10,0.15]
    for i in data_balance:
        # cross validation training data
        cv_tr_data = tr_data[:int(tr_data_size*i),:]
        # cross validation labels
        cv_tr_labels = tr_labels[:int(tr_data_size*i)]
        nb_loss,svm_loss,svm_c1_loss,svm_c2_loss,svm_c3_loss = \
            k_fold_cross_validation(cv_tr_data,cv_tr_labels,cv_split_number)
        nb_losses += [nb_loss] # list of lists where each row is cv_split_number
        svm_losses += [svm_loss]
        svm_c1_losses += [svm_c1_loss]
        svm_c2_losses += [svm_c2_loss]
        svm_c3_losses += [svm_c3_loss]

    model_losses = {"nb": nb_losses,
                    "svm: c = 1.0": svm_losses,
                    "svm: c = 0.25": svm_c1_losses,
                    "svm: c = 0.50": svm_c2_losses,
                    "svm: c = 0.75": svm_c3_losses}
    write_losses(model_losses,data_balance,"exp1")

    return np.array(nb_losses),np.array(svm_losses),\
        np.array(svm_c1_losses),np.array(svm_c2_losses),np.array(svm_c3_losses),\
        np.array(data_balance)*tr_data_size,\
        "Cross Validation Sample Size"

def experiment_2(tr_data,tr_labels,te_data,te_labels,cv_split_number):
    ## NUMBER OF EXAMPLES
    tr_data = data
    tr_labels = labels
    tr_data_size = len(tr_data)
    nb_losses = []
    svm_losses = []
    svm_c1_losses = []
    svm_c2_losses = []
    svm_c3_losses = []
    #data_balance = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0]
    data_balance = [0.01,0.05,0.10,0.15]
    for i in data_balance:
        # cross validation training data
        cv_tr_data = tr_data[:int(tr_data_size*i),:]
        # cross validation labels
        cv_tr_labels = tr_labels[:int(tr_data_size*i)]
        nb_loss,svm_loss,svm_c1_loss,svm_c2_loss,svm_c3_loss = \
            k_fold_cross_validation(cv_tr_data,cv_tr_labels,cv_split_number)
        nb_losses += [nb_loss] # list of lists where each row is cv_split_number
        svm_losses += [svm_loss]
        svm_c1_losses += [svm_c1_loss]
        svm_c2_losses += [svm_c2_loss]
        svm_c3_losses += [svm_c3_loss]

    model_losses = {"nb": nb_losses,
                    "svm: c = 0.50": svm_losses,
    write_losses(model_losses,data_balance,"exp1")

    return np.array(nb_losses),np.array(svm_losses),\
        np.array(svm_c1_losses),np.array(svm_c2_losses),np.array(svm_c3_losses),\
        np.array(data_balance)*tr_data_size,\
        "Cross Validation Sample Size"

if __name__ == "__main__":

    if False:
        test_model()
    else:
        if len(sys.argv) < 3:
            print("Usaege: python main.py <data_file> <exp_no>")
        exp_no = sys.argv[2]

        cv_split_number=10
        fast = False
        tr_data,tr_labels = get_data(sys.argv[1])

        no_examples = len(tr_data)
        if fast:
            no_examples = 50

        if exp_no == "1" or exp_no == "-1":
            nb_losses,svm_losses,svm_c1_losses,svm_c2_losses,svm_c3_losses,xaxis_var,xlabel = experiment_1(tr_data,tr_labels,cv_split_number)
            plot_experiment(nb_losses,svm_losses,svm_c1_losses,svm_c2_losses,svm_c3_losses,xaxis_var,xlabel,cv_split_number,"./figures/exp_1.pdf")
        if exp_no == "2" or exp_no == "-1":
            nb_losses,svm_losses,svm_c1_losses,svm_c2_losses,svm_c3_losses,xaxis_var,xlabel = experiment_2(tr_data,tr_labels,te_data,te_labels,cv_split_number)
            plot_experiment(nb_losses,svm_losses,svm_c1_losses,svm_c2_losses,svm_c3_losses,xaxis_var,xlabel,cv_split_number,"./figures/exp_1.pdf")
