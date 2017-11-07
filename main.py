#!/usr/bin/env python3
import os,sys,re,math,operator,string
import numpy as np
import matplotlib.pyplot as plt
from svm import SupportVectorMachine
from data_class import *
from tree_classes import *
from utils import *

def k_fold_cross_validation(tr_data,tr_labels,split_number):

    # zero-one losses for naive bayes (nb) and support vector machine (svm)
    nb_zo_loss = []
    svm_zo_loss = []

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
        cv_tr_data = add_bias_term(cv_tr_data)
        cv_te_data = add_bias_term(cv_te_data)

        ## NAIVE DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION

        ## SVM DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION
        svm = SupportVectorMachine(len(cv_tr_data[0]),0.5,0.01)

        svm.train(cv_tr_data,cv_tr_labels)

        preds = svm.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)        
        svm_zo_loss += [loss]

    return np.array(nb_zo_loss),np.array(svm_zo_loss)

def plot_experiment(nb_losses,svm_losses,xaxis_var,xlabel,cv_split_number,fn=None):

    model_losses = {"nb":[nb_losses,"b--"],"svm":[svm_losses,"g--"]}
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
        print("modelIdx\n   1 : decision tree\n   2 : bagging\n   3 : random forests")
        sys.exit()
    else:
        tr_data,tr_labels,cw = get_data(sys.argv[1],c_words=None,total_f=1000)
        te_data,te_labels,_ = get_data(sys.argv[2],c_words=cw,total_f=1000)

        if sys.argv[3] == "1":
            model = DecisionTree(tr_data,tr_labels)
            post_fix = "DT"
        elif sys.argv[3] == "2":
            model = Bagging(tr_data,tr_labels)
            post_fix = "BT"
        elif sys.argv[3] == "3":
            model = RandomForest(tr_data,tr_labels)
            post_fix = "RF"
        elif sys.argv[3] == "4":
            model = RandomForest(tr_data,tr_labels)
            post_fix = "RF"
        else:
            sys.exit("modelIdx must be in {1,2,3}")

        model.train()
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
    data_balance = [0.025,0.05,0.125,0.25]    
    for i in data_balance:
        # cross validation training data
        cv_tr_data = tr_data[:int(len(tr_data)*i),:1000]
        # cross validation labels
        cv_tr_labels = tr_labels[:int(len(tr_labels)*i)]
        nb_loss,svm_loss = \
            k_fold_cross_validation(cv_tr_data,cv_tr_labels,cv_split_number)
        nb_losses += [nb_loss]
        svm_losses += [svm_loss]

    model_losses = {"nb": nb_losses,
                    "svm": svm_losses}

    write_losses(model_losses,data_balance,"exp1")

    return np.array(nb_losses),np.array(svm_losses),\
        np.array(data_balance)*tr_data_size,\
        "Cross Validation Sample Size"


if __name__ == "__main__":

    if True:
        test_model()
    else:
        if len(sys.argv) < 3:
            print("Usaege: python main.py <data_file> <exp_no>")
        exp_no = sys.argv[2]

        cv_split_number=10
        fast = True
        tr_data,tr_labels,_ = get_data(sys.argv[1],c_words=None)

        no_examples = 500
        if fast:
            no_examples = 50

        if exp_no == "1" or exp_no == "-1":
            nb_losses,svm_losses,xaxis_var,xlabel =\
                   experiment_1(tr_data,tr_labels,cv_split_number)
            plot_experiment(dt_losses,bt_losses,rf_losses,svm_losses,\
                            xaxis_var,xlabel,cv_split_number,"./figures/exp_1.pdf")
