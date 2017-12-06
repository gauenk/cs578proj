#!/usr/bin/env python3
import math
import operator
import os
import re
import string
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

from ClassifierResult import *
from utils import *

from sklearn.metrics import roc_curve,auc

def generate_roc_curves(tr_data,tr_labels,split_number, te_data, te_labels):

    # zero-one losses for naive bayes (nb) and support vector machine (svm)
    params_to_keep = set()
    for j in range(num_features):
        best_feature = 0
        best_loss = 1.0
        for k in range(tr_data.shape[1]):
            if k in params_to_keep: continue
            svm = LinearSVC(penalty = 'l2', C = 0.5, dual=False)
            params_to_keep.add(k)

            lparams = list(params_to_keep)
            svm.fit(tr_data[:, lparams], tr_labels)
            preds = svm.predict(tr_data[:, lparams])
            loss = zero_one_loss(preds, tr_labels)
            params_to_keep.discard(k)
            if (loss <= best_loss):
                best_feature = k
                best_loss = loss
        params_to_keep.add(best_feature)

    # We now have the best features
    lparams = list(params_to_keep)
    nb1 = ClassifierResult('Naive Bayes (L1 features)', [], [])
    svm1 = SVMClassifierResult('svm:_c_=_1.0', [], [], [])
    svm2 = SVMClassifierResult('svm:_c_=_0.75', [], [], [])
    svm3 = SVMClassifierResult('svm:_c_=_0.50', [], [], [])
    svm4 = SVMClassifierResult('svm:_c_=_0.25', [], [], [])
    naive = ClassifierResult('Naive Classifier', [], [])

    ## SVM DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION
    classifiers = {'svm:_c_=_1':svm1, 'svm:_c_=_.75':svm2,
                   'svm:_c_=_.50':svm3, 'svm:_c_=_.25':svm4}
    random_state = np.random.RandomState(0)

    for model_type in classifiers:
        train_data = tr_data
        test_data = te_data
        if model_type == "svm:_c_=_1":
            model = LinearSVC(penalty='l1',C=1,dual=False)
        elif model_type == "svm:_c_=_.75":
            model = LinearSVC(penalty='l1',C=0.75,dual=False)
        elif model_type == "svm:_c_=_.50":
            model = LinearSVC(penalty='l1',C=0.50,dual=False)
        elif model_type == "svm:_c_=_.25":
            model = LinearSVC(penalty='l1',C=0.25,dual=False)
        # elif model_type == "naive bayes":
        #     model = MultinomialNB()
        model.fit(train_data,tr_labels)            
        y_score = model.decision_function(test_data)
        print(y_score)
        print("-=-=-")
        print(te_labels)
        fpr, tpr,_ = roc_curve(te_labels-1,y_score)
        print(fpr,tpr)
        roc_auc = auc(fpr,tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr,tpr,color="darkorange",lw=lw, label="ROC Curve Area = {:.4f}".format(roc_auc))
        plt.plot([0,1],[0,1],color="navy",lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        fn = "figures/ROC_" + model_type + ".png"
        plt.savefig(fn,bbox_inches='tight')
        plt.clf()


def k_fold_cross_validation(tr_data,tr_labels,split_number, te_data, te_labels):

    # zero-one losses for naive bayes (nb) and support vector machine (svm)

    nb1 = ClassifierResult('Naive Bayes (L1 features)', [], [])
    svm1 = SVMClassifierResult('svm: c = 1.0', [], [], [])
    svm2 = SVMClassifierResult('svm: c = 0.75', [], [], [])
    svm3 = SVMClassifierResult('svm: c = 0.50', [], [], [])
    svm4 = SVMClassifierResult('svm: c = 0.25', [], [], [])
    naive = ClassifierResult('Naive Classifier', [], [])

    for i in range(split_number):
        cv_tr_data,cv_tr_labels,cv_te_data,cv_te_labels\
            = split_data(tr_data,tr_labels,split_number,i)
        
        ## SVM DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION
        svm = LinearSVC(penalty='l1',C=1.0,dual=False)
        svm.fit(cv_tr_data,cv_tr_labels)
        preds = svm.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)        
        svm1.zero_one_loss += [loss]
        svm1.params += [svm.coef_.ravel()]

        preds = svm.predict(te_data)
        loss = zero_one_loss(preds,te_labels)        
        svm1.test_loss += [loss]

        ## SVM_C1 DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION
        svm_c1 = LinearSVC(penalty='l1',C=0.75,dual=False)
        svm_c1.fit(cv_tr_data,cv_tr_labels)
        preds = svm_c1.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)        
        svm2.zero_one_loss += [loss]
        svm2.params += [svm.coef_.ravel()]

        preds = svm_c1.predict(te_data)
        loss = zero_one_loss(preds,te_labels)        
        svm2.test_loss += [loss]

        ## SVM_C2 DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION
        svm_c2 = LinearSVC(penalty='l1',C=.50,dual=False)
        svm_c2.fit(cv_tr_data,cv_tr_labels)
        preds = svm_c2.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)        
        svm3.zero_one_loss += [loss]
        svm3.params += [svm.coef_.ravel()]

        preds = svm_c2.predict(te_data)
        loss = zero_one_loss(preds,te_labels)        
        svm3.test_loss += [loss]

        ## SVM_C3 DECLARED HERE TO BE INIT-ed WITH THE DATA FOR CROSS VALIDATION
        svm_c3 = LinearSVC(penalty='l1',C=0.25,dual=False)
        svm_c3.fit(cv_tr_data,cv_tr_labels)
        preds = svm_c3.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)        
        svm4.zero_one_loss += [loss]
        svm4.params += [svm.coef_.ravel()]

        preds = svm_c3.predict(te_data)
        loss = zero_one_loss(preds,te_labels)        
        svm4.test_loss += [loss]

        nb = MultinomialNB()
        params_to_use = [i for i,x in enumerate(svm_c2.coef_.ravel()) if x != 0]
        nb.fit(cv_tr_data[:, list(params_to_use)],cv_tr_labels)
        preds = nb.predict(cv_te_data[:, list(params_to_use)])
        loss = zero_one_loss(preds,cv_te_labels)        
        nb1.zero_one_loss += [loss]

        preds = nb.predict(te_data[:, list(params_to_use)])
        loss = zero_one_loss(preds,te_labels)        
        nb1.test_loss += [loss]

        # Naive
        preds = [2 for x in range(len(cv_te_labels))]
        loss = zero_one_loss(preds, cv_te_labels)
        naive.zero_one_loss += [loss]

        preds = [2 for x in range(len(te_labels))]
        loss = zero_one_loss(preds, te_labels)
        naive.test_loss += [loss]

    return nb1, svm1, svm2, svm3, svm4, naive

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
    #plt.legend(list(model_lines.values()),list(model_lines.keys()))
    plt.legend(list(model_lines.values()),list(model_lines.keys()), bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
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

def write_results(model_losses, output_file):
    with open(output_file, 'w+') as f:
        for model in model_losses.keys():
            f.write('Model: ' + model)
            f.write('\nLoss: ' + str(np.mean(model_losses[model][-1])))
            f.write('\n\n')

def greedy_subset_svm(tr_data, tr_labels, num_features, split_number, te_data, te_labels):
    nb1 = ClassifierResult('GS Naive Bayes', [], [])
    svm1 = GSSVMClassifierResult('GS svm: c = 0.5', [], [], [])
    naive = ClassifierResult('Naive Classifier', [], [])
    greed = GreedyResult('Greedy SVM', num_features)

    for i in range(split_number):
        print('Fold: ' + str(i))
        cv_tr_data,cv_tr_labels,cv_te_data,cv_te_labels\
            = split_data(tr_data,tr_labels,split_number,i)
        
        # We want to use some of the training data as 'validation' data for picking
        # the best subset.  We will use 90% of the data for training, 10% for validation.
        cv_training_data = np.array(cv_tr_data[:int(len(cv_tr_data)*.9)])
        cv_training_labels = cv_tr_labels[:int(len(cv_tr_labels)*.9)]
        cv_validation_data = np.array(cv_tr_data[int(len(cv_tr_data)*.9):])
        cv_validation_labels = cv_tr_labels[int(len(cv_tr_labels)*.9):]
       
        params_to_keep = set()
        for j in range(num_features):
            # print('Feature: ' + str(j))

            best_feature = 0
            best_loss = 1.0
            for k in range(cv_training_data.shape[1]):
                if k in params_to_keep: continue
                svm = LinearSVC(penalty = 'l2', C = 0.5, dual=False)
                params_to_keep.add(k)

                lparams = list(params_to_keep)
                svm.fit(cv_training_data[:, lparams], cv_training_labels)
                preds = svm.predict(cv_validation_data[:, lparams])
                loss = zero_one_loss(preds, cv_validation_labels)
                params_to_keep.discard(k)
                if (loss <= best_loss):
                    best_feature = k
                    best_loss = loss
            params_to_keep.add(best_feature)
            greed.losses[j] = greed.losses[j] + [best_loss]
        
        # We now have the best features
        lparams = list(params_to_keep)
        svm = LinearSVC(penalty = 'l2', C = 0.5, dual=False)
        svm.fit(cv_training_data[:, lparams], cv_training_labels)
        # Use the real cross validation testing data now to get an accurate loss
        preds = svm.predict(cv_te_data[:, lparams])
        loss = zero_one_loss(preds, cv_te_labels)
        svm1.zero_one_loss += [loss]
        params = [0 for x in range(cv_training_data.shape[1])]
        coefs = svm.coef_.ravel()
        for i in range(0, len(lparams)):
            params[lparams[i]] = coefs[i]
        svm1.params += [params]
        svm1.columns += [lparams]

        preds = svm.predict(te_data[:, lparams])
        loss = zero_one_loss(preds,te_labels)        
        svm1.test_loss += [loss]

        nb = MultinomialNB()
        nb.fit(cv_training_data[:, lparams], cv_training_labels)
        preds = nb.predict(cv_te_data[:, lparams])
        loss = zero_one_loss(preds,cv_te_labels)        
        nb1.zero_one_loss += [loss]

        preds = nb.predict(te_data[:, lparams])
        loss = zero_one_loss(preds,te_labels)        
        nb1.test_loss += [loss]

        # Naive
        preds = [2 for x in range(len(cv_te_labels))]
        loss = zero_one_loss(preds, cv_te_labels)
        naive.zero_one_loss += [loss]

        preds = [2 for x in range(len(te_labels))]
        loss = zero_one_loss(preds, te_labels)
        naive.test_loss += [loss]
    
    return nb1, svm1, naive, greed

def experiment_2(data, labels, num_features, cv_split_number, te_data, te_labels):
    tr_data = data
    tr_labels = labels
    tr_data_size = len(tr_data)

    nb_losses = []
    svm_losses = []
    naive_losses = []

    nb_losses_t = []
    svm_losses_t = []
    naive_losses_t = []

    svm_params = []

    svm_columns = []

    #data_balance = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0]
    data_balance = [0.01,0.05,0.10]
    #data_balance = [0.01,0.02]
    for i in data_balance:
        print('Processing: ' + str(i))

        # cross validation training data
        cv_tr_data = tr_data[:int(tr_data_size*i),:]
        # cross validation labels
        cv_tr_labels = tr_labels[:int(tr_data_size*i)]

        nb, svm, naive, greed = greedy_subset_svm(cv_tr_data, cv_tr_labels, num_features, cv_split_number, te_data, te_labels)

        nb_losses += [nb.zero_one_loss] # list of lists where each row is cv_split_number
        svm_losses += [svm.zero_one_loss]
        naive_losses += [naive.zero_one_loss]

        nb_losses_t += [nb.test_loss] # list of lists where each row is cv_split_number
        svm_losses_t += [svm.test_loss]
        naive_losses_t += [naive.test_loss]

        svm_params += [svm.params]
        svm_columns += [svm.columns]
    
    model_losses = {
        nb.name: nb_losses,
        svm.name: svm_losses,
        naive.name: naive_losses,
        nb.test_name: nb_losses_t,
        svm.test_name: svm_losses_t,
        naive.test_name: naive_losses_t
    }
    svm_model_params = {
        svm.name: svm_params
    }
    svm_columns = {
        svm.name: svm_columns
    }
    write_losses(model_losses,data_balance,"exp2")
    write_columns(svm_columns,data_balance,"exp2")

    return model_losses, svm_model_params, np.array(data_balance)*tr_data_size

    
def experiment_1(data,labels,cv_split_number, te_data, te_labels):
    ## NUMBER OF EXAMPLES
    tr_data = data
    tr_labels = labels
    tr_data_size = len(tr_data)
    nb_losses = []

    svm_losses = []
    svm_c1_losses = []
    svm_c2_losses = []
    svm_c3_losses = []

    svm1_losses = []
    svm2_losses = []
    svm3_losses = []
    svm4_losses = []
    naive_losses = []

    nb_losses_t = []
    svm1_losses_t = []
    svm2_losses_t = []
    svm3_losses_t = []
    svm4_losses_t = []
    naive_losses_t = []


    svm1_params = []
    svm2_params = []
    svm3_params = []
    svm4_params = []

    data_balance = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0]
    #data_balance = [0.01,0.05,0.10]
    #data_balance = [0.01,0.02,0.03]
    for i in data_balance:
        print('Processing: ' + str(i))
        # cross validation training data
        cv_tr_data = tr_data[:int(tr_data_size*i),:]
        # cross validation labels
        cv_tr_labels = tr_labels[:int(tr_data_size*i)]

        nb1, svm1, svm2, svm3, svm4, naive = k_fold_cross_validation(cv_tr_data, cv_tr_labels, cv_split_number, te_data, te_labels)
        roc_curve(cv_tr_data, cv_tr_labels, cv_split_number, te_data, te_labels)

        nb_losses += [nb1.zero_one_loss] # list of lists where each row is cv_split_number
        svm1_losses += [svm1.zero_one_loss]
        svm2_losses += [svm2.zero_one_loss]
        svm3_losses += [svm3.zero_one_loss]
        svm4_losses += [svm4.zero_one_loss]
        naive_losses += [naive.zero_one_loss]

        nb_losses_t += [nb1.test_loss] # list of lists where each row is cv_split_number
        svm1_losses_t += [svm1.test_loss]
        svm2_losses_t += [svm2.test_loss]
        svm3_losses_t += [svm3.test_loss]
        svm4_losses_t += [svm4.test_loss]
        naive_losses_t += [naive.test_loss]


        svm1_params += [svm1.params]
        svm2_params += [svm2.params]
        svm3_params += [svm3.params]
        svm4_params += [svm4.params]

    model_losses = {
        nb1.name: nb_losses,
        svm1.name: svm1_losses,
        svm2.name: svm2_losses,
        svm3.name: svm3_losses,
        svm4.name: svm4_losses,
        naive.name: naive_losses,
        nb1.test_name: nb_losses_t,
        svm1.test_name: svm1_losses_t,
        svm2.test_name: svm2_losses_t,
        svm3.test_name: svm3_losses_t,
        svm4.test_name: svm4_losses_t,
        naive.test_name: naive_losses_t
    }
    model_params = {
        svm1.name: svm1_params,
        svm2.name: svm2_params,
        svm3.name: svm3_params,
        svm4.name: svm4_params,
    }
    write_losses(model_losses,data_balance,"exp1")

    return model_losses, model_params, np.array(data_balance)*tr_data_size

def experiment_3(data, labels, num_features, cv_split_number, te_data, te_labels):
    tr_data = data
    tr_labels = labels
    tr_data_size = len(tr_data)

    nb_losses = []
    svm_losses = []
    naive_losses = []

    nb_losses_t = []
    svm_losses_t = []
    naive_losses_t = []

    svm_params = []

    svm_columns = []

    #data_balance = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0]
    data_balance = [0.2]
    #data_balance = [0.01,0.02]
    for i in data_balance:
        print('Processing: ' + str(i))

        # cross validation training data
        cv_tr_data = tr_data[:int(tr_data_size*i),:]
        # cross validation labels
        cv_tr_labels = tr_labels[:int(tr_data_size*i)]

        nb, svm, naive, greed = greedy_subset_svm(cv_tr_data, cv_tr_labels, num_features, cv_split_number, te_data, te_labels)
    return greed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <exp_no>")
    exp_no = sys.argv[1]

    cv_split_number = 10
    num_features = 30
    fast = False
    tr_data,tr_labels = get_data('data/training_clean.csv')
    te_data,te_labels = get_data('data/testing_clean.csv')

    # Randomize data order
    tr_data, tr_labels = shuffle(tr_data, tr_labels)

    no_examples = len(tr_data)
    #no_examples = 100
    if fast:
        no_examples = 50
    
    line_types = ['y--', 'b--', 'r--', 'k--', 'o--', 'g--', 'y-', 'b-', 'r-', 'k-', 'o-', 'g-']

    generate_roc_curves(tr_data, tr_labels, cv_split_number, te_data, te_labels)
    
    if exp_no == "1" or exp_no == "-1":
        model_losses, model_params, cv_training_sizes = experiment_1(tr_data, tr_labels, cv_split_number, te_data, te_labels)
        plot_experiment_losses(model_losses, cv_training_sizes, 'Cross Validation Sample Size', cv_split_number, line_types, './figures/exp_1_losses.png')
        plot_experiment_params(model_params, cv_training_sizes, cv_split_number, './figures/exp_1_params_{}.png')
        write_results(model_losses, './output/exp1.res')

    if exp_no == "2" or exp_no == "-2":
        model_losses, svm_model_params, cv_training_sizes = experiment_2(tr_data, tr_labels, num_features, cv_split_number, te_data, te_labels)
        plot_experiment_losses(model_losses, cv_training_sizes, 'Cross Validation Sample Size', cv_split_number, line_types, './figures/exp_2_losses.png')
        plot_experiment_params(svm_model_params, cv_training_sizes, cv_split_number, './figures/exp_2_params_{}.png')
        write_results(model_losses, './output/exp2.res')

    if exp_no == "3":
        print("hello world")
        greed = experiment_3(tr_data, tr_labels, 90, cv_split_number, te_data, te_labels)
        greed_means = [np.mean(greed.losses[i]) for i in range(90)]
        greed_sds = [np.std(greed.losses[i]) for i in range(90)]
        greed_nums = [i + 1 for i in range(90)]
        plt.errorbar(greed_nums, greed_means, yerr = greed_sds, fmt = '-')
        plt.xlabel('Number of Features')
        plt.ylabel('Zero One Error')
        plt.title('Greedy SVM Error Vs. No. Features')
        plt.savefig('./figures/greedy_by_feature.png', bbox_inches='tight')
        plt.show()



