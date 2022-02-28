from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier


import os 
import torch
from torch import nn
from sklearn.metrics import accuracy_score
import numpy as np

from dmvaes.models.trento_encoders import EncoderB2
from dmvaes.dataset import TrentoDataset

PATH_E =  "/home/pigi/Downloads/models(1)/models/trento-relaxed_nparticules_30/encoder_z1_-7913891864283884002.pt"
from trento_utils import (
    DATASET,
    N_INPUT,


)
N_LATENT=10
N_SAMPLES=50

encoder =  nn.ModuleDict(
            {"default": EncoderB2( 
                n_input=N_INPUT,
                n_output=N_LATENT,
                n_hidden=512,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        )

if os.path.exists(PATH_E):
    print("model exists; loadizng from .pt")
    encoder.load_state_dict(torch.load(PATH_E))
encoder.eval()

with torch.no_grad():
    x_train,y_train = DATASET.train_dataset_labelled.tensors
    z1_train = encoder['default'](x_train, n_samples = N_SAMPLES)['latent']
    z1_train = z1_train.reshape(len(y_train),-1)

    x_test,y_test = DATASET.test_dataset.tensors
    z1_test = encoder['default'](x_test, n_samples = N_SAMPLES)['latent']
    z1_test = z1_test.reshape(len(y_test),-1)

    print(x_train.shape, z1_train.shape, y_train.shape)
    print(x_test.shape, z1_test.shape, y_test.shape)

    clf = SVC(C=1.5, kernel="rbf")
    clf.fit(z1_train, y_train)
    print("SVM train score: ", clf.score(z1_train, y_train))
    print("SVM score: ", clf.score(z1_test, y_test))

    clf = RandomForestClassifier(
        n_estimators=50,
        min_samples_leaf=2,
        max_depth=80,
        min_samples_split=5,
        bootstrap=True,
        max_features="sqrt",
        )    
    clf.fit(z1_train, y_train)
    print("RF train score: ", clf.score(z1_train, y_train))
    print("RF score: ", clf.score(z1_test, y_test))

    clf = GaussianProcessClassifier()
    clf.fit(z1_train, y_train)
    print("GP train score: ", clf.score(z1_train, y_train))
    print("GP score: ", clf.score(z1_test, y_test))