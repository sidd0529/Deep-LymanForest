"""
This function builds a neural model which uses the generated mocks to build a model
which can robustly predict gravitational lensing potentials.
"""
import os
import pickle
import torch
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb
from typing import List, Union
from scipy import stats
from sklearn.metrics import r2_score
import json
from scipy.stats.stats import pearsonr 


# -------------------------------- Import data -------------------------------------------------------------
"""
Parameters
------------
mock_folder : Path of data files
"""
mock_folder = '/pylon5/as5fphp/sidd529/projects/lya/mock_generate/Aug26_2019/Data_nqso400_npix20_boost10.0_ngrid16/'

""" Store names of all files inside mock_folder in list 'fnames'. """
for root, dirs, fnames in os.walk( mock_folder ):
    print( "Number of mocks: ", len(fnames) )

boost = 10    



""" 
Parameters
------------
nmocks : number of mocks.
ngrid : grid size of image.
nqso : number of quasars.
npix : number of pixels in each sightline.
""" 
with open(mock_folder + fnames[0], 'rb') as file:    
    mock = pickle.load(file)

nmocks = len(fnames)
ngrid = mock[0]
nqso = mock[1]
npix = mock[2]


""" Import mock files. Concatenate the files and create X_mock and y_mock. """
for i in range(0, nmocks):
    with open(mock_folder + fnames[i], 'rb') as file:
        mock = pickle.load(file)
        iseed_potIGM = mock[4]
        phi_field = mock[5]
        spectra = mock[8]

        # scale and shift the inputs to lie between ~0 and ~1.
        spectra = spectra-np.mean(spectra)
        therange = 2.*1.02*np.amax(abs(spectra))
        spectra = 0.5+spectra/therange
        inputs = np.reshape( np.asfarray(spectra) , nqso*npix )
        if(i==0): X_mock = inputs
        else: X_mock = np.vstack( (X_mock , inputs) )

        # set up target output values.
        phi_field = phi_field-np.mean(phi_field)
        therange = 2.*np.amax(abs(phi_field))
        therangeweuse = 4.e-7
        phi_field = 0.5+phi_field/therangeweuse
        targets = np.reshape( np.asfarray(phi_field) , ngrid*ngrid )
        if(i==0): y_mock = targets
        else: y_mock = np.vstack( (y_mock , targets) )


# -------------------------------- Partition data into train and test datasets -----------------------------
def train_test_split(X, y, test_ratio):
    """
    Parameters
    ----------
    X : array, shape = [n_mocks , nqso*npix]
    y : array, shape = [n_mocks , ngrid*ngrid]
    test_ratio : float (less than one.)
        Fraction of mocks which should be used for creation of test data set.
    """
    test_num = int( X.shape[0]*test_ratio )

    test_indices = np.random.choice( X.shape[0], test_num, replace=False )
    train_indices = np.setdiff1d( np.arange(X.shape[0]), test_indices )

    X_train = X[ train_indices , : ]
    y_train = y[ train_indices , : ]
    X_test = X[ test_indices , : ]
    y_test = y[ test_indices , : ]

    return X_train, y_train, X_test, y_test


# -------------------------------- Specify model parameters ------------------------------------------------
input_size = nqso*npix
output_size = ngrid*ngrid
batch_size = 64
learning_rate = 0.0005
num_epoch = 50


# -------------------------------- Obtain partitioned data (train , test) ----------------------------------
"""Obtain train, test partitioned data.
Parameters
----------
trainX : array, shape = [n_mocks_train , nqso*npix]
trainY : array, shape = [n_mocks_train , ngrid*ngrid]
testX : array, shape = [n_mocks_test , nqso*npix]
testY : array, shape = [n_mocks_test , ngrid*ngrid]
"""
trainX, trainY, testX, testY = train_test_split(X_mock, y_mock, 0.2)
print(trainX.shape, trainY.shape, testX.shape, testY.shape, ' Kaya')

train_dataset = data.TensorDataset( torch.from_numpy(trainX).float(), torch.from_numpy(trainY).float() )
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True )

test_dataset = data.TensorDataset(torch.from_numpy(testX).float(), torch.from_numpy(testY).float())
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True )


############################################
###  Neural Net
###########################################
class Net(nn.Module):

    def __init__(self, size_list):
        super(Net, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list)-2):
            layers.append(nn.Linear(size_list[i], size_list[i+1]))
            # layers.append(nn.Dropout(p=0.5))
            layers.append(nn.BatchNorm1d(size_list[i+1]))
            layers.append(nn.Sigmoid())  #layers.append(nn.ReLU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        # layers.append(nn.Softmax(0))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x = x.view(-1, self.size_list[0])
        return self.net(x)

model = Net([input_size, 400, 400, 400, 400, output_size])  #4 hidden layers , 400 hidden units in each layer.
print(model)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam( model.parameters(), lr=learning_rate, weight_decay=0.0005 )



##########################################################
###  Compute train and validation loss at different epochs
##########################################################
total_batches = len(train_loader)

train_loss_lst = []
validation_loss_lst = []

for epoch in range(num_epoch):
    running_loss_train = 0.0
    for i, (xs, ys) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(xs)        
        loss = criterion(outputs, ys)
        running_loss_train += loss.item()

        loss.backward()
        optimizer.step()

    running_loss_validation = 0.0
    for i, (xs, ys) in enumerate(test_loader):
        outputs = model(xs)
        loss = criterion(outputs, ys)
        running_loss_validation += loss.item()


    print('Epoch [{}/{}],  Loss (train): {:.4f},  Loss (test): {:.4f}'
    .format(epoch+1, num_epoch, running_loss_train*(ngrid**2), running_loss_validation*(ngrid**2) ) )

    train_loss_lst += [running_loss_train*(ngrid**2)]
    validation_loss_lst += [running_loss_validation*(ngrid**2)]




##########################################################
###  Plot train and validation loss at different epochs
##########################################################
def plot_error():
    plt.figure(1)
    plt.plot( np.arange(epoch+1), train_loss_lst, marker='o', markersize=5, \
            ls='', color='r', label='training loss' )
    plt.plot( np.arange(epoch+1), validation_loss_lst, marker='o', markersize=5, \
            ls='', color='b', label='validation loss' )
    plt.legend()
    plt.ylabel('Mean squared error', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    plt.tight_layout()
    plt.savefig('Error_boost_'+boost+'.png')
    #plt.show()
    plt.close()



##########################################################
###  Plot selected train and validation potentials
##########################################################
def plot_model(num):
    idx_query = np.random.randint(0, testX.shape[0]-1)

    # pdb.set_trace()
    image_reconstructed_test = ( model( test_dataset[idx_query:idx_query+2][0])[0].detach().numpy() ).reshape( (ngrid, ngrid) )
    image_true_test = np.asfarray( testY[idx_query] ).reshape( (ngrid, ngrid) )

    image_reconstructed_train = ( model( train_dataset[idx_query:idx_query+2][0])[0].detach().numpy() ).reshape( (ngrid, ngrid) )
    image_true_train = np.asfarray( trainY[idx_query] ).reshape( (ngrid, ngrid) )



    f = plt.figure(num+2)

    f.add_subplot(2,2, 1)
    plt.imshow(image_true_train)
    plt.title('True potential (train)')
    f.add_subplot(2,2, 2)
    plt.imshow(image_reconstructed_train)
    plt.title('Reconstructed potential (train)')

    f.add_subplot(2,2, 3)
    plt.imshow(image_true_test)
    plt.title('True potential (validation)')
    f.add_subplot(2,2, 4)
    plt.imshow(image_reconstructed_test)
    plt.title('Reconstructed potential (validation)')

    plt.tight_layout()
    plt.savefig('Test_boost_'+str(boost) + '_num_' + str(i)+'.png')
    #plt.show(block=True)
    plt.close()



###################################################################
###  Pixel by pixel comparison of true and reconstructed potentials
###################################################################
def correlation_stats():
    idx_query = np.random.randint(0, testX.shape[0]-1)

    image_reconstructed_test = ( model( test_dataset[:][0])[0].detach().numpy() )
    image_true_test = np.asfarray( testY[idx_query] )

    image_reconstructed_train = ( model( train_dataset[:][0])[0].detach().numpy() )
    image_true_train = np.asfarray( trainY[idx_query] )

    r2_test = pearsonr( image_true_test, image_reconstructed_test)[0]
    pvalue_test = pearsonr( image_true_test, image_reconstructed_test)[1]

    r2_train = pearsonr( image_true_train, image_reconstructed_train)[0]
    pvalue_train = pearsonr( image_true_train, image_reconstructed_train)[1]   

    return r2_test , r2_train , pvalue_test , pvalue_train     



if(1):
    plot_error()


if(1):
    for i in range(5000):
        plot_model(num=i)


if(1):
    r2_test_list = []
    r2_train_list = []

    pvalue_test_list = []
    pvalue_train_list =[]

    for i in range(5000):
        r2_test , r2_train , pvalue_test , pvalue_train = correlation_stats()
        r2_test_list += [r2_test]
        r2_train_list += [r2_train]
        pvalue_test_list += [pvalue_test]
        pvalue_train_list += [pvalue_train]



    with open('r2_test_list_boost_'+boost+'.txt', 'w') as outfile:
        json.dump(r2_test_list, outfile)


    with open('r2_train_list_boost_'+boost+'.txt', 'w') as outfile:
        json.dump(r2_train_list, outfile)        