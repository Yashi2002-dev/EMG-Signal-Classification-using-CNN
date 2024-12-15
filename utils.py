import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
import os
import random
from sklearn.preprocessing import MinMaxScaler
from os.path import exists

def divide_Subjects(path_files, nval = 5, ntest = 0):
    """
    Divides randomly the subjects in  path_files into:
    ntest subjects for testing
    nval  subjects for validation
    total of subjects- (ntest + nval) subjects for training
    """
    train_subjects = os.listdir(path_files)
    val_subjects = random.sample(train_subjects, nval)
    test_subjects = []   

    for subject in val_subjects:
        if subject in train_subjects:
            train_subjects.remove(subject)
    
    if ntest>0:
        test_subjects = random.sample(train_subjects, ntest)
        for subject in test_subjects:
            if subject in train_subjects:
                train_subjects.remove(subject)
    
    return(train_subjects, val_subjects, test_subjects)


def WindowClass_df(DataFrame, window_size = 800, wait_size = 250, interleaved = (0,0)):
    """
    Takes the dataframe DataFrame that has the 9 columns: ['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8', 'class'] and
    returns a new dataframe of two columns: ['Window', 'Class']. The first column contains the windows of dimensions (window_size, 8) with window_size 
    consecutive meditions in channels 1-8.
    
    The last window_size-wait_size meditions of every window is overlaped with the first window_size-wait_size meditions of the next window.
    This overlap is present to allow the generation of more windows (therefore, more data).

    Classes 0 and 7 are ignored, the first one because is unlabeled data, and the second one because not every subject performed that gesture.

    interleaved: is used as the first technique of data augmentation: generates the dataframes of the meditions every interleaved[0] until interleaved[1]-1 meditions
    """     
    
    df_filtered = DataFrame[(DataFrame['class'] != 0) & (DataFrame['class'] != 7)]
    winds = []
    clasess = []
    for i in range(1,7):
        df_temp = df_filtered[df_filtered['class'] == i]
        
        list_Lists_Dfs = []

        if (interleaved[0] > 0 and interleaved[1] > 0): #Applying the interleave
            list_Lists_Dfs = [[df_temp.iloc[j::k,:] for j in range(k)] for k in range(interleaved[0], interleaved[1])]
             

        list_Lists_Dfs.append([df_temp])
        
        for listdf in list_Lists_Dfs:
            for df in listdf:        
                window = df.head(window_size)
                
                while window.shape == (window_size, df.shape[1]):
                    df_temp = df_temp.iloc[(wait_size): , :]
                    Wind_Array = window.iloc[:, 1:9].to_numpy().reshape((1,window_size,8)) #This are the windows to be used
                    Wind_Class = i-1 #Note that the classes are mapped: (1,2,...,6) -> (0,1,...5)
                    winds.append(Wind_Array)
                    clasess.append(Wind_Class)
                    window = df_temp.head(window_size)       

    dict_ = {'Window': winds, 'Class': clasess}
    return(pd.DataFrame(dict_))


def Windows_df_from_subjects(subjects, path_files, window_size = 800, wait_size = 250, interleaved = (0,0), scaler = None):
    """
    Creates the dataframe of pairs (Windows,Classes) for all subject in the list "subjects" (subjects for training, validation or testing)
    If a scaler is given (see function get_scaler()), then it is used to normalize the data.
    """
    

    frames = []        

    for sub in subjects:
        path = os.path.join(path_files,sub)
        for file in os.listdir(path):
            df = pd.read_csv(os.path.join(path,file), sep = "\t")
            
            if scaler is not None:
                cols = ['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']
                df[cols] = pd.DataFrame(scaler.transform(df[cols]), columns = cols)
            
            
            frames.append(WindowClass_df(df, window_size, wait_size, interleaved))           

    df = pd.concat(frames)    
        
    return df



def get_scaler(train_subjects, path_files):

    """
    This function gives a scaler from sklearn.preprocessing (already fitted to the training data) that's going to be used to normalize the validation and testing data.
    MinMaxScaler is used, but it can be changed to any other such as StandardScaler, Normalizer, etc. A future benchmark between different scalers could be helpful.

    """

    train_frames = []   
    cols = ['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']    

    for sub in train_subjects:
        path = os.path.join(path_files,sub)
        for file in os.listdir(path):
            df = pd.read_csv(os.path.join(path,file), sep = "\t")            
            train_frames.append(df[(df['class'] != 0) & (df['class'] != 7)])
            
    df_train = pd.concat(train_frames)
    scaler = MinMaxScaler().fit(df_train[cols])
            
    return scaler



# The next three functions are used to do the data augmentation by gaussian noise


def means_PchannelPclass(df_train):

    """
    This function calculates the mean of each channel per class in the train data frame (could be anyone, but should be used only in training).
    Returns a dictionary where each pair (key : value) is ('the class' : and array of 8 elements, each being the mean of the corresponding channel).
    """


    Means_dict = {}
    for c in df_train['Class'].unique():
        total = len(df_train[df_train['Class']==c])
        sum_1_800_8 = df_train[df_train['Class']==c]['Window'].sum()
        sum_1_8 = sum_1_800_8.sum(axis=1)
        sum_8 = sum_1_8.reshape(8)
        sum_8 = sum_8/ (800*total)
        Means_dict[c] = list(sum_8)
        
    return Means_dict

def GN_WindowGenerator(class_df, Means_dict, SNR):

    """
    This function will create a new pair (Window, Class) for data augmentation from a Gaussian distribution.
    Args:
    class_df: the class of the new window.
    Means_dict: a dictionary  where each pair (key : value) is equivalent tho the given created by means_PchannelPclass().
    SNR: signal-to-noise-ratio. An integer hyperparameter. A future benchmark between different SNRs could be helpful.    

    Using the means in Means_dict[class_df] for each channel, this function creates a new pair  (Window, class_df) to be used in training.

    The same value of SNR is used for every channel, in the future could be tried different SNRs values for each channel.
    """

    means_perChannel = Means_dict[class_df]
    noise = []
    for mean in means_perChannel:
            scale = np.sqrt((mean**2)/SNR)
            noise.append(np.random.normal(loc=0.0, scale=scale, size=(1,800,1))) 
    
    GN_W = np.stack(noise, axis = 2).reshape(1,800,8)
    return GN_W

def GN_DataAugmentation(df_train, SNR):
    """
    Uses means_PchannelPclass() and GN_WindowGenerator() for create one new pair (Window, class) for each element in the dataframe for training "df_train".
    """
    Means_dict = means_PchannelPclass(df_train)
    new_Data = []
    new_Classes = []
    for index, row in df_train.iterrows():
        Class = row['Class']
        Window = row['Window']    
        GN_W = GN_WindowGenerator(Class, Means_dict, SNR)
        new_Window = Window + GN_W
        new_Data.append(new_Window)
        new_Classes.append(Class)
    
    new_DF = pd.DataFrame({'Window':new_Data,'Class':new_Classes})
    
    return new_DF





# The pytorch dataset to be used by the models.
# Given that all the hardwork was already done by the previous functions, this class is pretty straightforward.

class Dataset_EMG(torch.utils.data.Dataset):

    def __init__(self, df):

        self.Windows = list(df['Window'])
        self.Classes = list(df['Class'])
        
    def __len__(self):
        return len(self.Windows)

    def __getitem__(self, idx): 
        
        Window_Array = np.array(self.Windows[idx])
        class_idx = np.array(self.Classes[idx])
        
        return Window_Array, class_idx




def train(model, optimizer, criterion, train_data_loader, val_data_loader, epochs,device, patience = 100, save = 0):
    """
    Just like the name says, this function is used to train the pytorch model. It is quite standard except for "patience" and "save":
    patience: Number of epochs without improvement over the best validation accuracy registered before applying early stopping. If no early stopping is wanted, then make patience = epochs
    save: If save is equal to 0 then nothing happens. If it is different, then the model with the best accuracy in the validation data is saved as: Best_model_<save>.pt
    """
    
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    epochs_without_improvement = 0
    start = time.time()
    Best_acc = 0        
    #print(epochs)
    epochs_without_improvement = 0
    for epoch_num in tqdm(range(epochs), desc = 'Epochs  '):
        model.train()
        total_loss_train = 0
        total_train=0
        success_train=0
        for data in train_data_loader:
            
            optimizer.zero_grad()
                
            inputs = data[0].to(device, dtype=torch.float)
            labels = data[1].to(device, dtype=torch.long)
            outputs = model(inputs)
            
                
            loss = criterion(outputs, labels)
            total_loss_train += loss.item()

            total_train += labels.size(0)
            success_train += (outputs.argmax(dim=1)==labels.data).sum().item()
            model.zero_grad()
            loss.backward()
            optimizer.step()

        train_accs.append(100.0*success_train/total_train)
        train_losses.append(total_loss_train/len(train_data_loader))
        
        model.eval()
        total_loss_val = 0
        total_val=0
        success_val=0
        
        with torch.no_grad():
                
            for data in val_data_loader:
                inputs = data[0].to(device, dtype=torch.float)
                labels = data[1].to(device, dtype=torch.long)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss_val += loss.item()
                total_val += labels.size(0)
                success_val += (outputs.argmax(dim=1)==labels.data).sum().item()

            val_accs.append(100.0*success_val/total_val)
            val_losses.append(total_loss_val/len(val_data_loader))

            epochs_without_improvement += 1
            if val_accs[-1] > Best_acc: #This is the best model so far, then save
                epochs_without_improvement = 0
                Best_acc = val_accs[-1]
                if save != 0: #We want to save 
                    if exists('Best_model_{0}.pt'.format(str(save))):
                        os.remove('Best_model_{0}.pt'.format(str(save)))
                    
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_acc' : val_accs[-1],
                                }, 'Best_model_{0}.pt'.format(str(save)))
        
        if epochs_without_improvement > patience:
            print(f'Early Stopping at epoch {epoch_num}')
            break
                

    return (train_accs, train_losses, val_accs, val_losses, time.time()-start)




def predictions_to_DF(model, test_data_loader,device):
    """
    Creates a dataframe of pairs: (true_class, predicted class) for the elements in the test set.
    """

    
    True_Classes = []
    Outputs = []
    
    model.eval()
    for data in tqdm(test_data_loader, desc = 'Predicting  :'):
        inputs = data[0].to(device, dtype=torch.float)
        Idx = data[1].to(device, dtype=torch.long)
        outs = model(inputs)
        True_Classes.append(Idx)
        Outputs.append(outs)
        
    True_Classes = torch.cat(True_Classes)
    Outs = []
    
    for out in Outputs:
        Outs.append(out.argmax(dim=1))
    
    Outs = torch.cat(Outs)
    
    Final_Outs = [int(x.cpu()) for x in list(Outs)]   
    
    
    df_Predicted = pd.DataFrame({'True Classes':True_Classes.cpu(), 'Predicted Classes': Final_Outs})
    
    return df_Predicted


