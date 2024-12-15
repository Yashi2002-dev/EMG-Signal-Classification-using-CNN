import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import models  # Module containing CNN models
import utils   # Module with utility functions

# Create training, validation, and testing datasets
path_files = 'EMG_data_for_gestures-master'
train_subjects, val_subjects, test_subjects = utils.divide_Subjects(path_files=path_files, nval=5, ntest=3)  # Random split of subjects

# Debugging purposes: use specific subjects
train_subjects = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15', '17', '19', '20', '21', '22', '24', '25', '26', '27', '28', '29', '31', '32', '34', '35']
val_subjects = ['30', '36', '16', '18', '07']
test_subjects = ['03', '33', '23']

# Normalize training data using MinMaxScaler
scaler = utils.get_scaler(train_subjects, path_files=path_files)

# Create windows of data for training, validation, and testing
df_Train = utils.Windows_df_from_subjects(train_subjects, path_files=path_files, wait_size=120, interleaved=(2,4), scaler=scaler)
df_Val = utils.Windows_df_from_subjects(val_subjects, path_files=path_files, wait_size=120, scaler=scaler)
df_Test = utils.Windows_df_from_subjects(test_subjects, path_files=path_files, wait_size=800, scaler=scaler)

print('Len df_Train with augment by inserts: ', len(df_Train))
print('Len df_Val : ', len(df_Val))
print('Len df_Test : ', len(df_Test))

# Data augmentation using Gaussian noise
SNR = 35  # Signal-to-noise ratio
N_Augmentation = 15
DF_TRAIN = [df_Train]

for _ in range(N_Augmentation):  # Generate augmented datasets
    DF_TRAIN.append(utils.GN_DataAugmentation(df_Train, SNR))

DF_TRAIN = pd.concat(DF_TRAIN).reset_index(drop=True)

# Create PyTorch datasets and dataloaders
BATCH_SIZE = 64
ds_train = utils.Dataset_EMG(DF_TRAIN)
ds_val = utils.Dataset_EMG(df_Val)
ds_test = utils.Dataset_EMG(df_Test)

train_loader = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

# Define hyperparameters and initialize model
LR = 0.035  # Learning rate
EPOCHS = 100  # Maximum epochs
PATIENCE = 20  # Early stopping patience
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.EMG2DClassifier_V0(dropout=0.15).to(device)
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # Optimizer

# Train the model
metrics = utils.train(model, optimizer, criterion, train_loader, val_loader, EPOCHS, device, patience=PATIENCE, save=1)  # Save the best model

# Reload the best model for evaluation
model = models.EMG2DClassifier_V0().to(device)
checkpoint = torch.load("Best_model_V0.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Generate predictions for the test set
preds_Test = utils.predictions_to_DF(model, test_loader, device)

# Calculate and display test accuracy
test_accuracy = len(preds_Test[preds_Test["True Classes"] == preds_Test["Predicted Classes"]]) / len(preds_Test)
print('Test Accuracy : ', test_accuracy)

# Generate confusion matrix for the test set
cm = confusion_matrix(list(preds_Test["True Classes"]), list(preds_Test["Predicted Classes"]), normalize='true')
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True).set(title='Confusion Matrix in Testing', xlabel='Predicted Label', ylabel='True Label')
plt.show()