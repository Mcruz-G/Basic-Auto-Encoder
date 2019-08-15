import matplotlib.pyplot as plt 

def viewAcc(history):
    #Observe accuracy evolution
    f = plt.figure(figsize=(10,10))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Set-Acc', 'Validation Set-Acc'], loc='upper left')
    plt.show()

def viewLoss(history):
    # summarize history for loss
    f = plt.figure(figsize=(10,10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Set-Loss', 'Validation Set-Loss'], loc='upper left')
    plt.show()

def corr_visual(dataSet,title):
    import matplotlib.pyplot as plt
    import copy

    #Use -7 because the last 7 columns represent a whole categorical variable (which is the target).
    df = copy.deepcopy(dataSet) 

    f = plt.figure(figsize=(10, 10))
    plt.matshow(df.corr(), fignum=f.number)
    # plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
    # plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=14)
    plt.title(title, fontsize=16);