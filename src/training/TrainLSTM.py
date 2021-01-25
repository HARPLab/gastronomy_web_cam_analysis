from feature_utils import *
import RestaurantFrame
FLAG_SVM = False
import time
#fid evaluate a model
def calc_precision_recall(cm,logfile):
    #print(cm.shape)
    p = []
    r = []
    p_denoms = cm.sum(axis=0)
    r_denoms = cm.sum(axis=1)
    for class_lbl in range(0, cm.shape[1]):
        precision = cm[class_lbl][class_lbl]/p_denoms[class_lbl]
        recall = cm[class_lbl][class_lbl]/r_denoms[class_lbl]
        logfile.write(activitydict_rev[class_lbl] + " precision: " + str(precision) + " recall: " + str(recall) + "\n")
        print(activitydict_rev[class_lbl] + " precision: " + str(precision) + " recall: " + str(recall))
        p.append(precision)
        r.append(recall)
    return p, r
class LSTMTrainer:
    def __init__(self, xtrain, ytrain, xtest, ytest, logfile):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.logfile = logfile
        model = None
        
    def generate_cm(self, model, filename):
        pred_Y = model.predict(self.xtest)
        #pickle.dump(pred_Y, open("pred_Y_LSTM1epochs.p", "wb"))
        #pred_Y = pickle.load(open("pred_Y_LSTM70epochs.p", "rb"))
        decodedpredY = pred_Y.argmax(axis=1)
        decodedtestY = self.ytest.argmax(axis=1)
        frequencytest = {}
        frequencypred = {}
        for num in decodedtestY:
                if num not in frequencytest.keys():
                        frequencytest[num] = 1
                else:
                        frequencytest[num] = frequencytest[num] + 1
        for num in decodedpredY:
                if num not in frequencypred.keys():
                        frequencypred[num] = 1
                else:
                        frequencypred[num] = frequencypred[num] + 1
        #print("stats:")
        #print(frequencytest)
        #print(frequencypred)
        #print(decodedpredY.shape)
        #print(decodedtestY.shape)
        predPadding = []
        testPadding = []
        i = 0
        for key in activitydict.keys():
                #print(key)
                predPadding.append(i)
                testPadding.append(23-i)
                i +=1
        decodedpredY = np.append(decodedpredY, predPadding)
        decodedtestY = np.append(decodedtestY, testPadding)
        cm = confusion_matrix(decodedtestY,decodedpredY)
        np.set_printoptions(precision=2)
        fig, ax = plt.subplots()
        sum_of_rows = cm.sum(axis=1)
        cm = cm / (sum_of_rows[:, np.newaxis]+1e-8)
        p, r = calc_precision_recall(cm,self.logfile)
        pickle.dump(cm, open(filename +"cm_mat.p", "wb"))
        plot_confusion_matrix(cm,cmap=plt.cm.Blues)
        plt.savefig(filename + "cm.png")
        plt.close()

    def evaluate_model(self,trainX, trainy, testX, testy, params, model_name):
        verbose, epochs, batch_size, hidden_dim = params['verbose'], params['epochs'], params['batch_size'], params['hidden_dim']
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
        model = Sequential()
        model.add(LSTM(hidden_dim, input_shape=(n_timesteps,n_features)))
        model.add(Dropout(params['dropout']))
        #model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.logfile.write("fiting model...." + model_name)
        print("fiting model...." + model_name)
        #plot_losses = TrainingPlot()
        history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_data=(testX, testy))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(model_name + "Losses.png")
        plt.close()
        model.save(model_name)
        _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
        return accuracy, model

# summarize scores
    def summarize_results(self,scores):
        print(scores)
        m, s = mean(scores), std(scores)
        self.logfile.write('Accuracy: %.3f%% (+/-%.3f)\n' % (m, s))
        print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
    def run_experiment(self, search_space, fold, savepath):
        # repeat experiment for batchsize, dropout, epochs
       scores = list()
       max_score = 0
       best_config = None
       #for dropout in list(np.linspace(0.1,0.8,8)):
       dropout=0.1
       if search_space:
           for epochs in [150]:
               for hidden_dim in [700, 800, 1000]: # 320, 350, 400
                   params = {'verbose': 0, 'epochs': epochs, 'batch_size':256,'dropout':dropout, 'hidden_dim': hidden_dim}
                   model_name = savepath + "CVLSTM"+str(fold)+"_"+ str(params['epochs']) + str(params['batch_size']) + str(params['dropout'])+str(params['hidden_dim'])
                   score,model  = self.evaluate_model(self.xtrain, self.ytrain, self.xtest, self.ytest, params, model_name)
                   score = score * 100.0
                   scores.append(score)
                   self.generate_cm(model, model_name)
                   self.logfile.write("finished training model " + model_name +" with accuracy of " + str(score) + "\n")
                   print("finished training model " + model_name +" with accuracy of " + str(score))
                   if score > max_score:
                       max_score = score
                       best_config = params
       else:
           dropout = 0.1
           batch_size = 256
           hidden_dim = 80
           params = {'verbose': 0, 'epochs': 70, 'batch_size':batch_size,'dropout':dropout, 'hidden_dim': hidden_dim}
           model_name = savepath + "CVLSTM"+ str(fold)+"_"+str(params['epochs']) + str(params['batch_size']) + str(params['dropout'])+str(params['hidden_dim'])
           score,model  = self.evaluate_model(self.xtrain, self.ytrain, self.xtest, self.ytest, params, model_name)
           score = score * 100.0
           scores.append(score)
           self.generate_cm(model, model_name)
           print("finished training model " + model_name +" with accuracy of " + str(score))
           if score > max_score:
               max_score = score
               best_config = params
       self.summarize_results(scores)
       self.logfile.write("best config: " + str(params) + "\n")
       print("best config: " + str(params))
       return max_score
# run the experiment
def RestarauntFrames2Vec(filename, training_suff):
    X_train = []
    Y_train = []

    X_test = []
    Y_test = []
    shuffled_list = pickle.load(open(filename,"rb"))#["timeline"] #copy.copy(timeline)
    #pickle.dump(shuffled_list, open("8-21-18_data_list.p", "wb"))
    #test_list = pickle.load(open("13-21_sift_data.pickle", "rb"))["timeline"]
    print("loaded pickle datasets")
    #index_reduc = int(len(shuffled_list) * (0.4))
    #shuffled_list = shuffled_list[:index_reduc]
    #percent_test = .2
    #index_split = int(len(shuffled_list) * (1.0 - percent_test))
    #train = shuffled_list[0:index_split]
    #test = shuffled_list[index_split:-1]
    #train = shuffled_list
    #index_reduc = int(len(test_list) * (0.4))
    #test = test_list[:index_reduc]
    #X_train = np.zeros((1, 150))
    #X_test = np.zeros((1,150))
    for frame in shuffled_list:
        newX, newY, ret = frame_to_vectors(frame)
        if not ret:
            continue
        X_train.extend(newX)
        Y_train.extend(newY)
    """
    for frame in train:
        newX, newY, ret = frame_to_vectors(frame)
        if not ret:
            continue
        X_train.extend(newX)
        Y_train.extend(newY)
    for frame in test:
        newX, newY, ret = frame_to_vectors(frame)
        if not ret:
            continue
        X_test.extend(newX)
        Y_test.extend(newY) 
    """
    #X_train = np.array([np.array(xi) for xi in X_train])#np.array(X_train)
    X_train = np.array(X_train) 
    Y_train = np.asarray(Y_train)
    #X_test = np.array(X_test)
    #X_test = np.array([np.array(xi) for xi in X_test])#np.array(X_test)
    #Y_test = np.asarray(Y_test)
    #pickle.dump(X_train, open("training_sets/X_train_" + training_suff +".p", "wb"))
    #pickle.dump(Y_train, open("training_sets/Y_train_" + training_suff +".p", "wb"))
    #pickle.dump(X_test, open("training_sets/X_test_" + training_suff +".p", "wb"))
    #pickle.dump(Y_test, open("training_sets/Y_test_" + training_suff +".p", "wb"))
    return X_train, Y_train #X_train, Y_train, X_test, Y_test# X_test, Y_test

def shuffle_in_unison(set_a, set_b):
    assert len(set_a) == len(set_b)
    p = np.random.permutation(len(set_a))
    return set_a[p], set_b[p]

def slice_vectors(X_train, Y_train, training_suff, window_size=128):#, X_test, Y_test, window_size=128):
    X_train_sliced = np.zeros((X_train.shape[0]-window_size, window_size, X_train.shape[1]))
    Y_train_sliced = np.zeros((Y_train.shape[0]-window_size, 1))
    #X_test_sliced = np.zeros((X_test.shape[0]-window_size, window_size, X_test.shape[1]))
    #Y_test_sliced = np.zeros((Y_test.shape[0]-window_size, 1))

    for idx in range(window_size, X_train.shape[0]):
        X_train_sliced[idx-window_size,:,:] = X_train[idx-window_size:idx]
    for idx in range(window_size, Y_train.shape[0]):
        Y_train_sliced[idx-window_size,0] = Y_train[idx]
    #for idx in range(window_size, X_test.shape[0]):
    #    X_test_sliced[idx-window_size,:] = X_test[idx-window_size:idx]
    #for idx in range(window_size, Y_test.shape[0]):
    #    Y_test_sliced[idx-window_size,0] = Y_test[idx]
    Y_train_sliced = to_categorical(Y_train_sliced)
    #Y_test_sliced = to_categorical(Y_test_sliced)
    #combined_X_data = np.concatenate((X_train_sliced, X_test_sliced), axis=0)
    #combined_Y_data = np.concatenate((Y_train_sliced, Y_test_sliced), axis=0)
    X_sliced, Y_sliced = shuffle_in_unison(X_train_sliced, Y_train_sliced)
    
    train_list = []
    test_list = []
    percent_test = .2
    k = int(1.0/percent_test)
    for fold in range(0, 1):    
        index_split = int(len(X_sliced) * (1.0 - percent_test))
        lower = int(fold*percent_test*len(X_sliced))
        upper = lower + int(percent_test*len(X_sliced))
        X_train_sliced = np.concatenate((X_sliced[0:lower], X_sliced[upper:]), axis=0)
        Y_train_sliced = np.concatenate((Y_sliced[0:lower], Y_sliced[upper:]), axis=0)
        X_test_sliced = X_sliced[lower:upper]
        Y_test_sliced = Y_sliced[lower:upper]
        train_list.append((X_train_sliced,Y_train_sliced))
        test_list.append((X_test_sliced,Y_test_sliced))
        print(X_train_sliced.shape, Y_train_sliced.shape, X_test_sliced.shape, Y_test_sliced.shape)
    #pickle.dump(X_train_sliced, open("training_sets/X_train_sliced_" + training_suff +".p", "wb"))
    #pickle.dump(Y_train_sliced, open("training_sets/Y_train_sliced_" + training_suff +".p", "wb"))
    #pickle.dump(X_test_sliced, open("training_sets/X_test_sliced_" + training_suff +".p", "wb"))
    #pickle.dump(Y_test_sliced, open("training_sets/Y_test_sliced_" + training_suff +".p", "wb"))
    #return X_train_sliced, Y_train_sliced, X_test_sliced, Y_test_sliced
    pickle.dump(train_list, open("training_sets/train_list_" + training_suff +".p", "wb"),protocol=4)
    pickle.dump(test_list, open("training_sets/test_list_" + training_suff +".p", "wb"),protocol=4)

    return train_list, test_list
training_suffix = "processed_ab_a"
modelpath = "models/AB_A/"
X_train, Y_train, X_test, Y_test = None, None, None, None
log = open("training_log", "a")
try:
    train_file = open("training_sets/train_list_" + training_suffix +".p", "rb")
    test_file = open("training_sets/test_list_" + training_suffix +".p", "rb")
    train_list = pickle.load(train_file)
    test_list =  pickle.load(test_file)
    train_file.close()
    test_file.close()
    """
    Y_train = pickle.load(open("training_sets/Y_train_sliced_" + training_suffix +".p", "rb"))
    X_test = pickle.load(open("training_sets/X_test_sliced_" + training_suffix +".p", "rb"))
    Y_test = pickle.load(open("training_sets/Y_test_sliced_" + training_suffix +".p", "rb"))
    """
except:
    print("generating sliced dataset...")
    X_train, Y_train = RestarauntFrames2Vec("training_sets/13-17-18-21_data_processed.pickle", training_suffix)#, X_test, Y_test = RestarauntFrames2Vec("13-17-18-21_data_processed.pickle", training_suffix)
    train_list, test_list = slice_vectors(X_train, Y_train, training_suffix, window_size=128)
#xt, yt, xtest, ytest = slice_vectors(X_train, Y_train, X_test, Y_test[0:X_test.shape[0]], window_size=128)
#train_list, test_list = slice_vectors(X_train, Y_train, training_suff, window_size=128)#, X_test, Y_test[0:X_test.shape[0]], window_size=128)

acc_sum = 0.0
for i in range(0, 1):# len(train_list)):
    log.write("training for " +str(i) + "'th fold of data\n")
    print("training for " +str(i) + "'th fold of data")
    xt, yt = train_list[i] 
    print(yt[0])
    xtest, ytest = test_list[i]
    trainer = LSTMTrainer(xt, yt, xtest, ytest,log)
    start_time = time.time()
    log.write("start_time: " + str(start_time))
    print("start_time: " + str(start_time)+"\n")
    acc = trainer.run_experiment(True, i, modelpath)
    #score, model = trainer.evaluate_model(xt, yt, xtest, ytest, {'verbose': 0, 'epochs': 1, 'batch_size': 64})
    end_time = time.time()
    acc_sum += acc
    log.write("end_time: " + str(end_time) + "\n")
    print("end_time: " + str(end_time))
    log.write("minutes elapsed: " + str((end_time-start_time)/60) + "\n")
    print("minutes elapsed: " + str((end_time-start_time)/60) )
log.write("avg acc: " + str(float(acc_sum/float(len(train_list)))) + "\n")
log.close()
print("avg acc: " + str(float(acc_sum/float(len(train_list)))))
