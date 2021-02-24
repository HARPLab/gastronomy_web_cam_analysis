from feature_utils import *
import RestaurantFrame
FLAG_SVM = False
import time

activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
                        'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
                        'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
                        'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20, 'leaving-table':21, 'cleaning-up':22, 'NONE':23}
FEATURES_SET_PA = 0
FEATURES_SET_PB = 1
FEATURES_SET_BOTH = 2
LABELS_SET_PA = 0
LABELS_SET_PB = 1
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
        logfile.write(activitydict_rev[class_lbl] + " precision: " + str(precision) + " recall: " + str(recall) + " total: " + str(r_denoms[class_lbl]) + "\n")
        print(activitydict_rev[class_lbl] + " precision: " + str(precision) + " recall: " + str(recall) + " total: " + str(r_denoms[class_lbl]))
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
        print("stats:")
        print(frequencytest)
        print(frequencypred)
        self.logfile.write("cm stats:\n")
        self.logfile.write(str(frequencytest) + "\n")
        self.logfile.write(str(frequencypred) + "\n")
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
        print("output dim " + str(n_outputs))
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
           for epochs in [500]:
               for hidden_dim in [800]: # 320, 350, 400
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
def RestarauntFrames2Vec(train_data, training_suff,f_flag, l_flag, test_data = None):
    X_train = []
    Y_train = []

    X_test = []
    Y_test = []
    shuffled_list = pickle.load(open(train_data,"rb"))#["timeline"] #copy.copy(timeline)
    print("loaded pickle datasets")
    for frame in shuffled_list:
        newX, newY, ret = frame_to_vectors(frame, f_flag, l_flag)
        if not ret:
            continue
        X_train.extend(newX)
        Y_train.extend(newY)
    if test_data is not None:
        test_list = pickle.load(open(test_data,"rb"))
        for frame in test_list:
            newX, newY, ret = frame_to_vectors(frame, f_flag, l_flag)
            if not ret:
                continue
            X_test.extend(newX)
            Y_test.extend(newY)
    X_train = np.array(X_train) 
    Y_train = np.asarray(Y_train)
    X_test = np.array(X_test)
    Y_test = np.asarray(Y_test)
    if test_data is None:
        return X_train, Y_train, None, None
    return X_train, Y_train, X_test, Y_test #X_train, Y_train, X_test, Y_test# X_test, Y_test

def shuffle_in_unison(set_a, set_b):
    assert len(set_a) == len(set_b)
    p = np.random.permutation(len(set_a))
    return set_a[p], set_b[p]

def slice_vectors(X_train, Y_train, training_suff, logfile, window_size=128,X_test=None, Y_test=None, clas=None):#, X_test, Y_test, window_size=128):
    test_exclusive = False
    test_set_provided = X_test is not None and Y_test is not None
    x_sliced_list = []
    y_sliced_list = []
    if test_set_provided:
        X_test_sliced = np.zeros((X_test.shape[0]-window_size, window_size, X_test.shape[1]))
        Y_test_sliced = np.zeros((Y_test.shape[0]-window_size, 1))
    label_freqs = {}
    for i in range(0,25):
        label_freqs[i] = 0
    for idx in range(window_size, X_train.shape[0]):
        x_sliced_list.append(X_train[idx-window_size:idx].tolist())
    X_train_sliced = np.array(x_sliced_list)
    for idx in range(window_size, Y_train.shape[0]):
        label_freqs[Y_train[idx]] += 1
        if clas is None or Y_train[idx] == clas:
            #y_sliced_list.append(Y_train[idx].tolist()) 
            y_sliced_list.append([1])
            print("found correct class")
        else:
            y_sliced_list.append([0])
    Y_train_sliced = np.array(y_sliced_list)
    print(X_train_sliced.shape)
    print(Y_train_sliced.shape)
    print(label_freqs)
    print(np.sum(Y_train_sliced))
    logfile.write("class freqs: " + str(label_freqs))
    if test_set_provided:
        for idx in range(window_size, X_test.shape[0]):
            X_test_sliced[idx-window_size,:] = X_test[idx-window_size:idx]
        for idx in range(window_size, Y_test.shape[0]):
            Y_test_sliced[idx-window_size,0] = Y_test[idx]
    if test_exclusive:
        test_set_provided = True
        x_sliced_test_list = []
        y_sliced_test_list = []
        for idx in range(window_size+2, X_train.shape[0]):
            x_sliced_test_list.append(X_train[idx-window_size:idx].tolist())
        for idx in range(window_size+2, Y_train.shape[0]):
            if clas is None or Y_train[idx] == clas:
                y_sliced_test_list.append([1])
            else:
                y_sliced_test_list.append([0])
        Y_test_sliced = np.array(y_sliced_test_list)
        X_test_sliced = np.array(x_sliced_test_list)
    if test_set_provided:
        Y_test_sliced = to_categorical(Y_test_sliced)
        X_test_sliced, Y_test_sliced = shuffle_in_unison(X_test_sliced, Y_test_sliced)
        print("test_set_provided")
        train_list = [(X_sliced,Y_sliced)]
        test_list = [(X_test_sliced, Y_test_sliced)]
        pickle.dump(train_list, open("training_sets/train_list_" + training_suff +".p", "wb"),protocol=4)
        pickle.dump(test_list, open("training_sets/test_list_" + training_suff +".p", "wb"),protocol=4)
        return train_list, test_list
    #Y_train_sliced = to_categorical(Y_train_sliced)
    print(Y_train_sliced)
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
    #pickle.dump(train_list, open("training_sets/train_list_" + training_suff +".p", "wb"),protocol=4)
    #pickle.dump(test_list, open("training_sets/test_list_" + training_suff +".p", "wb"),protocol=4)

    return train_list, test_list
training_suffix = "processed_wn1_b_b_reading_menu"
modelpath = "models/B_B/classes/reading_menu"
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
    X_train, Y_train, X_test, Y_test = RestarauntFrames2Vec("training_sets/13-17-18-21_data_processed.pickle", training_suffix, FEATURES_SET_PB, LABELS_SET_PB, test_data=None)#"training_sets/8-9-18_data_processed.pickle")#, X_test, Y_test = RestarauntFrames2Vec("13-17-18-21_data_processed.pickle", training_suffix)
    train_list, test_list = slice_vectors(X_train, Y_train, training_suffix, log, X_test=X_test, Y_test=Y_test, window_size=64,clas=activitydict["reading:menu"])

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
