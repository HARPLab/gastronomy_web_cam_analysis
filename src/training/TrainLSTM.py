from feature_utils import *
import RestaurantFrame
FLAG_SVM = False
#fid evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
        verbose, epochs, batch_size = 0, 1, 64
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
        model = Sequential()
        model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("fiting model....")
        plot_losses = TrainingPlot()
        model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_data=(testX, testy), callbacks=[plot_losses])
        try:
                pickle.dump(model, open("LSTM_4train-1test-1epochs.p", "wb"))
        except:
               print("error")
        _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
        return accuracy

# summarize scores
def summarize_results(scores):
        print(scores)
        m, s = mean(scores), std(scores)
        print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(trainX, trainy, testX, testy, repeats=1):
        # repeat experiment
        scores = list()
        for r in range(repeats):
                score = evaluate_model(trainX, trainy, testX, testy)
                score = score * 100.0
                print('>#%d: %.3f' % (r+1, score))
                scores.append(score)
        # summarize results
        summarize_results(scores)
# run the experiment
if not FLAG_SVM:
        print("training LSTM")

        X_train = []
        Y_train = []

        X_test = []
        Y_test = []

        #shuffled_list = copy.copy(timeline)
        """
        shuffled_list = pickle.load(open("13-17-18-21_data.pickle","rb"))["timeline"] #copy.copy(timeline)
        #pickle.dump(shuffled_list, open("8-21-18_data_list.p", "wb"))
        test_list = pickle.load(open("8-21-18_data.pickle", "rb"))["timeline"]
        print("loaded pickle datasets")
        index_reduc = int(len(shuffled_list) * (0.4))
        shuffled_list = shuffled_list[:index_reduc]
        #percent_test = .2
        #index_split = int(len(shuffled_list) * (1.0 - percent_test))
        #train = shuffled_list[:index_split]
        #test = shuffled_list[index_split:]
        train = shuffled_list
        index_reduc = int(len(test_list) * (0.4))
        test = test_list[:index_reduc]
        for frame in train:
                newX, newY, ret = frame_to_vectors(frame)
                if not ret:
                        continue
                X_train.extend(newX)
                Y_train.append(newY)
        for frame in test:
                newX, newY, ret = frame_to_vectors(frame)
                if not ret:
                        continue
                X_test.extend(newX)
                Y_test.append(newY)
        print("slicing...")
        
        window_size=128
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        X_test = np.asarray(X_test)
        Y_test = np.asarray(Y_test)
        print(X_train.shape)
        #pickle.dump(X_train, open("X_train.p", "wb"), protocol=4)
        #pickle.dump(X_test, open("X_test.p", "wb"), protocol=4)
        #pickle.dump(Y_train, open("Y_train.p", "wb"), protocol=4)
        #pickle.dump(Y_test, open("Y_test.p", "wb"), protocol=4)
         """ 
        X_train = pickle.load(open("X_train.p", "rb"))
        Y_train = pickle.load(open("Y_train.p", "rb"))
        X_test = pickle.load(open("X_test.p", "rb"))
        Y_test = pickle.load(open("Y_test.p", "rb"))
        
        
        window_size=128         
        X_train_sliced = np.zeros((X_train.shape[0]-window_size, window_size, X_train.shape[1]))
        Y_train_sliced = np.zeros((Y_train.shape[0]-window_size, 1))
        X_test_sliced = np.zeros((X_test.shape[0]-window_size, window_size, X_test.shape[1]))
        Y_test_sliced = np.zeros((Y_test.shape[0]-window_size, 1))

        for idx in range(window_size, X_train.shape[0]):
                X_train_sliced[idx-window_size,:,:] = X_train[idx-window_size:idx]
        for idx in range(window_size, Y_train.shape[0]):
                Y_train_sliced[idx-window_size,0] = Y_train[idx]
        for idx in range(window_size, X_test.shape[0]):
                X_test_sliced[idx-window_size,:] = X_test[idx-window_size:idx]
        for idx in range(window_size, Y_test.shape[0]):
                Y_test_sliced[idx-window_size,0] = Y_test[idx]
        Y_train_sliced = to_categorical(Y_train_sliced)
        Y_test_sliced = to_categorical(Y_test_sliced)
        print(X_train_sliced.shape, Y_train_sliced.shape, X_test_sliced.shape, Y_test_sliced.shape)
        #pickle.dump(X_train_sliced, open("X_train_sliced.p", "wb"), protocol=4)
        #pickle.dump(X_test_sliced, open("X_test_sliced.p", "wb"), protocol=4)
        #pickle.dump(Y_train_sliced, open("Y_train_sliced.p", "wb"), protocol=4)
        #pickle.dump(Y_test_sliced, open("Y_test_sliced.p", "wb"), protocol=4)
        try:
                model = pickle.load(open("LSTMblah_4train-1test-70epochs.p", "rb"))
                #pred_Y = model.predict(X_test_sliced)
                #pickle.dump(pred_Y, open("pred_Y_LSTM70epochs.p", "wb"))
                pred_Y = pickle.load(open("pred_Y_LSTM70epochs.p", "rb"))
                decodedpredY = pred_Y.argmax(axis=1)
                decodedtestY = Y_test_sliced.argmax(axis=1)
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
                print(decodedpredY.shape)
                print(decodedtestY.shape)
                predPadding = []
                testPadding = []
                i = 0
                for key in activitydict.keys():
                        print(key)
                        predPadding.append(i)
                        testPadding.append(23-i)
                        i +=1
                decodedpredY = np.append(decodedpredY, predPadding)
                decodedtestY = np.append(decodedtestY, testPadding)
                print(decodedpredY.shape)
                print(decodedtestY.shape)
                cm = confusion_matrix(decodedtestY,decodedpredY)
                print("unormalized confusion matrix")
                np.set_printoptions(precision=2)
                fig, ax = plt.subplots()
                sum_of_rows = cm.sum(axis=1)
                cm = cm / (sum_of_rows[:, np.newaxis]+1e-8)
                print(cm)
                plot_confusion_matrix(cm,cmap=plt.cm.Blues)
                plt.savefig("LSTM4170epochs_confusion_mat.png")
        except:
                run_experiment(X_train_sliced, Y_train_sliced, X_test_sliced, Y_test_sliced)
                print("error")

