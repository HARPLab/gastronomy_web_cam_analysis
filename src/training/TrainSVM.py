import * from feature_utils.py
print("training SVM...")
        X_train = []
        Y_train = []

        X_test = []
        Y_test = []


        #shuffled_list = copy.copy(timeline)
        #random.shuffle(shuffled_list)
        shuffled_list = pickle.load(open("13-17-18-21_list.p","rb"))
        test_list = pickle.load(open("8-21-18_data_list.p", "rb"))
        #percent_test = .2
        #index_split = int(len(shuffled_list) * (1.0 - percent_test))
        index_reduc = int(len(shuffled_list) * (0.4))
        shuffled_list = shuffled_list[:index_reduc]

        #train = #shuffled_list[:index_split]
        #test = shuffled_list[index_split:]
        train = shuffled_list
        index_reduc = int(len(test_list) * (0.4))
        test = test_list[:index_reduc]
        for frame in train:
                newX, newY, ret = frame_to_vectors(frame)
                #print(newX)
                #print(newY)
                if not ret:
                        continue
                X_train.extend(newX)
                Y_train.append(newY)
        for frame in test:
                newX, newY, ret = frame_to_vectors(frame)
                #print(newX)
                #print(newY)
                if not ret:
                        continue
                X_test.extend(newX)
                Y_test.append(newY)


        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        X_test = np.asarray(X_test)
        Y_test = np.asarray(Y_test)

        print("X_train")
        print(X_train.shape)
        print("Y_train")
        print(Y_train.shape)
        predicted = []
        correct = Y_test
        try:
                clf = pickle.load(open("svm41_trained.p", "rb"))
                print("Successfully imported pickle svm")
        except (OSError, IOError) as e:
                clf = svm.SVC()
                print("fitting...")
                clf.fit(X_train, Y_train)
                print("dumping...")
                pickle.dump(clf, open("svm41_trained.p", "wb"))

