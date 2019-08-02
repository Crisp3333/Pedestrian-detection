# Dane Hylton: Machine Learning
# Professor Dr. Kang

import numpy as np
import pandas as pd
import regional_cnn
from PIL import Image

def test():
    person_train = pd.read_csv('train.csv', header=None)
    # person_test = pd.read_csv('INRIAPerson_test.csv', header=None)

    # train_y = person_train.iloc[:, 0]
    #train_x = person_train.drop(person_train.columns[0], axis=1)
    #
    # test_y = person_test.iloc[:, 0]
    # test_x = person_test.drop(person_test.columns[0], axis=1)

    # To numpy array
    train = person_train.values
    print train.shape

    # train = tr_x
    # print train.shape
    # # To python list
    # tr_x = train_x.values.tolist()
    # te_x = test_x.values.tolist()
    # tr_y = train_y.values.tolist()
    # te_y = test_y.values.tolist()

    # print tr_x
    #print tr_x

    cnn = regional_cnn.RegionalCNN(train, filters=32, step=32)
    result = cnn.train()
    print result
    print "result pedestrian detected: %s, not detected %s" % (result[0][0], result[0][1])
    # evaluate = cnn.evaluate()
    # print(evaluate)


# Call function
test()
