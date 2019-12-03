import sklearn
import pandas as pd

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from prettytable import PrettyTable
import ensemble
def preprocess_cardio_dataset(
    normalization = True, 
    exclusive_all_did_wrong=True,
    is_clean_blood_pressure = True,
    save = False, new_filename = None,
    PCA = False,
    is_using_onehot = True,
    include_gender = False,
    secondary_label = None):
    raw_dataset_location = 'datasets/'
    # cardio_coloumns = ['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','cardio']
    cardio_dataset = 'cardiovascular-disease-dataset.csv'
    X = pd.read_csv(raw_dataset_location+cardio_dataset, sep=';', index_col = 'id')
    # print(raw_dataframe.head(3))


    # print(X.describe(include='all'))

    # print('we are here')
    if is_clean_blood_pressure:
        X = clean_blood_pressure(X)
    # print(after_blood_presure_cleanned.describe(include='all'))



    # pandas_series_to_density(after_blood_presure_cleanned['height'], 'height after clean up')
    # pandas_series_to_density(after_blood_presure_cleanned['height'], 'height_density')
    # pandas_series_to_density(after_blood_presure_cleanned['weight'], 'weight_density')
    


    if exclusive_all_did_wrong:
        all_did_wrong = outlier_exp()
        X = excluding_all_did_wrong(X, all_did_wrong)


    if is_using_onehot:
        X = pd.get_dummies(X, columns = [ 'smoke', 'alco', 'active'])
    
    
    feature_names = list(X.columns.values)
    # print(X.describe(include='all'))
    # print(feature_names)

    feature_dict = {}
    for i in range(len(feature_names)):
        feature_dict[i] = feature_names[i]

    # print(one_hot_encoded)
    # print('*******************************************')
    # print('* PCA on the dataset after one_hot_encode *')
    # print('*******************************************')
    # my_PCA(X, n_components=13)
    # print('*******************************************')
    # print('* PCA on the dataset after normalization  *')
    # print('*******************************************')
    # my_PCA(normalized, n_components=13)

    # X = one_hot_encoded
    y_true = X['cardio'].to_numpy()

    if not include_gender:
        X.drop('gender', axis=1)

    X = X.drop('cardio', axis=1)

    if secondary_label != None:
        y_true_secondary = X[secondary_label].to_numpy()
        X = X.drop(secondary_label, axis=1)
        y_true = [y_true, y_true_secondary]
    


    feature_names = list(X.columns.values)

    if normalization:
        print('*******************************************')
        print('*     We are using normlized dataset      *')
        print('*******************************************')

        feature_names = list(X.columns.values)

        X = df_MinMaxNormalization(X, feature_min=-1, feature_max=1)
        
        # y_true[y_true == -1] = 0

        X = pd.DataFrame(X)
        X = X.rename(columns=feature_dict)
    else:
        print('*******************************************')
        print('*   We are NOT using normlized dataset    *')
        print('*******************************************')

    if PCA:
        print('*******************************************')
        print('*     We are applying PCA on dataset      *')
        print('*******************************************')
        feature_names = list(X.columns.values)

        X = my_PCA(X, n_components=len(X.columns))
        # my_PCA(X, n_components=5)

    else:
        print('*******************************************')
        print('*   We are not applying PCA on dataset    *')
        print('*******************************************')


    X = pd.DataFrame(X)
    feature_names = list(X.columns.values)

    if save:
        X.to_csv('datasets/'+new_filename+'.csv')
    return X.to_numpy(), y_true, feature_names

def excluding_all_did_wrong(df, all_did_wrong):
    feature_names = list(df.columns.values)
    df = df.to_numpy()
    feature_dict = {}
    for i in range(len(feature_names)):
        feature_dict[i] = feature_names[i]
    df = np.delete(df, all_did_wrong, axis=0)

    # print(len(df))

    df = pd.DataFrame(df)
    df = df.rename(columns=feature_dict)
    print('!!!after excluding all did wrong instances!!!')
    print(df.describe(include='all'))
    return df

    
def load_dataset(name, exclusive_all_did_wrong = False):
    dataset_location = 'datasets/'
    df = pd.read_csv(dataset_location+name, index_col = 0)
    
    print(df.describe(include='all'))
    feature_names = list(df.columns.values)

    y_true = df['cardio'].to_numpy()
    y_true[y_true == -1] = 0
    # print(y_true)
    df = df.drop('cardio', axis=1)
    df = df.drop('gender', axis=1)


    if exclusive_all_did_wrong:
        all_did_wrong = outlier_exp()
        excluding_all_did_wrong(df, all_did_wrong)

    
    return df.to_numpy(), y_true, feature_names

def df_MinMaxNormalization(np_arr, feature_min, feature_max):
	from sklearn import preprocessing
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(feature_min, feature_max))
	x_scaled = min_max_scaler.fit_transform(np_arr)
	return x_scaled

def my_PCA(np_arr, n_components):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca.fit(np_arr) 
    # print()
    print('The raw PCA variance ratio is:', pca.explained_variance_ratio_)
    for i in range(n_components):
        if pca.explained_variance_ratio_[i] < 0.01:
            print('Selecting the first', i, 'components')
            pca = PCA(n_components=i+1)
            pca.fit(np_arr) 
            print('PCA variance ratio after selecting is:', pca.explained_variance_ratio_)
            return pca.transform(np_arr)
    return pca.transform(np_arr)

def clean_blood_pressure(df):
    ap_hi_lower_bound = 60
    ap_lo_lower_bound = 40
    ap_hi_upper_bound = 250
    ap_lo_upper_bound = 140
    count_ap_hi_lower_than_lower_bound = 0
    count_ap_hi_higher_than_upper_bound = 0
    count_ap_low_lower_than_lower_bound = 0
    count_ap_low_higher_than_upper_bound = 0
    count_ap_hi_lower_ap_lo = 0


    # pandas_series_to_density(df['ap_hi'], 'ap_hi before clean up')
    # pandas_series_to_density(df['ap_lo'], 'ap_lo before clean up')

    for index, row in df.iterrows():
        # print(type(row['ap_hi']))
        if row['ap_hi'] < row['ap_lo']:
            count_ap_hi_lower_ap_lo += 1
            # print(row)
            df = df.drop([index])
            continue
        if row['ap_hi'] < ap_hi_lower_bound:
            count_ap_hi_lower_than_lower_bound += 1
            df = df.drop([index])
            continue
        if row['ap_lo'] < ap_lo_lower_bound:
            count_ap_low_lower_than_lower_bound += 1
            df = df.drop([index])
            continue
        if row['ap_hi'] > ap_hi_upper_bound:
            count_ap_hi_higher_than_upper_bound += 1
            df = df.drop([index])
            continue
        if row['ap_lo'] > ap_lo_upper_bound:
            count_ap_low_higher_than_upper_bound += 1
            df = df.drop([index])
            continue
        
    print("There are ", count_ap_hi_lower_than_lower_bound, ' rows which ap_hi lower than ', ap_hi_lower_bound)
    print("There are ", count_ap_hi_higher_than_upper_bound, ' rows which ap_hi larger than ', ap_hi_upper_bound)
    print("There are ", count_ap_low_lower_than_lower_bound, ' rows which ap_lo lower than ', ap_lo_lower_bound)
    print("There are ", count_ap_low_higher_than_upper_bound, ' rows which ap_lo larger than ', ap_lo_upper_bound)
    print("There are ", count_ap_hi_lower_ap_lo, ' rows which ap_lo is higher than ap_hi')
    # pandas_series_to_density(df['ap_hi'], 'ap_hi after clean up')
    # pandas_series_to_density(df['ap_lo'], 'ap_lo after clean up')

    return df

def pandas_series_to_density(series, file_name):
    
    import matplotlib.pyplot as plt
    # print(type(series))
    fig = series.plot.kde()
    plt.savefig('density_plot/'+file_name+'.pdf')
    plt.clf()

    # fig.clf()

def get_label_dict(Y, Y_onehot, num_classes):
    label_dict = dict()
    i = 0
    while num_classes != 0:
        # print(i)
        key = np.array2string(Y_onehot[i])

        if key not in label_dict:
            num_classes -= 1
            label_dict[key] = Y[i]
        i += 1
    return label_dict

# this only works for binary classification
def model_evaluation(clf, x_val, y_val, backtrack_dict, label_dict = None):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    y_pred = clf.predict(x_val)

    wrong_instances = []
    y_predict_classes_onthot = []
    if label_dict != None:
        for i in range(len(y_pred)):
            maxElement = np.where(y_pred[i] == np.amax(y_pred[i]))[0][0]
            temp = '['
            temp += '0. ' * (maxElement)
            temp += '1.'
            temp += ' 0.' * (2-maxElement-1)
            temp += ']'
            y_predict_classes_onthot.append(label_dict[temp])
            # print(y_predict_classes_onthot)

        y_pred = y_predict_classes_onthot
    for i in range(len(y_pred)):
        if y_pred[i] != y_val[i]:
            wrong_instances.append(backtrack_dict[i])

    
    acc       = accuracy_score (y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall    = recall_score   (y_val, y_pred)
    f_score   = f1_score       (y_val, y_pred)
    return (acc, precision, recall, f_score, wrong_instances)

# def where_i_did_wrong():
def reset_weights(model):
    
    import tensorflow as tf

    session = tf.compat.v1.keras.backend.get_session(op_input_list=())
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

def cross_validation(clfs, X, y_true, num_fold = 10):
    from sklearn.model_selection import StratifiedKFold
    from tensorflow.keras.utils import to_categorical

    skf = StratifiedKFold(n_splits=num_fold, random_state=10)
    wrong_instances_clf = {}

    f1_records = {}

    for clf_name in clfs:

        avg_matrics = [0,0,0,0]

        print('we are trying classifier: ', clf_name)
        clf = clfs[clf_name]
        best_f_score = 0
        best_clf = None
        i = 1
        t = PrettyTable(['Fold', 'acc', 'precision', 'recall', 'f_score'])

        wrong_instances = []
        for train_index, val_index in skf.split(X, y_true):

            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y_true[train_index], y_true[val_index]


            backtrack_dict = {}
            for index in range(len(val_index)):
                backtrack_dict[index] = val_index[index]

            if clf_name == 'MLP':
                y_train_org = y_train
                # y_val_org = y_val
                y_train = to_categorical(y_train)
                # y_val = to_categorical(y_val)
                train_label_dict = get_label_dict(y_train_org, y_train, 2)
                # val_label_dict = get_label_dict(y_val_org, y_val, 2)

                clf.fit(X_train, y_train,
                  verbose = 0,
                  epochs=100,
                  batch_size=8192)               
                acc, precision, recall, f_score, wrong_instances_fold = model_evaluation(clf, X_val, y_val, backtrack_dict, label_dict = train_label_dict)
                reset_weights(clf)
            else:
                clf.fit(X_train, y_train)
                acc, precision, recall, f_score, wrong_instances_fold = model_evaluation(clf, X_val, y_val, backtrack_dict)
            wrong_instances.extend(wrong_instances_fold)

            # print(' '+str(i)+' ', acc, precision, recall, f_score)
            t.add_row([' '+str(i)+' ', '{0:.3f}'.format(acc), '{0:.3f}'.format(precision), '{0:.3f}'.format(recall), '{0:.3f}'.format(f_score)])

            avg_matrics [0] += acc
            avg_matrics [1] += precision
            avg_matrics [2] += recall
            avg_matrics [3] += f_score
            i += 1
        i-=1
        t.add_row(['avg','{0:.3f}'.format(avg_matrics[0]/i),'{0:.3f}'.format(avg_matrics[1]/i),'{0:.3f}'.format(avg_matrics[2]/i),'{0:.3f}'.format(avg_matrics[3]/i)])

        # print('avg',avg_matrics[0]/i,avg_matrics[1]/i,avg_matrics[2]/i,avg_matrics[3]/i)
        wrong_instances_clf[clf_name] = wrong_instances
        print(t)
        f1_records[clf_name] = avg_matrics[3]/i

    return wrong_instances_clf, f1_records

def getMLP(input_dim, num_class):
    # from keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras import backend
    from tensorflow.keras.utils import multi_gpu_model

    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(16, activation='relu'))

    model.add(Dense(num_class, activation='softmax'))

    model = multi_gpu_model(model, gpus=2)

    return model


def DL_exp(x_train, x_test, y_train, y_test):
    from keras.utils import to_categorical
    from keras import optimizers
    model = getMLP(x_train.shape[-1], num_class = 2)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr = 0.001),
              metrics=['accuracy'])

    model.fit(x_train, y_train,
          verbose = 1,
          epochs=1,
          batch_size=8192)
    score = model.evaluate(x_test, y_test, batch_size=128)
    print(score)

def ML_exp(X, y_true, feature_names):
    
    from sklearn import tree
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from skrules import SkopeRules
    from tensorflow.keras import optimizers

    clfs = {}
    clfs['KNN'] = KNeighborsClassifier(n_neighbors=3)
    clfs['DT'] = tree.DecisionTreeClassifier()
    clfs['NB'] = GaussianNB()
    clfs['RB'] = SkopeRules(max_depth_duplication=None,
                     n_estimators=30,
                     precision_min=0.6,
                     recall_min=0.01,
                     feature_names=feature_names)

    mlp = getMLP(X.shape[-1], num_class = 2)
    mlp.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr = 0.001),
              metrics=['accuracy'])
    clfs['MLP'] = mlp

    clfs['Voting'] = ensemble.get_VotingClassifier_ensemble_model(feature_names)
    boosting_clfs = ensemble.get_ada_boosting_clfs(feature_names)
    for key in boosting_clfs:
        clfs[key] = boosting_clfs[key]

    wrong_instances_clf, f1_records = cross_validation(clfs, X, y_true)
    return wrong_instances_clf, f1_records
    

def exp(preprocess_again = True, exclusive_all_did_wrong = False, normalization = True, PCA = False, is_clean_blood_pressure = False, is_using_onehot = True):
    from sklearn.model_selection import train_test_split
    dataset = 'normalized.csv'
    # X, y_true, feature_names = preprocess_cardio_dataset(normalization = True)

    # exclusive_all_did_wrong = False


    if preprocess_again:
        X, y_true, feature_names = preprocess_cardio_dataset(
            normalization = normalization, 
            is_clean_blood_pressure = is_clean_blood_pressure, 
            exclusive_all_did_wrong=exclusive_all_did_wrong,
            is_using_onehot = is_using_onehot,
            save = False, new_filename = None)
    else:
        X, y_true, feature_names = load_dataset(dataset, exclusive_all_did_wrong)



    # X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.33, random_state=42)

    print('number of instances',len(X))
    unique, counts = np.unique(y_true, return_counts=True)
    
    print('classes distribution: ', dict(zip(unique, counts)))
    wrong_instances_clf, f1_records = ML_exp(X, y_true, feature_names)
    if not exclusive_all_did_wrong:
        with open('wrong_instances_clf.txt', 'w+') as f:
            for key in wrong_instances_clf:
                f.write("%s:\n" % key)
                f.write("%s\n" % wrong_instances_clf[key])
                print(key, 'classifier wrongly classified',len(wrong_instances_clf[key]), 'instances')
    return f1_records

def outlier_exp(tolerate = 0):
    file_name = 'wrong_instances_clf'
    wrong_instances_clf = {}
    num_clf = 0
    all_instances = {}
    all_did_wrong = []
    with open('wrong_instances_clf.txt', 'r') as f:
        content = f.readlines()
        for line in content:
            if line[0] == '[':
                wrong_instances = [int(s) for s in line[1:-2].split(',')]
                for wrong_instance in wrong_instances:
                    all_instances[wrong_instance] = all_instances.get(wrong_instance, 0) + 1
            else:
                num_clf += 1
    for instance in all_instances:
        if all_instances[instance] >= num_clf - tolerate:
            all_did_wrong += [instance]

    print('There are', len(all_did_wrong), 'instances wrongly classified by at least', num_clf - tolerate, 'classifier')
    return all_did_wrong


def format_row(f1_records, keys):
    second_half_row = []

    for key in keys:
        second_half_row.append('{0:.3f}'.format(f1_records[key]))

    return second_half_row

def truth_combination(num_of_variable):
    import itertools
    l=[False,True]
    return [list(i) for i in itertools.product(l,repeat=num_of_variable)]


def main():

    # outlier_exp()
    # preprocess_cardio_dataset(
    #     normalization = True, 
    #     exclusive_all_did_wrong=False,
    #     save = False, new_filename = None,
    #     PCA = True,
    #     secondary_label = 'gender')
    testcases = truth_combination(5)

    cols =  ['is_clean_blood_pressure', 'normalization', 'PCA', 'exclusive_all_did_wrong', 'is_using_onehot']

    f1_records = exp(preprocess_again = True, is_clean_blood_pressure = testcases[0][0], normalization = testcases[0][1], PCA = testcases[0][2], exclusive_all_did_wrong = testcases[0][3], is_using_onehot = testcases[0][4])
    keys = list(f1_records.keys())
    # print(keys)
    cols.extend(keys)
    final_result = PrettyTable(cols)

    final_result.add_row([testcases[0][0], testcases[0][1], testcases[0][2], testcases[0][3], testcases[0][4]] + format_row(f1_records, keys))
    
    for i in range(1, len(testcases)):
        f1_records = exp(preprocess_again = True, is_clean_blood_pressure = testcases[i][0], normalization = testcases[i][1], PCA = testcases[i][2], exclusive_all_did_wrong = testcases[i][3], is_using_onehot = testcases[i][4])    
        final_result.add_row([testcases[i][0], testcases[i][1], testcases[i][2], testcases[i][3], testcases[i][4]] + format_row(f1_records, keys))


    
    print(final_result)

if __name__ == "__main__":
    # execute only if run as a script
    main()
















