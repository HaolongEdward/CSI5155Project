import sklearn
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
def preprocess_cardio_dataset(normalization = True):
    raw_dataset_location = 'raw_datasets/'
    # cardio_coloumns = ['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','cardio']
    cardio_dataset = 'cardiovascular-disease-dataset.csv'
    raw_dataframe = pd.read_csv(raw_dataset_location+cardio_dataset, sep=';', index_col = 'id')
    # print(raw_dataframe.head(3))


    print(raw_dataframe.describe(include='all'))

    # print('we are here')
    after_blood_presure_cleanned = clean_blood_pressure(raw_dataframe)

    print(after_blood_presure_cleanned.describe(include='all'))
    # pandas_series_to_density(after_blood_presure_cleanned['height'], 'height after clean up')


    one_hot_encoded = pd.get_dummies(after_blood_presure_cleanned, columns = [ 'smoke', 'alco', 'active'])
    


    y_true = one_hot_encoded['cardio'].to_numpy()

    one_hot_encoded = one_hot_encoded.drop('cardio', axis=1)
    one_hot_encoded = one_hot_encoded.drop('gender', axis=1)
    
    feature_names = list(one_hot_encoded.columns.values)

    X = one_hot_encoded.to_numpy()
    # print(one_hot_encoded)
    # print('*******************************************')
    # print('* PCA on the dataset after one_hot_encode *')
    # print('*******************************************')
    # my_PCA(X, n_components=13)
    # print('*******************************************')
    # print('* PCA on the dataset after normalization  *')
    # print('*******************************************')
    # my_PCA(normalized, n_components=13)



    if normalization:
        X = df_MinMaxNormalization(X, feature_min=-1, feature_max=1)
        print('*******************************************')
        print('*     We are using normlized dataset      *')
        print('*******************************************')
    else:
        print('*******************************************')
        print('*   We are NOT using normlized dataset    *')
        print('*******************************************')
    return X, y_true, feature_names


    


def df_MinMaxNormalization(np_arr, feature_min, feature_max):
	from sklearn import preprocessing
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(feature_min, feature_max))
	x_scaled = min_max_scaler.fit_transform(np_arr)
	return x_scaled

def my_PCA(np_arr, n_components):
	from sklearn.decomposition import PCA
	pca = PCA(n_components=n_components)
	pca.fit(np_arr) 
	print(pca.explained_variance_ratio_)

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


    pandas_series_to_density(df['ap_hi'], 'ap_hi before clean up')
    pandas_series_to_density(df['ap_lo'], 'ap_lo before clean up')

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
    
    # print(type(series))
    fig = series.plot.kde()
    plt.savefig('density_plot/'+file_name+'.pdf')
    plt.clf()

    # fig.clf()


# this only works for binary classification
def model_evaluation(clf, x_val, y_val):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    y_pred = clf.predict(x_val)

    acc       = accuracy_score (y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall    = recall_score   (y_val, y_pred)
    f_score   = f1_score       (y_val, y_pred)
    return (acc, precision, recall, f_score)



def cross_validation(clfs, X, y_true, num_fold = 10):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=num_fold, random_state=10)
    for clf_name in clfs:
        avg_matrics = [0,0,0,0]

        print('we are trying classifier: ', clf_name)
        clf = clfs[clf_name]
        best_f_score = 0
        best_clf = None
        i = 1
        for train_index, val_index in skf.split(X, y_true):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y_true[train_index], y_true[val_index]

            clf.fit(X_train, y_train)
            acc, precision, recall, f_score = model_evaluation(clf, X_val, y_val)
            # print('acc')
            print(' '+str(i)+' ', acc, precision, recall, f_score)
            avg_matrics [0] += acc
            avg_matrics [1] += precision
            avg_matrics [2] += recall
            avg_matrics [3] += f_score
            i += 1
        i-=1
        print('avg',avg_matrics[0]/i,avg_matrics[1]/i,avg_matrics[2]/i,avg_matrics[3]/i)

    return 

def getMLP(input_dim, num_class):
    # from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras import backend as K
    from keras.utils import multi_gpu_model

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
          epochs=100,
          batch_size=8192)
    score = model.evaluate(x_test, y_test, batch_size=128)
    print(score)

def ML_exp(X_train, X_test, y_train, y_test, feature_names):
    clfs = {}
    from sklearn import tree
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from skrules import SkopeRules

    clfs['KNN'] = KNeighborsClassifier(n_neighbors=3)
    clfs['Decision Tree'] = tree.DecisionTreeClassifier()
    clfs['naive_bayes'] = GaussianNB()
    clfs['SkopeRules'] = SkopeRules(max_depth_duplication=None,
                     n_estimators=30,
                     precision_min=0.2,
                     recall_min=0.01,
                     feature_names=feature_names)

    best_clfs = cross_validation(clfs, X_train, y_train)

#Begin of Fish adds on ensemble model
def ensemble_model(X_train, X_test, y_train, y_test, feature_names):
    from sklearn import ensemble
    from sklearn.ensemble import VotingClassifier
    from sklearn import tree
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from skrules import SkopeRules

    model_1=KNeighborsClassifier(n_neighbors=3)
    model_2=tree.DecisionTreeClassifier()
    model_3=GaussianNB()
    model_4=SkopeRules(max_depth_duplication=None,
                     n_estimators=30,
                     precision_min=0.2,
                     recall_min=0.01,
                     feature_names=feature_names)

    model_1.fit(X_train,y_train)
    model_2.fit(X_train,y_train)
    model_3.fit(X_train,y_train)
    model_4.fit(X_train,y_train)

    pred1=model_1.predict(X_test)
    pred2=model_2.predict(X_test)
    pred3=model_3.predict(X_test)
    pred4=model_4.predict(X_test)

    final_pred = np.array([])
    print("Ensemble model: Voting System")
    for i in range(0,len(X_test)):
        final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i],pred4[i]]))
    print(final_pred)
    return final_pred
#end of fish edit part

def main():
    from sklearn.model_selection import train_test_split
    
    X, y_true, feature_names = preprocess_cardio_dataset(normalization = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.33, random_state=42)

    print('number of instances',len(X))
    print('number of positive classes: ',sum(y_true))
    # ML_exp(X_train, X_test, y_train, y_test, feature_names)
    # DL_exp(X_train, X_test, y_train, y_test)
    ensemble_model(X_train, X_test, y_train, y_test, feature_names)

    


if __name__ == "__main__":
    # execute only if run as a script
    main()






