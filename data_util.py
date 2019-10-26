import sklearn
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def preprocess_cardio_dataset():
    raw_dataset_location = 'raw_datasets/'
    # cardio_coloumns = ['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','cardio']
    cardio_dataset = 'cardiovascular-disease-dataset.csv'
    raw_dataframe = pd.read_csv(raw_dataset_location+cardio_dataset, sep=';', index_col = 'id')
    # print(raw_dataframe.head(3))


    print(raw_dataframe.describe(include='all'))

    # print('we are here')
    after_blood_presure_cleanned = clean_blood_pressure(raw_dataframe)

    print(after_blood_presure_cleanned.describe(include='all'))
    pandas_series_to_density(after_blood_presure_cleanned['height'], 'height after clean up')

    one_hot_encoded = pd.get_dummies(after_blood_presure_cleanned, columns = [ 'smoke', 'alco', 'active'])
    


    y_true = one_hot_encoded['cardio'].to_numpy()

    one_hot_encoded = one_hot_encoded.drop('cardio', axis=1)
    # one_hot_encoded = one_hot_encoded.drop('gender', axis=1)
    

    X = one_hot_encoded.to_numpy()
    # print(one_hot_encoded)
    
    return X, y_true

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
    pandas_series_to_density(df['ap_hi'], 'ap_hi after clean up')
    pandas_series_to_density(df['ap_lo'], 'ap_lo after clean up')

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

def model_test(best_clf, x_test, y_test):
    print('we are here')
    return


def cross_validation(clfs, X, y_true, num_fold = 10):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=num_fold, random_state=10)
    best_clfs = []
    for clf in clfs:
        best_f_score = 0
        best_clf = None
        for train_index, val_index in skf.split(X, y_true):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y_true[train_index], y_true[val_index]

            clf.fit(X_train, y_train)
            acc, precision, recall, f_score = model_evaluation(clf, X_val, y_val)
            # print('acc')
            print(acc, precision, recall, f_score)
            if best_f_score < f_score:
                best_clf = clf
                best_f_score = f_score
        best_clfs.append(best_clf)

    return best_clfs

def main():
    from sklearn.model_selection import train_test_split
    from sklearn import tree
    from sklearn.naive_bayes import GaussianNB
    X, y_true = preprocess_cardio_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.33, random_state=42)
    
    # print(y_true)

    print('number of instances',len(X))
    print('number of positive classes: ',sum(y_true))
    clfs = []
    # clfs.append(tree.DecisionTreeClassifier())

    clfs.append(GaussianNB())

    best_clfs = cross_validation(clfs, X_test, y_test)
    # model_test(best_clfs, X_test, y_test)

if __name__ == "__main__":
    # execute only if run as a script
    main()