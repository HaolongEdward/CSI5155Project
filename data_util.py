import sklearn
import pandas as pd

def preprocess_cardio_dataset():
    raw_dataset_location = 'raw_datasets/'
    cardio_dataset = 'cardiovascular-disease-dataset.csv'
    raw_dataframe = pd.read_csv(raw_dataset_location+cardio_dataset, sep=';', index_col = 'id')

    one_hot_encoded = pd.get_dummies(raw_dataframe, columns = ['gender','cholesterol', 'smoke', 'alco', 'active'])
    y_true = one_hot_encoded['cardio'].to_numpy()

    one_hot_encoded = one_hot_encoded.drop('cardio', axis=1)
    X = one_hot_encoded.to_numpy()
    
    return X, y_true

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

    clfs.append(tree.DecisionTreeClassifier())
    clfs.append(GaussianNB())

    best_clfs = cross_validation(clfs, X_test, y_test)
    model_test(best_clfs, X_test, y_test)

if __name__ == "__main__":
    # execute only if run as a script
    main()