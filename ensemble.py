import data_util
import numpy as np

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

    X, y_true, feature_names = data_util.preprocess_cardio_dataset(normalization = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.33, random_state=42)

    ensemble_model(X_train, X_test, y_train, y_test, feature_names)

if __name__ == '__main__':
    main()