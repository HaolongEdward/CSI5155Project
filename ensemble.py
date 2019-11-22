import data_util
import numpy as np

#Begin of Fish adds on ensemble model
def VotingClassifier_ensemble_model(X, y_true, feature_names):
    from sklearn import ensemble
    from sklearn.ensemble import VotingClassifier
    from sklearn import tree
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from skrules import SkopeRules

    KNN = KNeighborsClassifier(n_neighbors=3)
    decision_tree = tree.DecisionTreeClassifier()
    NB=GaussianNB()
    RB=SkopeRules(max_depth_duplication=None,
                     n_estimators=30,
                     precision_min=0.6,
                     recall_min=0.01,
                     feature_names=feature_names)
    eclf1 = VotingClassifier(estimators=[('KNN', KNN), ('DT', decision_tree), ('NB', NB)], voting='hard')
    # model_1.fit(X_train,y_train)
    # model_2.fit(X_train,y_train)
    # model_3.fit(X_train,y_train)
    # model_4.fit(X_train,y_train)

    # pred1=model_1.predict(X_test)
    # pred2=model_2.predict(X_test)
    # pred3=model_3.predict(X_test)
    # pred4=model_4.predict(X_test)

    # final_pred = np.array([])
    # print("Ensemble model: Voting System")
    # for i in range(0,len(X_test)):
    #     final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i],pred4[i]]))
    # print(final_pred)
    # return final_pred


    clfs = {'voteing-1':eclf1}
    data_util.cross_validation(clfs, X, y_true)
#end of fish edit part

def ada_boosting_exp(X, y_true, feature_names):

    clfs = {}
    from sklearn import tree
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from skrules import SkopeRules
    from sklearn.ensemble import AdaBoostClassifier

    # clfs['KNN'] = KNeighborsClassifier(n_neighbors=3)
    clfs['Decision Tree'] = tree.DecisionTreeClassifier()
    clfs['naive_bayes'] = GaussianNB()
    # clfs['SkopeRules'] = SkopeRules(max_depth_duplication=None,
    #                  n_estimators=30,
    #                  precision_min=0.6,
    #                  recall_min=0.01,
    #                  feature_names=feature_names)

    for clf_name in clfs:
        clfs[clf_name] = AdaBoostClassifier(
            base_estimator = clfs[clf_name],
            n_estimators=10, random_state=0)
    data_util.cross_validation(clfs, X, y_true)


def main():


    X, y_true, feature_names = data_util.preprocess_cardio_dataset(
        normalization = True,
        exclusive_all_did_wrong=False)

    VotingClassifier_ensemble_model(X, y_true, feature_names)
    # ada_boosting_exp(X, y_true, feature_names)

if __name__ == '__main__':
    main()