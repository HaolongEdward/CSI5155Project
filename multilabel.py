import data_util
from tensorflow.keras.utils import to_categorical
import numpy as np
# this only works for binary classification

def model_evaluation(clf, x_val, y_val, cardio_dict, gender_dict):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    y_pred = clf.predict(x_val)

    y_pred_cardio = y_pred[:, :2]
    y_pred_gender = y_pred[:, 2:]

    y_preds = [y_pred_cardio, y_pred_gender]

    y_val_cardio = y_val[:, :2]
    y_val_gender = y_val[:, 2:]

    y_vals = [y_val_cardio, y_val_gender]



    acc = []
    precision = []
    recall = []
    f_score = []

    for i in range(len(y_preds)):
        y_pred = y_preds[i]
        y_val = y_vals[i]
        if i == 0:
           label_dict = cardio_dict
        else:
           label_dict = gender_dict

        # wrong_instances = []
        y_predict_classes_onthot = []
        y_val_classes_onthot = []
        if label_dict != None:
            for i in range(len(y_pred)):
                maxElement = np.where(y_pred[i] == np.amax(y_pred[i]))[0][0]
                temp = '['
                temp += '0. ' * (maxElement)
                temp += '1.'
                temp += ' 0.' * (2-maxElement-1)
                temp += ']'
                y_predict_classes_onthot.append(label_dict[temp])
                y_val_classes_onthot.append(label_dict[np.array2string(y_val[i])])
                # print(y_predict_classes_onthot)

            y_pred = y_predict_classes_onthot
            y_val = y_val_classes_onthot

        # for i in range(len(y_pred)):
        #     if y_pred[i] != y_val[i]:
        #         wrong_instances.append(backtrack_dict[i])

        # print(y_val)
        # print(y_pred)

        acc       += [accuracy_score (y_val, y_pred)]
        precision += [precision_score(y_val, y_pred)]
        recall    += [recall_score   (y_val, y_pred)]
        f_score   += [f1_score       (y_val, y_pred)]
    return (acc, precision, recall, f_score)

# def where_i_did_wrong():

# http://scikit.ml/api/skmultilearn.model_selection.iterative_stratification.html
def cross_validation(clfs, X, y_true, cardio_dict, gender_dict, num_fold = 10):
    from sklearn.model_selection import StratifiedKFold
    from prettytable import PrettyTable
    from tensorflow.keras.utils import to_categorical
    from skmultilearn.model_selection import IterativeStratification

    skf = IterativeStratification(n_splits=num_fold, random_state=10)
    wrong_instances_clf = {}
    for clf_name in clfs:

        avg_matrics_cardio = [0,0,0,0]
        avg_matrics_gender = [0,0,0,0]


        print('we are trying classifier: ', clf_name)
        clf = clfs[clf_name]
        best_f_score = 0
        best_clf = None
        i = 1

        t_cardio = PrettyTable(['Fold', 'acc', 'precision', 'recall', 'f_score'])
        # t_cardio.title = 'cardio'

        t_gender = PrettyTable(['Fold', 'acc', 'precision', 'recall', 'f_score'])
        # t_gender.title = 'gender'

        wrong_instances = []
        for train_index, val_index in skf.split(X, y_true):

            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y_true[train_index], y_true[val_index]


            backtrack_dict = {}
            for index in range(len(val_index)):
                backtrack_dict[index] = val_index[index]

            clf.fit(X_train, y_train,
                  verbose = 0,
                  epochs=100,
                  batch_size=8192)               
            acc, precision, recall, f_score = model_evaluation(clf, X_val, y_val, cardio_dict, gender_dict)
            data_util.reset_weights(clf)
            
            # wrong_instances.extend(wrong_instances_fold)

            # print(' '+str(i)+' ', acc, precision, recall, f_score)
            t_cardio.add_row([' '+str(i)+' ', '{0:.3f}'.format(acc[0]), '{0:.3f}'.format(precision[0]), '{0:.3f}'.format(recall[0]), '{0:.3f}'.format(f_score[0])])
            t_gender.add_row([' '+str(i)+' ', '{0:.3f}'.format(acc[1]), '{0:.3f}'.format(precision[1]), '{0:.3f}'.format(recall[1]), '{0:.3f}'.format(f_score[1])])

            avg_matrics_cardio [0] += acc[0]
            avg_matrics_cardio [1] += precision[0]
            avg_matrics_cardio [2] += recall[0]
            avg_matrics_cardio [3] += f_score[0]
            avg_matrics_gender [0] += acc[1]
            avg_matrics_gender [1] += precision[1]
            avg_matrics_gender [2] += recall[1]
            avg_matrics_gender [3] += f_score[1]
            i += 1
        i-=1
        t_cardio.add_row(['avg','{0:.3f}'.format(avg_matrics_cardio[0]/i),'{0:.3f}'.format(avg_matrics_cardio[1]/i),'{0:.3f}'.format(avg_matrics_cardio[2]/i),'{0:.3f}'.format(avg_matrics_cardio[3]/i)])
        t_gender.add_row(['avg','{0:.3f}'.format(avg_matrics_gender[0]/i),'{0:.3f}'.format(avg_matrics_gender[1]/i),'{0:.3f}'.format(avg_matrics_gender[2]/i),'{0:.3f}'.format(avg_matrics_cardio[3]/i)])

        # print('avg',avg_matrics[0]/i,avg_matrics[1]/i,avg_matrics[2]/i,avg_matrics[3]/i)
        # wrong_instances_clf[clf_name] = wrong_instances
        print('+--------------------------------------------+')
        print('|                  cardio                    |')
        print('+--------------------------------------------+')
        print(t_cardio)
        print('+--------------------------------------------+')
        print('|                   gender                   |')
        print('+--------------------------------------------+')
        print(t_gender)

    return 


def main():
    from tensorflow.keras import optimizers
    X, y_true, feature_names = data_util.preprocess_cardio_dataset(
    normalization = True, 
    exclusive_all_did_wrong=True,
    is_clean_blood_pressure = True,
    save = False, new_filename = None,
    PCA = False,
    is_using_onehot = True,
    include_gender = True,
    secondary_label = 'gender')


    y_true_cardio = y_true[0]
    y_true_gender = y_true[1]

    y_true_gender[y_true_gender == 2] = 0
    
    mlp = data_util.getMLP(X.shape[-1], num_class = 2)

    mlp.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr = 0.001),
              metrics=['accuracy'])

    clfs = {'MLP':mlp}
    print('we are trying to classify label \'gender\'')
    print('+--------------------------------------------+')
    print('|                   gender                   |')
    print('+--------------------------------------------+')
    data_util.cross_validation(clfs, X, y_true_gender, num_fold = 10)



    mlp = data_util.getMLP(X.shape[-1], num_class = 2)

    mlp.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr = 0.001),
              metrics=['accuracy'])

    clfs = {'MLP':mlp}
    print('we are trying to classify label \'cardio\'')   
    print('+--------------------------------------------+')
    print('|                  cardio                    |')
    print('+--------------------------------------------+') 
    data_util.cross_validation(clfs, X, y_true_cardio, num_fold = 10)



    y_true_cardio_onehot = to_categorical(y_true_cardio)
    y_true_gender_onehot = to_categorical(y_true_gender)




    unique, counts = np.unique(y_true_gender_onehot, return_counts=True)

    cardio_dict = data_util.get_label_dict(y_true_cardio, y_true_cardio_onehot, 2)
    gender_dict = data_util.get_label_dict(y_true_gender, y_true_gender_onehot, 2)


    y_true = np.concatenate((y_true_cardio_onehot, y_true_gender_onehot), axis=1)


    mlp = data_util.getMLP(X.shape[-1], num_class = 4)

    mlp.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr = 0.001),
              metrics=['accuracy'])
    print('let the game start')
    clfs = {'MLP_multilabel':mlp}
    cross_validation(clfs, X, y_true, cardio_dict, gender_dict, num_fold = 10)




if __name__ == "__main__":
    # execute only if run as a script
    main()