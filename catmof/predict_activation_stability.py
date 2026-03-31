import os
import shutil
import pandas as pd
import numpy as np
from functools import partial
from keras.callbacks import EarlyStopping
import sklearn
import keras
import keras.backend as K
import sklearn.preprocessing
from sklearn.neighbors import BallTree
from scipy.stats import pearsonr, spearmanr
from molSimplify.python_nn.clf_analysis_tool import get_entropy, get_layer_outputs
from data.atomic_data import RACs, geo
from catmof.paths import get_paths

def normalize_data(df_train, df_test, fnames, lname, debug=False):

    _df_train = df_train.copy().dropna(subset=fnames+lname)
    _df_test = df_test.copy().dropna(subset=fnames)
    X_train, X_test = _df_train[fnames].values, _df_test[fnames].values

    if debug:
        print("training data reduced from %d -> %d because of nan." % (len(df_train), len(_df_train)))
        print("testing data reduced from %d -> %d because of nan." % (len(df_test), len(_df_test)))

    x_scaler = sklearn.preprocessing.StandardScaler()
    x_scaler.fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)

    return X_train, X_test, x_scaler

def optimize(X, y, y_name,
             regression=False, hyperopt_step=200,
             arch=False, epochs=2000,
             X_val=False, y_val=False,
             model=False, path=False):
    
    np.random.seed(1234)
    if arch == False:
        architectures = [(200, 200),
                         (300, 300),
                         (500, 500),
                         (100, 100, 100),
                         (200, 200, 200),
                         (300, 300, 300),
                         (500, 500, 500)]
    else:
        architectures = [arch]
    batches = [10, 20, 30, 50, 100, 200, 300, 500]
    space = {'lr': hp.uniform('lr', 1e-5, 1e-3),
             'drop_rate': hp.uniform('drop_rate', 0, 0.5),
             'reg': hp.loguniform('reg', np.log(1e-5), np.log(5e-1)),
             'batch_size': hp.choice('batch_size', batches),
             'hidden_size': hp.choice('hidden_size', architectures),
             'beta_1': hp.uniform('beta_1', 0.75, 0.99),
             'decay': hp.loguniform('decay', np.log(1e-5), np.log(1e-1)),
             'amsgrad': True,
             'patience': 200,
             }
    objective_func = partial(train_model_hyperopt,
                             X=X,
                             y=y,
                             lname=y_name,
                             regression=regression,
                             epochs=epochs,
                             X_val=X_val,
                             y_val=y_val,
                             model=model,
                             path=path)
    trials = Trials()
    best_params = fmin(objective_func,
                       space,
                       algo=tpe.suggest,
                       trials=trials,
                       max_evals=hyperopt_step,
                       rstate=np.random.RandomState(0)
                       )
    best_params.update({'hidden_size': architectures[best_params['hidden_size']],
                        'batch_size': batches[best_params['batch_size']],
                        'amsgrad': True,
                        'patience': 200,
                        })
    
    return trials, best_params

def standard_labels(df, key="flag"):

    flags = [1 if row[key] == 1 else 0 for _, row in df.iterrows()]
    df[key] = flags

    return df

def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())

    return precision

def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())

    return recall

def f1(y_true, y_pred):

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return 2 * ((p * r) / (p + r + K.epsilon()))

def dist_neighbor(fmat1, fmat2, labels, l=10, dist_ref=1):
    tree = BallTree(fmat2, leaf_size=2, metric='cityblock')
    dist_mat, inds = tree.query(fmat1, l)
    dist_mat = dist_mat * 1.0 / dist_ref
    dist_avrg = np.mean(dist_mat, axis=1)
    labels_list = labels[inds]
    return dist_avrg, dist_mat, labels_list

def get_lse(model, df_train, X_train, X_test, neighbors=10):

    train_latent = get_layer_outputs(model, 12, X_train, training_flag=False)
    test_latent = get_layer_outputs(model, 12, X_test, training_flag=False)

    __, nn_dists, nn_labels = dist_neighbor((train_latent.astype(float)), (train_latent.astype(float)), 
                                            np.array([np.array(df_train["flag"].tolist())]).T, l=neighbors, dist_ref=1)
    
    avg = np.mean(nn_dists)

    __, nn_dists_core_mofs, nn_labels_core_mofs = dist_neighbor((test_latent.astype(float)), (train_latent.astype(float)), 
                                                            np.array([np.array(df_train["flag"].tolist())]).T, l=neighbors, dist_ref=avg)
    
    lse = get_entropy(nn_dists_core_mofs, nn_labels_core_mofs)

    return lse

def main():
    p = get_paths()
    base_dir = p.mof_stability_dir
    os.makedirs(base_dir, exist_ok=True)
    shutil.copy(p.merged_features_featurizable_csv, p.merged_features_in_stability_dir)
    path_to_models = p.stability_models_dir
    df_train = pd.read_csv(path_to_models + '/solvent/solvent_train.csv')
    df_train = df_train.loc[:, (df_train != df_train.iloc[0]).any()]
    df_test = pd.read_csv(p.merged_features_in_stability_dir)
    features = [val for val in df_train.columns.values if val in RACs+geo]

    df_train = standard_labels(df_train, key="flag")

    X_train, X_test, _ = normalize_data(df_train, df_test, features, ["flag"], debug=False)

    dependencies = {'precision':precision, 'recall':recall,'f1':f1}
    model = keras.models.load_model(path_to_models + '/solvent/final_model_flag_few_epochs.h5',custom_objects=dependencies)
    test_pred = np.round(model.predict(X_test))

    lse = get_lse(model, df_train, X_train, X_test)

    df_test['predicted'] = test_pred
    df_test['probability'] = model.predict(X_test)
    df_test['lse'] = lse
    df_test.to_csv(p.activation_predictions_csv, index=False)
    print(model.summary())

    return

if __name__ == "__main__":
    main()