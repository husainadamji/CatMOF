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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
from numpy.random import seed
seed(1)
from data.atomic_data import RACs, geo
from catalmof.paths import get_paths

def normalize_data(df_train, df_test, fnames, lname, debug=False):

    _df_train = df_train.copy().dropna(subset=fnames+lname)
    _df_test = df_test.copy().dropna(subset=fnames)
    X_train, X_test = _df_train[fnames].values, _df_test[fnames].values
    y_train = _df_train[lname].values

    if debug:
        print("training data reduced from %d -> %d because of nan." % (len(df_train), len(_df_train)))
        print("test data reduced from %d -> %d because of nan." % (len(df_test), len(_df_test)))

    x_scaler = sklearn.preprocessing.StandardScaler()
    x_scaler.fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_scaler = sklearn.preprocessing.StandardScaler()
    y_scaler.fit(y_train)
    y_train = y_scaler.transform(y_train)

    return X_train, X_test, x_scaler, y_scaler

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

def get_latent_distances(model, df_train, X_train, df_test, X_test):

    get_latent = K.function([model.layers[0].input],
                            [model.layers[8].output])
    training_latent = get_latent([X_train])[0]
    test_latent = get_latent([X_test])[0]

    test_LD = pairwise_distances(test_latent, training_latent, n_jobs=30)
    test_LD_df = pd.DataFrame(data=test_LD, index=df_test['name'].tolist(), columns=df_train['CoRE_name'].tolist())

    return test_LD_df

def get_ten_nn(test_LD_df, base_dir, scaling_factor=26.139462100000003, debug=False):

    ten_nn = []
    namelist = []
    for i, row in test_LD_df.iterrows():
        namelist.append(i)
        ten_closest = np.argsort(np.array(row.values))[0:10]
        vals = np.array(row.values)[ten_closest]
        ten_nn.append(float(np.mean(vals)))

    scaled_ten_nn = (np.array(ten_nn) / scaling_factor).tolist()

    ten_nn_df = pd.DataFrame()
    ten_nn_df['CoRE_name'] = namelist
    ten_nn_df['latent_10NN'] = ten_nn
    ten_nn_df['scaled_latent_10NN'] = scaled_ten_nn
    ten_nn_df = ten_nn_df.sort_values(by='CoRE_name')

    if debug == True:
        ten_nn_df.to_csv(base_dir + '/ten_nn.csv', index=False)

    return ten_nn_df

def main():
    p = get_paths()
    base_dir = p.mof_stability_dir
    os.makedirs(base_dir, exist_ok=True)
    # When running thermal-only (activation bypassed), merged features may not be in stability dir yet
    if not os.path.isfile(p.merged_features_in_stability_dir):
        shutil.copy(p.merged_features_featurizable_csv, p.merged_features_in_stability_dir)
    path_to_models = p.stability_models_dir
    df_train = pd.read_csv(path_to_models + '/thermal/thermal_train.csv')
    df_train = df_train.loc[:, (df_train != df_train.iloc[0]).any()]
    df_test = pd.read_csv(p.merged_features_in_stability_dir)
    features = [val for val in df_train.columns.values if val in RACs+geo]

    X_train, X_test, _, y_scaler = normalize_data(df_train, df_test, features, ["T"], debug=False)

    dependencies = {'precision':precision,'recall':recall,'f1':f1}
    model = keras.models.load_model(path_to_models + '/thermal/final_model_T_few_epochs.h5',custom_objects=dependencies)
    test_pred = y_scaler.inverse_transform(model.predict(X_test))

    test_LD_df = get_latent_distances(model, df_train, X_train, df_test, X_test)
    ten_nn_df = get_ten_nn(test_LD_df, base_dir, debug=False)

    df_test['predicted'] = test_pred
    final_test_df = pd.concat([df_test, ten_nn_df], axis=1)
    final_test_df.drop('CoRE_name', axis=1, inplace=True)
    final_test_df.to_csv(p.thermal_predictions_csv, index=False)
    print(model.summary())

    return

if __name__ == "__main__":
    main()
