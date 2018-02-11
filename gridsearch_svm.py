from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import multiprocessing as mp

from svm import SVM
from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from sklearn.model_selection import ParameterGrid


def run_one_candidate(dataset, candidate_params, test_on_valid_set=True):
    print("========== Tune parameters for classification ==========")

    np.random.seed(1000)
    list_results = []
    for it in range(2):
        learner = SVM(
            trade_off=1.0,
            gamma=candidate_params['gamma'],
            batch_size=10,
            rf_dim=400,
            learning_rate=candidate_params['learning_rate'],
            num_epochs=20,
        )

        if test_on_valid_set:
            x_train_all, y_train_all = load_svmlight_file(dataset + '.txt')
            x_train_all = x_train_all.toarray()
            total_samples = x_train_all.shape[0]
            mask = np.zeros(total_samples, dtype=bool)
            num_train_samples = int(0.8 * total_samples)
            mask[np.random.permutation(total_samples)[num_train_samples]] = True
            x_train = x_train_all[mask, :]
            y_train = y_train_all[mask]
            mask = not mask
            x_test = x_train_all[mask, :]
            y_test = y_train_all[mask]
        else:
            x_train, y_train = load_svmlight_file(dataset + '.txt')
            x_test, y_test = load_svmlight_file(dataset + '_t.txt')
            x_train = x_train.toarray()
            x_test = x_test.toarray()

        y_train[y_train == 0] = -1
        y_test[y_test == 0] = -1

        if test_on_valid_set:
            learner.fit(x_train, y_train, x_test, y_test)
        else:
            learner.fit(x_train, y_train)

        y_test_predict = learner.predict(x_test)
        test_acc = metrics.accuracy_score(y_test, y_test_predict)
        # print('Test accuracy:', test_acc)
        list_results.append(test_acc)

    print('Test acc:', list_results)
    return np.mean(np.array(list_results)), candidate_params

test_acc_lst = []
run_param_lst = []


def log_result(result):
    test_mean_acc, params = result
    test_acc_lst.append(test_mean_acc)
    run_param_lst.append(params)


def run_grid_search_multicore(dataset):
    params_gridsearch = {
        'gamma': 2.0**np.arange(-5, 15, 4),
        'learning_rate': [1e-3, 1e-5],
    }

    candidate_params_lst = list(ParameterGrid(params_gridsearch))

    pool = mp.Pool(2)  # maximum of workers
    result_lst = []
    for candidate_params in candidate_params_lst:
        result = pool.apply_async(
            run_one_candidate,
            args=(dataset, candidate_params),
            callback=log_result
        )
        result_lst.append(result)

    for result in result_lst:
        result.get()
    pool.close()
    pool.join()

    print("========== FINAL RESULT ==========")

    idx_best = np.argmax(np.array(test_acc_lst))
    print('Best acc on valid set: {}'.format(test_acc_lst[idx_best]))
    print('Best params: {}'.format(run_param_lst[idx_best]))

if __name__ == '__main__':
    run_grid_search_multicore('svmguide1')
