from timeit import default_timer as timer
import pandas as pd
import numpy as np
import itertools
import time
from sklearn import preprocessing
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import tensorflow as tf

"""
Goal: given different information about purchases of audio books, predict returning customers

Inputs:
- Total books length minutes
- Average books length minutes
- Price overall
- Price average
- Review left
- Review 10/10
- Minutes listened
- Completion percent of books purchased
- Number of support requests
- Difference of 1st and last date visited site

Outputs:
- 1 - came back (purchased a book) in 6 months after 2 years of inputs
- 0 - didn't come back

TODOs:
- 
"""

VALIDATE_FRACTION = 0.1
TEST_FRACTION = 0.1
INPUT_FILE = 'Audiobooks_data.csv'
OUTPUT_TRAIN_FILE = 'Audiobooks_data_train.npz'
OUTPUT_VALID_FILE = 'Audiobooks_data_validation.npz'
OUTPUT_TEST_FILE = 'Audiobooks_data_test.npz'

# Maximum number of epochs.  Currently by having a very high value, practically not used and relying on EarlyStopping.
# See reasons in README.md
MAX_NUM_EPOCHS = 1000

class config:
    loops_per_model = 5

    # validate_loss_improve_deltas = [0.0001, 0.00001]
    validate_loss_improve_deltas = [0.0001]

    # validate_loss_improve_patiences = [7, 10, 15]
    validate_loss_improve_patiences = [7]

    improve_restore_best_weights_values = [True]

    # batch_sizes = [200, 400, 600]
    batch_sizes = [100]

    # hidden_widths = [200, 450, 784]
    hidden_widths = [100]

    nums_layers = [4]

    functions = ['sigmoid', 'tanh', 'relu', 'softmax']
    # functions = ['sigmoid', 'tanh', 'relu']

    #learning_rates = [0.001, 0.0005, 0.00001]  # default in tf.keras.optimizers.Adam is 0.001
    # learning_rates = [0.001, 0.0005]  # default in tf.keras.optimizers.Adam is 0.001
    learning_rates = [0.001]  # default in tf.keras.optimizers.Adam is 0.001

def acquire_preprocess_data():
    np.set_printoptions(formatter={'float': lambda x: "{0: 0.2f}".format(x)}, linewidth=120)

    raw_data = np.loadtxt(INPUT_FILE, delimiter=',')
    data = raw_data[:, 1:]
    # print(f'===================== beginning of data no index:\n{data[:5, :]}')

    return data


def shuffle_data(data):
    shuffled_indices = np.arange(data.shape[0])
    np.random.shuffle(shuffled_indices)
    data = data[shuffled_indices]
    # print(f'===================== beginning of shuffled data:\n{data[:5, :]}')
    return data


def prepare_write_data(data):
    # shuffle before balancing
    data = shuffle_data(data)

    # balance the data to have the same number of 0s and 1s are
    num_one_targets = int(sum(data[:, -1]))

    zero_targets_counter = 0
    indices_to_remove = []

    for i in range(data.shape[0]):
        if data[i][-1] == 0:
            zero_targets_counter += 1
            if zero_targets_counter > num_one_targets:
                indices_to_remove.append(i)

    data = np.delete(data, indices_to_remove, axis=0)
    print(f'num_one_targets: {num_one_targets}, data shape after balancing: {data.shape}')
    # print(f'===================== beginning of balanced data:\n{data[:5, :]}')

    # shuffle again because all the end are 1's
    data = shuffle_data(data)

    # scale the inputs
    preprocessing.scale(data[:, :-1], copy=False)
    # print(f'shapes after scaling: {data.shape}')
    # print(f'===================== beginning of scaled data:\n{data[:5, :]}')

    # split into train, validate, test
    samples_count = data.shape[0]
    validation_samples_count = int(samples_count * VALIDATE_FRACTION)
    test_samples_count = int(samples_count * TEST_FRACTION)
    train_samples_count = samples_count - validation_samples_count - test_samples_count
    print(f'num_samples: {samples_count}, '
          f'num_train: {train_samples_count}, num_valid: {validation_samples_count}, num_test: {test_samples_count}')

    train_inputs = data[ : train_samples_count, :-1]
    train_targets = data[ : train_samples_count, -1]

    validation_inputs = data[train_samples_count: train_samples_count + validation_samples_count, :-1]
    validation_targets = data[train_samples_count: train_samples_count + validation_samples_count, -1]

    test_inputs = data[train_samples_count + validation_samples_count :, :-1]
    test_targets = data[train_samples_count + validation_samples_count:, -1]

    """
    print(f'shapes train: train_inputs: {train_inputs.shape}, train_targets: {train_targets.shape}, '
          f'num 1s fraction: {round(int(sum(train_targets)) / train_targets.shape[0], 2)}')
    print(f'shapes validation: '
          f'validation_inputs: {validation_inputs.shape}, validation_targets: {validation_targets.shape}, '
          f'num 1s fraction: {round(int(sum(validation_targets)) / validation_targets.shape[0], 2)}')
    print(f'shapes test: test_inputs: {test_inputs.shape}, test_targets: {test_targets.shape}, '
          f'num 1s fraction: {round(int(sum(test_targets)) / test_targets.shape[0], 2)}')
    """
    # write to files
    np.savez(OUTPUT_TRAIN_FILE, inputs=train_inputs, targets=train_targets)
    np.savez(OUTPUT_VALID_FILE, inputs=validation_inputs, targets=validation_targets)
    np.savez(OUTPUT_TEST_FILE, inputs=test_inputs, targets=test_targets)


def read_npz_file(file_name):
    npz = np.load(file_name)
    inputs = npz['inputs'].astype(np.float)
    targets = npz['targets'].astype(np.int)
    # print(f'{file_name}: inputs.shape: {inputs.shape}, targets: {targets.shape}')
    return inputs, targets


def read_prepared_data():
    train_inputs, train_targets = read_npz_file(OUTPUT_TRAIN_FILE)
    valid_inputs, valid_targets = read_npz_file(OUTPUT_VALID_FILE)
    test_inputs, test_targets = read_npz_file(OUTPUT_TEST_FILE)

    return train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets


def prepare_data(data):
    # Splitting into writing to file and reading from file, so that in theory in the future can do just the second part,
    #  and work exactly on the same data
    prepare_write_data(data)
    return read_prepared_data()


def prepare_model(in_dic):
    output_size = 2

    # Add input layer
    model = tf.keras.Sequential([])

    # add hidden layers
    for i in range(in_dic['Num layers'] - 2):
        model.add(tf.keras.layers.Dense(in_dic['Hidden width'], activation=in_dic['Hidden funcs'][i]))

    # Add output layer
    model.add(tf.keras.layers.Dense(output_size, activation='softmax'))

    """ Compile the model
    Adam - optimizer that uses both momentum and learning rate schedule - combines best from both worlds
    Loss sparse_categorical_crossentropy - for classification problems that were already one-hoted
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=in_dic['Learning rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


def single_model(train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets, in_dic):
    """
    Perform a run of a single model, returns dictionary with results
    """
    out_dic = {}

    model = prepare_model(in_dic)

    # Stops the model when val_loss doesn't improve by min_delta for patience loops.
    #  Possibly restores weights to best value of val_loss
    early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      min_delta=in_dic['Validate loss improvement delta'],
                                                      patience=in_dic['Validate loss improvement patience'],
                                                      restore_best_weights=in_dic['Restore best weights'])

    start = timer()
    history = model.fit(train_inputs,
                        train_targets,
                        batch_size=in_dic['Batch size'],
                        epochs=MAX_NUM_EPOCHS,  # large value, practically not used
                        callbacks=[early_callback],
                        validation_data=(valid_inputs, valid_targets),
                        verbose=2)
    end = timer()

    # Test the model
    test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)

    actual_num_epochs = len(history.history["val_accuracy"])

    # Prepare output of results - see explanation of outputs in README.md
    out_dic['Train Loss'] = history.history["loss"][-1].round(4)
    out_dic['Validate Loss'] = history.history["val_loss"][-1].round(4)
    out_dic['Test Loss'] = round(test_loss, 4)
    out_dic['Loss * Time'] = round(out_dic['Train Loss'] * (end - start), 4)
    out_dic['Loss Product'] = round(out_dic['Train Loss'] * out_dic['Validate Loss'] * out_dic['Test Loss'], 4)
    out_dic['Train Accuracy'] = history.history["accuracy"][-1].round(4)
    out_dic['Validate Accuracy'] = history.history["val_accuracy"][-1].round(4)
    out_dic['Test Accuracy'] = round(test_accuracy, 4)
    out_dic['Train Time'] = round(end - start, 3)
    out_dic['Num epochs'] = actual_num_epochs
    out_dic['Average epoch time'] = round((end - start) / actual_num_epochs, 4)

    return out_dic

def update_best(type, best_no_type, best_results):
    best_with_type = {'Type': type}
    best_with_type.update(best_no_type)
    best_results.append(best_with_type)

def do_numerous_loops(given_dic=None):
    """ Performs numerous models with different inputs / hyperparameters.
        Outputs to output files - see README.md for explanation """

    results = []
    best_results = []

    # Inputs are given in 2 ways:
    #  1. Mostly for local testing - explicit by giving given_dic, and only 1 run is done (possibly with numerous loops)
    #  2. By not giving given_dic, and then numerous inputs / hyperparameters are taken from the settings
    if given_dic:
        local_loops_per_model = given_dic['Loops per model']
        local_validate_loss_improve_deltas = [given_dic['Validate loss improvement delta']]
        local_validate_loss_improve_patiences = [given_dic['Validate loss improvement patience']]
        local_improve_restore_best_weights_values = [given_dic['Restore best weights']]
        local_batch_sizes = [given_dic['Batch size']]
        local_hidden_widths = [given_dic['Hidden width']]
        local_nums_layers = [given_dic['Num layers']]
        local_functions = [given_dic['Hidden funcs']]
        local_learning_rates = [given_dic['Learning rate']]
    else:
        local_loops_per_model = config.loops_per_model
        local_validate_loss_improve_deltas = config.validate_loss_improve_deltas
        local_validate_loss_improve_patiences = config.validate_loss_improve_patiences
        local_improve_restore_best_weights_values = config.improve_restore_best_weights_values
        local_batch_sizes = config.batch_sizes
        local_hidden_widths = config.hidden_widths
        local_nums_layers = config.nums_layers
        local_functions = config.functions
        local_learning_rates = config.learning_rates

    # For printing purposes, calculate number of models to be done
    #  (not including activation functions that are harder to calculate and also depend on number of layers)
    num_regressions_without_functions = local_loops_per_model * \
                                        len(local_validate_loss_improve_deltas) * \
                                        len(local_validate_loss_improve_patiences) * \
                                        len(local_improve_restore_best_weights_values) * \
                                        len(local_batch_sizes) * \
                                        len(local_hidden_widths) * \
                                        len(local_learning_rates)
    print(f'To perform number regressions: {num_regressions_without_functions} '
          f'with layers: {local_nums_layers} and functions: {local_functions}')

    # acquiring data is done once to save time
    data = acquire_preprocess_data()

    num_model_trainings_total = 0
    time_run_started = timer()
    in_dic = {}

    for loop in range(1, local_loops_per_model + 1):
        in_dic['Loop'] = loop
        num_model_trainings_in_loop = 0
        # initiate best values that will be overridden when finding a good result
        best_train_loss = {'Train Loss': 100000}
        best_validate_loss = {'Validate Loss': 100000}
        best_test_loss = {'Test Loss': 100000}
        best_loss_efficiency = {'Loss * Time': 100000}
        best_loss_product = {'Loss Product': 100000}

        train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets = prepare_data(data)
        for batch_size in local_batch_sizes:
            in_dic['Batch size'] = batch_size
            for validate_loss_improve_delta in local_validate_loss_improve_deltas:
                in_dic['Validate loss improvement delta'] = validate_loss_improve_delta
                for validate_loss_improve_patience in local_validate_loss_improve_patiences:
                    in_dic['Validate loss improvement patience'] = validate_loss_improve_patience
                    for improve_restore_best_weights in local_improve_restore_best_weights_values:
                        in_dic['Restore best weights'] = improve_restore_best_weights
                        for num_layers in local_nums_layers:
                            in_dic['Num layers'] = num_layers
                            # if given specific run, don't do a product of all activation functions
                            # If not given specific run, perform a product of all activation functions
                            #  to put each one of the functions in each hidden layer
                            if not given_dic:
                                funcs_product = itertools.product(local_functions, repeat=(num_layers - 2))
                            else:
                                funcs_product = [given_dic['Hidden funcs']]
                            print(f'funcs_product: {funcs_product}')
                            for hidden_funcs in funcs_product:
                                in_dic['Hidden funcs'] = hidden_funcs
                                for hidden_width in local_hidden_widths:
                                    in_dic['Hidden width'] = hidden_width
                                    for learning_rate in local_learning_rates:
                                        in_dic['Learning rate'] = learning_rate

                                        num_model_trainings_in_loop += 1
                                        num_model_trainings_total += 1

                                        train_inputs, train_targets, \
                                            valid_inputs, valid_targets, \
                                            test_inputs, test_targets = prepare_data(data)

                                        # Printing progress of models, time and loop
                                        time_running_sec = timer() - time_run_started
                                        print(f'loop {loop}/{local_loops_per_model}, '
                                              f'Model in loop {num_model_trainings_in_loop}, '
                                              f'Model total: {num_model_trainings_total}, '
                                              f'total time min: {round(time_running_sec / 60, 1)}, '
                                              f'total time hours: {round(time_running_sec / 60 / 60, 2)}: '
                                              f'seconds per model: {round(time_running_sec / num_model_trainings_total)}'
                                              f' ====================================')
                                        out_dic = single_model(train_inputs, train_targets,
                                                               valid_inputs, valid_targets,
                                                               test_inputs, test_targets,
                                                               in_dic)
                                        # result for purposes of output to file contains both inputs and outputs
                                        result = in_dic.copy()
                                        result.update(out_dic)
                                        results.append(result)
                                        print(f'\nCURRENT: {result}')

                                        if (result['Train Loss']) < best_train_loss['Train Loss']:
                                            best_train_loss = result.copy()
                                        if (result['Validate Loss']) < best_validate_loss['Validate Loss']:
                                            best_validate_loss = result.copy()
                                        if (result['Test Loss']) < best_test_loss['Test Loss']:
                                            best_test_loss = result.copy()
                                        if (result['Loss * Time']) < best_loss_efficiency['Loss * Time']:
                                            best_loss_efficiency = result.copy()
                                        if (result['Loss Product']) < best_loss_product['Loss Product']:
                                            best_loss_product = result.copy()


        print("Finished loop +++++++++++++++++++++++++++++++++++++++++++++")
        print(f'BEST TRAIN LOSS:        {best_train_loss}')
        print(f'BEST VALIDATE LOSS:     {best_validate_loss}')
        print(f'BEST TEST LOSS:         {best_test_loss}')
        print(f'BEST LOSS EFFICIENCY:   {best_loss_efficiency}')
        print(f'BEST LOSS PRODUCT:      {best_loss_product}')

        update_best('TRAIN LOSS', best_train_loss, best_results)
        update_best('VALIDATE LOSS', best_validate_loss, best_results)
        update_best('TEST LOSS', best_test_loss, best_results)
        update_best('LOSS EFFICIENCY', best_loss_efficiency, best_results)
        update_best('LOSS PRODUCT', best_loss_product, best_results)


    # Output all results to full.xlsx
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'Total number of models trained: {num_model_trainings_in_loop} with {local_loops_per_model} loops per model')
    pf = pd.DataFrame(results)
    print(f'ALL RESULTS:')
    print(pf.to_string())
    writer = pd.ExcelWriter(f'output/results_{time.strftime("%Y_%m_%d_%H_%M_%S")}.xlsx')
    pf.to_excel(writer, sheet_name='Full')

    time_running_sec = timer() - time_run_started

    # Output all inputs / hyperparameters to hyperparams.xlsx
    hyperparams = {
        'Num Model Trainings': num_model_trainings_in_loop,
        'Num Loops per model': local_loops_per_model,
        'Total time minutes': round(time_running_sec / 60, 1),
        'Total time hours': round(time_running_sec / 60 / 60, 2),
        'Seconds per model': round(time_running_sec / num_model_trainings_total),
        'Batch sizes': local_batch_sizes,
        'Hidden Widths': local_hidden_widths,
        'Nums layers': local_nums_layers,
        'Functions': local_functions,
        'Learning rates': local_learning_rates,
        'Improvement deltas': local_validate_loss_improve_deltas,
        'Improvement patience': local_validate_loss_improve_patiences,
        'Improvement restore weights': local_improve_restore_best_weights_values}

    pf = pd.DataFrame([hyperparams])
    print(f'HYPERPARAMS:')
    print(pf.to_string())
    pf.to_excel(writer, sheet_name='Hyperparams')

    # Output all best results (in 3 categories, see explanation in README.md) to best.xlsx
    best_test_accuracy_with_type = {'Type': 'TEST ACCURACY'}
    best_test_accuracy_with_type.update(best_test_accuracy)
    best_test_loss_with_type = {'Type': 'TEST LOSS'}
    best_test_loss_with_type.update(best_test_loss)
    best_loss_efficiency_with_type = {'Type': 'TEST LOSS EFFICIENCY'}
    best_loss_efficiency_with_type.update(best_loss_efficiency)

    pf = pd.DataFrame(best_results)
    print(f'BEST RESULTS:')
    print(pf.to_string())
    pf.to_excel(writer, sheet_name='Best')

    writer.save()


""" Ways to run:
1. Regular way - numerous models for later comparison between them.
    - Update settings / hyperparameters options from beginning of file.  Each combination will be done
    - Run do_numerous_loops without params
2. Debug / confirmation / researching one model way:
    - Give explicit dictionary of model to run
    Values given below are for some of the best models
"""
do_numerous_loops()
"""
do_numerous_loops({'Loops per model': 5,
                   'Validate loss improvement delta': 0.0001,
                   'Validate loss improvement patience': 10,
                   'Restore best weights': True,
                   'Batch size': 450,
                   'Num layers': 4,
                   'Hidden funcs': ('relu', 'relu'),
                   'Hidden width': 450,
                   'Learning rate': 0.001})
"""
