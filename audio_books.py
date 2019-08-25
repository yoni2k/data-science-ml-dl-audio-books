from timeit import default_timer as timer
import pandas as pd
import numpy as np
import itertools
import time
from pprint import pprint
import functools
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
- Update requirements.txt
- Remove all references to "lecture" and duplicates of 2 ways to do something
- Decide if putting seeds all over, or removing altogether 
- Put all inputs and targets into 1 struct
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
    validate_loss_improve_deltas = [0.0001, 0.00001]

    # validate_loss_improve_patiences = [7, 10, 15]
    validate_loss_improve_patiences = [10, 15]

    improve_restore_best_weights_values = [True]

    batch_sizes = [200, 400, 600]

    hidden_widths = [200, 450, 784]

    nums_layers = [4, 5]

    functions = ['sigmoid', 'tanh', 'relu', 'softmax']
    functions = ['sigmoid', 'tanh', 'relu']

    #learning_rates = [0.001, 0.0005, 0.00001]  # default in tf.keras.optimizers.Adam is 0.001
    learning_rates = [0.001, 0.0005]  # default in tf.keras.optimizers.Adam is 0.001

def acquire_preprocess_data():
    np.set_printoptions(formatter={'float': lambda x: "{0: 0.2f}".format(x)}, linewidth=120)

    raw_data = np.loadtxt(INPUT_FILE, delimiter=',')
    data = raw_data[:, 1:]
    print(f'===================== beginning of data no index:\n{data[:5, :]}')

    return data


def shuffle_data(data):
    shuffled_indices = np.arange(data.shape[0])
    np.random.shuffle(shuffled_indices)
    data = data[shuffled_indices]
    print(f'===================== beginning of shuffled data:\n{data[:5, :]}')
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
    print(f'num_one_targets: {num_one_targets}, data shape: {data.shape}')
    print(f'===================== beginning of balanced data:\n{data[:5, :]}')

    # shuffle again because all the end are 1's
    data = shuffle_data(data)

    # scale the inputs
    preprocessing.scale(data[:, :-1], copy=False)
    print(f'shapes after scaling: {data.shape}')
    print(f'===================== beginning of scaled data:\n{data[:5, :]}')

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

    print(f'shapes train: train_inputs: {train_inputs.shape}, train_targets: {train_targets.shape}, '
          f'num 1s fraction: {round(int(sum(train_targets)) / train_targets.shape[0], 2)}')
    print(f'shapes validation: '
          f'validation_inputs: {validation_inputs.shape}, validation_targets: {validation_targets.shape}, '
          f'num 1s fraction: {round(int(sum(validation_targets)) / validation_targets.shape[0], 2)}')
    print(f'shapes test: test_inputs: {test_inputs.shape}, test_targets: {test_targets.shape}, '
          f'num 1s fraction: {round(int(sum(test_targets)) / test_targets.shape[0], 2)}')

    # write to files
    np.savez(OUTPUT_TRAIN_FILE, inputs=train_inputs, targets=train_targets)
    np.savez(OUTPUT_VALID_FILE, inputs=validation_inputs, targets=validation_targets)
    np.savez(OUTPUT_TEST_FILE, inputs=test_inputs, targets=test_targets)


def read_npz_file(file_name):
    npz = np.load(file_name)
    inputs = npz['inputs'].astype(np.float)
    targets = npz['targets'].astype(np.int)
    print(f'{file_name}: inputs.shape: {inputs.shape}, targets: {targets.shape}')
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

    print(f'train_inputs.shape: {train_inputs.shape}')
    print(f'train_targets.shape: {train_targets.shape}')
    print(f'batch_size: {in_dic["Batch size"]}')
    print(f'max_epochs: {in_dic["Max num epochs"]}')
    print(f'valid_inputs.shape: {valid_inputs.shape}')
    print(f'valid_targest.shape: {valid_targets.shape}')


    print(f'train_inputs.shape: {train_inputs.shape}, train_targets.shape: {train_targets.shape}, '
          f'batch_size: {in_dic["Batch size"]}, max_epochs: {in_dic["Max num epochs"]}, '
          f'valid_inputs.shape: {valid_inputs.shape}, valid_targest.shape: {valid_targets.shape}')

    start = timer()
    history = model.fit(train_inputs,
                        train_targets,
                        batch_size=in_dic['Batch size'],
                        epochs=in_dic['Max num epochs'],
                        callbacks=[early_callback],
                        validation_data=(valid_inputs, valid_targets),
                        verbose=2)
    end = timer()

    # Test the model
    test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)

    actual_num_epochs = len(history.history["val_accuracy"])

    # Prepare output of results - see explanation of outputs in README.md
    out_dic['Test Accuracy'] = round(test_accuracy, 4)
    out_dic['Test Loss'] = round(test_loss, 4)
    out_dic['Train Time'] = round(end - start, 3)
    out_dic['Loss * Time'] = round(out_dic['Test Loss'] * (end - start), 4)
    out_dic['Validate Accuracy'] = history.history["val_accuracy"][-1].round(4)
    out_dic['Train Accuracy'] = history.history["accuracy"][-1].round(4)
    out_dic['Validate Loss'] = history.history["val_loss"][-1].round(4)
    out_dic['Train Loss'] = history.history["loss"][-1].round(4)
    out_dic['Num epochs'] = actual_num_epochs
    out_dic['Average epoch time'] = round((end - start) / actual_num_epochs, 4)

    return out_dic


def do_numerous_loops(num_loops=1, given_dic=None):
    """ Performs numerous models with different inputs / hyperparameters.
        Outputs to output files - see README.md for explanation """

    results = []
    # Inputs are given in 2 ways:
    #  1. Mostly for local testing - explicit by giving given_dic, and only 1 run is done (possibly with numerous loops)
    #  2. By not giving given_dic, and then numerous inputs / hyperparameters are taken from the settings
    if given_dic:
        in_dic = given_dic
        local_validate_loss_improve_deltas = [in_dic['Validate loss improvement delta']]
        local_validate_loss_improve_patiences = [in_dic['Validate loss improvement patience']]
        local_improve_restore_best_weights_values = [in_dic['Restore best weights']]
        local_batch_sizes = [in_dic['Batch size']]
        local_hidden_widths = [in_dic['Hidden width']]
        local_nums_layers = [in_dic['Num layers']]
        local_functions = [in_dic['Hidden funcs']]
        local_learning_rates = [in_dic['Learning rate']]
    else:
        # MAX_NUM_EPOCHS is constant - no loops on different values
        in_dic = {'Max num epochs': MAX_NUM_EPOCHS,
                  'Shuffle seed': 100}
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
    num_regressions_without_functions = num_loops * \
                                        len(local_validate_loss_improve_deltas) * \
                                        len(local_validate_loss_improve_patiences) * \
                                        len(local_improve_restore_best_weights_values) * \
                                        len(local_batch_sizes) * \
                                        len(local_hidden_widths) * \
                                        len(local_nums_layers) * \
                                        len(local_learning_rates)
    print(f'To perform number regressions: {num_regressions_without_functions} '
          f'with layers: {local_nums_layers} and functions: {local_functions}')

    # TODO 1 - search for mnist - remove
    # acquiring data is done once to save time
    data = acquire_preprocess_data()

    # TODO 2 - debug and resolve best accuracies - seems there is a bug there with values, what gets saved etc.
    # initiate best values that will be overridden when finding a good result
    best_test_accuracy = {'Test Accuracy': 0.001}
    best_test_loss = {'Test Loss': 100000}
    best_loss_efficiency = {'Loss * Time': 100000}

    num_model_trainings = 0
    time_run_started = timer()

    for batch_size in local_batch_sizes:
        in_dic['Batch size'] = batch_size
        if num_loops == 1:
            # to save time not to do it every time if there is only 1 loop
            in_dic['Shuffle seed'] = 1
            # Shuffling is affected by 2 hyperparameters: batch_size (therefore needs to be done per batch size), and
            #   Shuffle seed that changes per loop. Therefore needs to be done later in the loops pass,
            #   however, if there is only 1 loop, shuffling can be done once here.
            train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets = \
                prepare_data(data)

        for validate_loss_improve_delta in local_validate_loss_improve_deltas:
            in_dic['Validate loss improvement delta'] = validate_loss_improve_delta
            for validate_loss_improve_patience in local_validate_loss_improve_patiences:
                in_dic['Validate loss improvement patience'] = validate_loss_improve_patience
                for improve_restore_best_weights in local_improve_restore_best_weights_values:
                    in_dic['Restore best weights'] = improve_restore_best_weights
                    for num_layers in local_nums_layers:
                        in_dic['Num layers'] = num_layers
                        # if given specific run, don't do a product of all activation functions, otherwise perform
                        #   a product of all activation functions to put each one of the functions in each hidden layer
                        if not given_dic:
                            funcs_product = itertools.product(local_functions, repeat=(num_layers - 2))
                        else:
                            funcs_product = [in_dic['Hidden funcs']]
                        print(f'funcs_product: {funcs_product}')
                        for hidden_funcs in funcs_product:
                            in_dic['Hidden funcs'] = hidden_funcs
                            for hidden_width in local_hidden_widths:
                                in_dic['Hidden width'] = hidden_width
                                for learning_rate in local_learning_rates:
                                    in_dic['Learning rate'] = learning_rate
                                    num_model_trainings += 1
                                    # Accumulate values are used for multiple loops with same hyperparameters
                                    #  (besides seed that's per loop)
                                    accum_test_accuracy = 0
                                    accum_test_loss = 0
                                    accum_loss_efficiency = 0

                                    #initiate values to be used initiated inside the loops and used outside
                                    # to quite a warning
                                    result = {}

                                    for loop in range(1, num_loops+1):
                                        if num_loops > 1:
                                            # if more than 1 loops, to allow for different seed, prepare data per loop
                                            in_dic['Shuffle seed'] = loop
                                            train_inputs, train_targets, \
                                                valid_inputs, valid_targets, \
                                                test_inputs, test_targets = \
                                                prepare_data(data)

                                        # Printing progress of epochs, models, time and loop
                                        time_running_sec = timer() - time_run_started
                                        print(f'Model {num_model_trainings}, loop {loop}/{num_loops}, '
                                              f'total time min: {round(time_running_sec / 60, 1)}, '
                                              f'total time hours: {round(time_running_sec / 60 / 60, 2)}: '
                                              f'seconds per model: {round(time_running_sec / num_model_trainings)} '
                                              f'====================================')
                                        out_dic = single_model(train_inputs, train_targets,
                                                               valid_inputs, valid_targets,
                                                               test_inputs, test_targets,
                                                               in_dic)
                                        # result for purposes of output to file contains both inputs and outputs
                                        result = in_dic.copy()
                                        result.update(out_dic)
                                        results.append(result)
                                        print(f'\nCURRENT: {result}')

                                        # calculate total for all loops of same model
                                        accum_test_accuracy += result['Test Accuracy']
                                        accum_test_loss += result['Test Loss']
                                        accum_loss_efficiency += result['Loss * Time']

                                    # After finishing all loops for a given model,
                                    #  update and print values of averages for loops - relevant only to
                                    #  printing and outputting to file of best values.  Full contains all specific
                                    #  runs of the loop
                                    if (accum_test_accuracy / num_loops) > best_test_accuracy['Test Accuracy']:
                                        best_test_accuracy = \
                                            {'Average loop result': round(accum_test_accuracy / num_loops, 4)}
                                        best_test_accuracy.update(result)
                                    if (accum_test_loss / num_loops) < best_test_loss['Test Loss']:
                                        best_test_loss = \
                                            {'Average loop result': round(accum_test_loss / num_loops, 4)}
                                        best_test_loss.update(result)
                                    if (accum_loss_efficiency / num_loops) < best_loss_efficiency['Loss * Time']:
                                        best_loss_efficiency = \
                                            {'Average loop result': round(accum_loss_efficiency / num_loops, 4)}
                                        best_loss_efficiency.update(result)

                                    print("Finished all loops +++++++++++++++++++++++++++++++++++++++++++++")
                                    print(f'BEST TEST ACCURACY:     {best_test_accuracy}')
                                    print(f'BEST TEST LOSS:         {best_test_loss}')
                                    print(f'BEST LOSS EFFICIENCY:   {best_loss_efficiency}')

    # Output all results to full.xlsx
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'Total number of models trained: {num_model_trainings} with {num_loops} loops per model')
    pf = pd.DataFrame(results)
    print(f'ALL RESULTS:')
    print(pf.to_string())
    writer = pd.ExcelWriter(f'output/results_{time.strftime("%Y_%m_%d_%H_%M_%S")}.xlsx')
    pf.to_excel(writer, sheet_name='Full')

    time_running_sec = timer() - time_run_started

    # Output all inputs / hyperparameters to hyperparams.xlsx
    hyperparams = {
        'Num Model Trainings': num_model_trainings,
        'Num Loops per model': num_loops,
        'Total time minutes': round(time_running_sec / 60, 1),
        'Total time hours': round(time_running_sec / 60 / 60, 2),
        'Seconds per model': round(time_running_sec / num_model_trainings),
        'Max Num Epochs': in_dic['Max num epochs'],
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

    pf = pd.DataFrame([best_test_accuracy_with_type,
                       best_test_loss_with_type,
                       best_loss_efficiency_with_type])
    print(f'BEST RESULTS:')
    print(pf.to_string())
    pf.to_excel(writer, sheet_name='Best')

    writer.save()


""" Ways to run:
1. Regular way - numerous models for later comparison between them.
    - Update settings / hyperparameters options from beginning of file.  Each combination will be done
    - Run do_numerous_loops with given number of loops per model (usually 1 since very time consuming even with 1 loop)
        and without a dictionary
2. Debug / confirmation / researching one model way:
    - Give number of loops (often > 1 to confirm behavior / results)
    - Give explicit dictionary of model to run
    Values given below are for some of the best models
"""
"""
# do_numerous_loops(1)
"""
# """
do_numerous_loops(1, {'Validate loss improvement delta': 0.0001,
                      'Validate loss improvement patience': 10,
                      'Restore best weights': True,
                      'Max num epochs': 1000,
                      'Batch size': 450,  # 200
                      'Num layers': 4,
                      'Hidden funcs': ('relu', 'relu'),
                      'Hidden width': 450,
                      'Learning rate': 0.001})
# """
