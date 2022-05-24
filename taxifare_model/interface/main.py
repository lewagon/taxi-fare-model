
from taxifare_model.ml_logic.data import (clean_data,
                                          get_chunk,
                                          save_chunk)

from taxifare_model.ml_logic.params import (CHUNK_SIZE,
                                            DATASET_SIZE,
                                            VALIDATION_DATASET_SIZE)

from taxifare_model.ml_logic.preprocessor import preprocess_features

from taxifare_model.ml_logic.model import (initialize_model,
                                           compile_model,
                                           train_model,
                                           evaluate_model)

from taxifare_model.ml_logic.registry import (save_model,
                                              load_model)

import numpy as np
import pandas as pd

from colorama import Fore, Style


def preprocess_and_train(
    first_row=0
    , stage="None"
):
    """
    Load data in memory, clean and preprocess it, train a Keras model on it,
    save the model, and finally compute & save a performance metric
    on a validation set holdout at the `model.fit()` level
    """

    print("\n⭐️ use case: preprocess and train")

    # retrieve the dataset
    data = get_chunk(source_name=f"train_{DATASET_SIZE}",
                     index=first_row,
                     chunk_size=None)  # retrieve all further data

    if data is None:
        print("\n✅ no data to preprocess and train")
        return None

    row_count = len(data)

    # clean the dataset
    data_cleaned = clean_data(data)

    cleaned_row_count = len(data_cleaned)

    if cleaned_row_count == 0:
        print("\n✅ no data to preprocess and train after after data cleaning")
        return None

    # create X, y as pandas DataFrames
    X = data_cleaned.drop("fare_amount", axis=1)
    y = data_cleaned[["fare_amount"]]

    # preprocess X and return a numpy array
    X_processed = preprocess_features(X)

    # model params
    learning_rate = 0.001
    batch_size = 256

    load_existing_model = False
    if first_row != 0:
        load_existing_model = True

    model = None
    if load_existing_model:
        model = load_model(
            stage=stage
        )

    # initialize model
    if model is None:
        model = initialize_model(X_processed)
        model = compile_model(model, learning_rate)

    # train model
    model, history = train_model(model, X_processed, y, batch_size, validation_split=0.3)

    # compute val_metrics
    val_mae = np.min(history.history['val_mae'])
    metrics = dict(val_mae=val_mae)

    # save model
    params = dict(
        # hyper parameters
        learning_rate=learning_rate,
        batch_size=batch_size,
        # package behavior
        context="preprocess and train",
        # data source
        first_row=first_row,
        row_count=row_count,
        cleaned_row_count=cleaned_row_count)

    save_model(model=model, params=params, metrics=metrics)

    print(f"\n✅ trained on {row_count} rows ({cleaned_row_count} cleaned) with mae {round(val_mae, 2)}")

    return val_mae


def preprocess(
    first_row=0
):
    """
    Preprocess the dataset by chunks fitting in memory.
    """

    print("\n⭐️ use case: preprocess")

    # iterate on the dataset, by chunks
    chunk_id = 0
    row_count = 0
    cleaned_row_count = 0

    while (True):

        print(Fore.BLUE + f"\nProcessing chunk n°{chunk_id}..." + Style.RESET_ALL)

        data_chunk = get_chunk(source_name=f"train_{DATASET_SIZE}",
                               index=(chunk_id * CHUNK_SIZE) + first_row,
                               chunk_size=CHUNK_SIZE)

        # Break out of while loop if data is none
        if data_chunk is None:
            print(Fore.BLUE + "\nNo data in latest chunk..." + Style.RESET_ALL)
            break

        row_count += data_chunk.shape[0]

        data_chunk_cleaned = clean_data(data_chunk)

        cleaned_row_count += len(data_chunk_cleaned)

        # break out of while loop if cleaning removed all rows
        if len(data_chunk_cleaned) == 0:
            print(Fore.BLUE + "\nNo cleaned data in latest chunk..." + Style.RESET_ALL)
            break

        X_chunk = data_chunk_cleaned.drop("fare_amount", axis=1)
        y_chunk = data_chunk_cleaned[["fare_amount"]]

        X_processed_chunk = preprocess_features(X_chunk)

        data_processed_chunk = pd.DataFrame(
            np.concatenate((X_processed_chunk, y_chunk), axis=1))

        # save and append the chunk
        is_first = chunk_id == 0 and first_row == 0

        save_chunk(source_name=f"train_processed_{DATASET_SIZE}",
                   is_first=is_first,
                   data=data_processed_chunk)

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for the preprocessing 👌")
        return

    # save params
    params = dict(
        # package behavior
        context="preprocess",
        chunk_size=CHUNK_SIZE,
        # data source
        first_row=first_row,
        row_count=row_count,
        cleaned_row_count=cleaned_row_count)

    save_model(params=params)

    print(f"\n✅ data processed saved entirely: {row_count} rows ({cleaned_row_count} cleaned)")


def train(
    first_row=0
    , stage="None"
):
    """
    Train a new model on the full (already preprocessed) dataset ITERATIVELY, by loading it
    chunk-by-chunk, and updating the weight of the model after each chunks.
    Save final model once it has seen all data, and compute validation metrics on a holdout validation set
    common to all chunks.
    """

    print("\n⭐️ use case: train")

    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # load a validation set common to all chunks, used to early stop model training
    data_val = get_chunk(source_name=f"val_{VALIDATION_DATASET_SIZE}",
                         index=0,  # retrieve from first row
                         chunk_size=None)  # retrieve all further data

    if data_val is None:
        print("\n✅ no data to train")
        return None

    X_val = data_val.drop("fare_amount", axis=1)
    y_val = data_val[["fare_amount"]]

    X_val_processed = preprocess_features(X_val)

    load_existing_model = False
    if first_row != 0:
        load_existing_model = True

    # model params
    learning_rate = 0.001
    batch_size = 256

    # iterate on the full dataset per chunks
    model = None
    chunk_id = 0
    row_count = 0
    metrics_val_list = []

    while (True):

        print(Fore.BLUE + f"\nLoading and training on preprocessed chunk n°{chunk_id}..." + Style.RESET_ALL)

        data_processed_chunk = get_chunk(source_name=f"train_processed_{DATASET_SIZE}",
                                         index=(chunk_id * CHUNK_SIZE) + first_row,
                                         chunk_size=CHUNK_SIZE)

        # check whether data source contain more data
        if data_processed_chunk is None:
            print(Fore.BLUE + "\nNo more chunk data..." + Style.RESET_ALL)
            break

        data_processed_chunk = data_processed_chunk.to_numpy()

        X_train_chunk = data_processed_chunk[:, :-1]
        y_train_chunk = data_processed_chunk[:, -1]

        # increment trained row count
        chunk_row_count = data_processed_chunk.shape[0]
        row_count += chunk_row_count

        if model is None:
            if load_existing_model:
                model = load_model(
                    stage=stage
                )

        # initialize model
        if model is None:
            model = initialize_model(X_train_chunk)
            model = compile_model(model, learning_rate)

        # train the model incrementally
        model, history = train_model(model,
                                     X_train_chunk,
                                     y_train_chunk,
                                     batch_size,
                                     validation_data=(X_val_processed, y_val))

        metrics_val_chunk = np.min(history.history['val_mae'])
        metrics_val_list.append(metrics_val_chunk)
        print(metrics_val_chunk)

        # check if chunk was full
        if chunk_row_count < CHUNK_SIZE:
            print(Fore.BLUE + "\nNo more chunks..." + Style.RESET_ALL)
            break

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for the training 👌")
        return

    mean_val_mae = np.mean(np.array(metrics_val_list))

    print(f"\n✅ trained on {row_count} rows: [{first_row}-{first_row + row_count - 1}] with mae {round(mean_val_mae, 2)}")

    params = dict(
        # hyper parameters
        learning_rate=learning_rate,
        batch_size=batch_size,
        # package behavior
        context="train",
        chunk_size=CHUNK_SIZE,
        # data source
        cleaned_first_row=first_row,
        cleaned_row_count=row_count)

    # process metrics
    metrics = dict(mean_val=mean_val_mae)

    # save model
    save_model(model=model, params=params, metrics=metrics)

    return mean_val_mae


def evaluate(first_row):
    """
    Evaluate the performance of the latest production model on new data
    """

    print("\n⭐️ use case: evaluate")

    # load new data
    new_data = get_chunk(source_name=f"train_{DATASET_SIZE}",
                         index=first_row,
                         chunk_size=None)  # retrieve all further data

    if new_data is None:
        print("\n✅ no data to evaluate")
        return None

    X_new = new_data.drop("fare_amount", axis=1)
    y_new = new_data[["fare_amount"]]

    X_new_processed = preprocess_features(X_new)

    model = load_model(
        stage="Production"
    )

    metrics_dict = evaluate_model(model=model, X=X_new_processed, y=y_new)

    # save evaluation
    params = dict(
        # package behavior
        context="evaluate",
        # data source
        first_row=first_row,
        row_count=len(X_new_processed))

    save_model(params=params, metrics=metrics_dict)

    mae = metrics_dict["mae"]

    return mae


def pred(
    X_pred: pd.DataFrame = None
    , stage="None"
) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ use case: predict")

    if X_pred is None:

        X_pred = pd.DataFrame(dict(
            key=["2013-07-06 17:18:00"],  # useless but the pipeline requires it
            pickup_datetime=["2013-07-06 17:18:00 UTC"],
            pickup_longitude=[-73.950655],
            pickup_latitude=[40.783282],
            dropoff_longitude=[-73.984365],
            dropoff_latitude=[40.769802],
            passenger_count=[1]))

    model = load_model(
        stage=stage
    )

    X_processed = preprocess_features(X_pred)

    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape)

    return y_pred


if __name__ == '__main__':
    try:
        # preprocess_and_train()
        # preprocess()
        # train()
        # evaluate(first_row=9000)
        pred()
    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
