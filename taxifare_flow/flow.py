
from taxifare_model.ml_logic.registry_db import get_latest_trained_row

from taxifare_model.interface.main import (preprocess_and_train,
                                           preprocess,
                                           train,
                                           evaluate)

from prefect import task, Flow, Parameter

import os

from colorama import Fore, Style


@task
def get_next_training_params(experiment):
    """
    retrieve the parameters for the next training

    the `taxifare_model` package saves after each training the run parameters
    in mlflow: `row_index` and `row_count` identifying the range of data used

    for the next training we want to determine the `first_row`
    containing new data which the model has not seen yet
    """

    print(Fore.GREEN + "\n# 🐙 Prefect task - get next training params:" + Style.RESET_ALL)

    next_row = get_latest_trained_row(experiment)

    print()

    return next_row


@task
def eval_perf(next_row):
    """
    evaluate the performance of the latest model in production on new data
    """

    print(Fore.GREEN + "\n# 🐙 Prefect task - eval past model perf:" + Style.RESET_ALL
          + f"\n- first row: {next_row}")

    # evaluate latest production model on new data
    past_perf = evaluate(next_row)

    print()

    return past_perf


@task
def train_model(next_row):
    """
    retrain the latest model in production with the new data
    the new data and model will be ignored if the performance is not improved
    """

    print(Fore.GREEN + "\n# 🐙 Prefect task - retrain production model:" + Style.RESET_ALL
          + f"\n- first row: {next_row}")

    # ⚠️ here we should decide whether to train the new data
    # chunk by chunk or in one piece depending on whether it fits in memory
    #
    # for the sake of simplicity, for this challenge, we will train
    # the first dataset chunk by chunk and the next ones in one piece
    #
    # if you wanted to decide based on data size, an option would be to:
    # - big query: query the new row count in order to assess the data size
    # - csv: approximate the new row count from the average line length

    # train new model (or existing production model if it exists) with new data
    if next_row == 0:

        # preprocess data chunk by chunk
        preprocess(first_row=next_row)

        # train model chunk by chunk
        new_perf = train(first_row=next_row,
                         stage="Production")

    else:

        # preprocess and train in one piece
        new_perf = preprocess_and_train(first_row=next_row,
                                        stage="Production")

    print()

    return new_perf


@task
def notify(past_perf, new_perf):
    """
    send a mail if the `mae` fluctuates above +- $.2
    """

    print(Fore.GREEN + "\n# 🐙 Prefect task - notify:" + Style.RESET_ALL
          + f"\n- past perf: {round(past_perf, 2) if past_perf is not None else 'None'}"
          + f"\n- new perf: {round(new_perf, 2) if new_perf is not None else 'None'}")

    # notify of performance evolution
    # TODO: trigger slack or mail task

    print()

    return "done"


def build_flow():
    """
    build the prefect workflow for the `taxifare-model` package
    """

    flow_name = os.environ.get("PREFECT_FLOW_NAME")

    with Flow(flow_name) as flow:

        # retrieve mlfow env params
        mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")

        # create workflow parameters
        experiment = Parameter(name="experiment", default=mlflow_experiment)

        # register tasks in the workflow
        next_row = get_next_training_params(experiment)
        past_perf = eval_perf(next_row)
        new_perf = train_model(next_row)
        notify(past_perf, new_perf)

    return flow
