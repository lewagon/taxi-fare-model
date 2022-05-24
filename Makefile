
default: pytest

pytest:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes

test_train_at_scale:
	TEST_ENV=development PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes

dev_test:
	@make write_results
	TEST_ENV=development PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes

# °º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸

# conf

TRAINING_PREFIX=train_${DATASET_SIZE}
TRAINING_PROCESSED_PREFIX=train_processed_${DATASET_SIZE}
VALIDATION_PREFIX=val_${DATASET_SIZE}

DATA_DIR=data
RAW_DIR=raw
TMP_DIR=tmp

fbold=$(shell echo "\033[1m")
fnormal=$(shell echo "\033[0m")

ccgreen=$(shell echo "\033[0;32m")
ccblue=$(shell echo "\033[0;34m")
ccreset=$(shell echo "\033[0;39m")

run_model:
	python -m taxifare_model.interface.main

show_env:
	@echo "\nEnvironment variables used by the \`taxifare-model\` package loaded by \`direnv\` from your \`.env\` located at:"
	@echo ${DIRENV_DIR}

	@echo "\n$(ccgreen)local storage:$(ccreset)"
	@env | grep -E "LOCAL_DATA_PATH|LOCAL_REGISTRY_PATH" || :
	@echo "\n$(ccgreen)dataset:$(ccreset)"
	@env | grep -E "DATASET_SIZE|VALIDATION_DATASET_SIZE|CHUNK_SIZE" || :
	@echo "\n$(ccgreen)package behavior:$(ccreset)"
	@env | grep -E "DATA_SOURCE|MODEL_TARGET" || :

	@echo "\n$(ccgreen)GCP:$(ccreset)"
	@env | grep -E "PROJECT|REGION" || :
	@echo "\n$(ccgreen)Cloud Storage:$(ccreset)"
	@env | grep -E "BUCKET_NAME|BLOB_LOCATION" || :

	@echo "\n$(ccgreen)Big Query:$(ccreset)"
	@env | grep -E "DATASET" | grep -Ev "DATASET_SIZE|VALIDATION_DATASET_SIZE" || :

	@echo "\n$(ccgreen)Compute Engine:$(ccreset)"
	@env | grep -E "INSTANCE" || :

	@echo "\n$(ccgreen)MLflow:$(ccreset)"
	@env | grep -E "MLFLOW_EXPERIMENT|MLFLOW_MODEL_NAME" || :
	@env | grep -E "MLFLOW_TRACKING_URI|MLFLOW_TRACKING_DB" || :

	@echo "\n$(ccgreen)Prefect:$(ccreset)"
	@env | grep -E "PREFECT_BACKEND|PREFECT_FLOW_NAME|PREFECT_LOG_LEVEL" || :

reinstall_package:
	@pip uninstall -y taxifare-model || :
	@pip install -e .

write_results:

	@echo "verify installed packages versions"
	@pip show taxifare-model | grep 'Summary' > tests/all/test_package_version.txt || :

list:
	@echo "\nHelp for the \`taxifare-model\` package \`Makefile\`"

	@echo "\n$(ccgreen)$(fbold)PACKAGE$(ccreset)"

	@echo "\n    $(ccgreen)$(fbold)environment rules:$(ccreset)"
	@echo "\n        $(fbold)show_env$(ccreset)"
	@echo "            Show the environment variables used by the package by category."

	@echo "\n    $(ccgreen)$(fbold)run rules:$(ccreset)"
	@echo "\n        $(fbold)run_model$(ccreset)"
	@echo "            Run the package (\`taxifare_model.interface.main\` module)."

	@echo "\n        $(fbold)run_flow$(ccreset)"
	@echo "            Start a prefect workflow locally (run the \`taxifare_flow.main\` module)."

	@echo "\n$(ccgreen)$(fbold)WORKFLOW$(ccreset)"

	@echo "\n    $(ccgreen)$(fbold)data operation rules:$(ccreset)"
	@echo "\n        $(fbold)show_data_sources$(ccreset)"
	@echo "            Show the local data sources."
	@echo "\n        $(fbold)show_bq_tables$(ccreset)"
	@echo "            Show the Big Query dataset tables used by the package."
	@echo "\n        $(fbold)reset_data_sources$(ccreset)"
	@echo "            Reset the content of the local CSV files."
	@echo "\n        $(fbold)reset_bq_tables$(ccreset)"
	@echo "            Reset the content of the Big Query dataset tables used by the package."
	@echo "\n        $(fbold)get_new_month$(ccreset)"
	@echo "            Get one more month in the local dataset to simulate the passing of time."
	@echo "\n        $(fbold)push_month_to_bq$(ccreset)"
	@echo "            Get one more month in the Big Query dataset to simulate the passing of time."

	@echo "\n$(ccgreen)$(fbold)TESTS$(ccreset)"

	@echo "\n    $(ccgreen)$(fbold)student rules:$(ccreset)"
	@echo "\n        $(fbold)reinstall_package$(ccreset)"
	@echo "            Install the version of the package corresponding to the challenge."
	@echo "\n        $(fbold)dev_test$(ccreset)"
	@echo "            Run the tests."

	@echo "\n    $(ccblue)$(fbold)internal rules:$(ccreset)"
	@echo "\n        $(fbold)write_results$(ccreset)"
	@echo "            Write the test results so they can be added and committed to git."
	@echo "\n        $(fbold)pylint$(ccreset)"
	@echo "            Print a report on code style."
	@echo "\n        $(fbold)pytest$(ccreset)"
	@echo "            Run the tests and print a test report."
