# =============================================================
# WHAT THIS FILE DOES:
# This is the front door. You run this file. It calls the other
# files in the right order.
#
# HOW THE FILES CONNECT, IN THE ORDER A REQUEST FLOWS:
#   main.py
#     -> ingest.py        download the public CSV
#     -> train.py         compare 3 models, save the winner
#          -> features.py prepare columns
#          -> evaluate.py score them
#     -> predict.py       score one customer on your laptop
#     -> azure_ml.py      register the winner and deploy it
#          -> scoring/score.py   the function Azure calls
#
# WHY THIS FILE EXISTS:
# You could run each file yourself. A single front door means you
# do not have to remember the order.
# =============================================================

import argparse
import json

from config import PROJECT_ROOT


# --------------------------------------------------------------
# FUNCTION: command_download
# WHAT IT DOES: Downloads the dataset and prints how many rows arrived
# WHY WE NEED IT: A quick check that the internet fetch worked before
#                 you spend time training
# INPUTS: None
# OUTPUTS: None. It prints a short summary.
# --------------------------------------------------------------
def command_download():
    from ingest import ingest

    table = ingest(force=True)
    yes_count = int((table["y"].astype(str).str.lower() == "yes").sum())
    print(f"Rows: {len(table)}")
    print(f"Subscribed (yes): {yes_count}")
    print(f"Did not subscribe (no): {len(table) - yes_count}")
    print(f"Saved under: {PROJECT_ROOT / 'data' / 'raw'}")


# --------------------------------------------------------------
# FUNCTION: command_train
# WHAT IT DOES: Trains the three models and writes the results table
# WHY WE NEED IT: This is the experiment
# INPUTS: use_azure (True logs the runs into Azure ML, not just a local folder)
# OUTPUTS: The summary dictionary from train.py
# --------------------------------------------------------------
def command_train(use_azure, register):
    from config import REGISTERED_MODEL_NAME
    from ingest import ingest
    from train import train_and_select

    tracking_uri = None
    if use_azure:
        # Imported only in this branch so a local train does not need
        # the Azure packages. An Azure ML job should use --register
        # instead: the platform already set the tracking address.
        from azure_ml import get_client
        from azure_ml import tracking_uri as azure_tracking_uri

        tracking_uri = azure_tracking_uri(get_client())

    summary = train_and_select(
        ingest(),
        tracking_uri=tracking_uri,
        register_name=REGISTERED_MODEL_NAME if register or use_azure else None,
    )
    print(f"Winner: {summary['winner']}")
    print(summary["results_markdown"])
    print(f"Saved model: {summary['model_path']}")
    print(f"Saved table: {summary['results_path']}")
    return summary


# --------------------------------------------------------------
# FUNCTION: command_predict
# WHAT IT DOES: Scores the sample customer using the local model file
# WHY WE NEED IT: You can see a prediction without paying for Azure
# INPUTS: None. It reads sample_request.json
# OUTPUTS: None. It prints JSON.
# --------------------------------------------------------------
def command_predict():
    from predict import load_model_card, load_pipeline, score_records

    request_path = PROJECT_ROOT / "sample_request.json"
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    card = load_model_card()
    predictions = score_records(load_pipeline(), payload["inputs"])
    print(json.dumps({"model_name": card.get("winner"), "predictions": predictions}, indent=2))


# --------------------------------------------------------------
# FUNCTION: command_deploy
# WHAT IT DOES: Registers the saved model and deploys the endpoint
# WHY WE NEED IT: Turns the local file into a URL
# INPUTS: None
# OUTPUTS: None. It writes reports/endpoint.json
# --------------------------------------------------------------
def command_deploy():
    from azure_ml import deploy, get_client, register_model

    client = get_client()
    registered = register_model(client, PROJECT_ROOT / "models")
    details = deploy(client, registered)
    print(json.dumps(details, indent=2))


# --------------------------------------------------------------
# FUNCTION: command_invoke
# WHAT IT DOES: Sends the sample request to the live Azure endpoint
# WHY WE NEED IT: Proves the URL answers, not just that it exists
# INPUTS: None
# OUTPUTS: None. It prints the response.
# --------------------------------------------------------------
def command_invoke():
    from azure_ml import get_client, invoke

    print(invoke(get_client()))


# --------------------------------------------------------------
# FUNCTION: command_teardown
# WHAT IT DOES: Deletes the endpoint so the virtual machine stops billing
# WHY WE NEED IT: A managed online endpoint does not scale to zero
# INPUTS: None
# OUTPUTS: None
# --------------------------------------------------------------
def command_teardown():
    from azure_ml import get_client, teardown_endpoint

    teardown_endpoint(get_client())


# --------------------------------------------------------------
# FUNCTION: build_parser
# WHAT IT DOES: Defines the commands this program accepts
# WHY WE NEED IT: argparse turns `python main.py train --azure`
#                 into a Python value we can branch on
# INPUTS: None
# OUTPUTS: An ArgumentParser
#
# LEARNING NOTE: A chain of input() questions would also work.
# argparse is the usual tool because you can rerun the exact command
# later, and GitHub Actions can call it without a human at the keyboard.
# --------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and deploy the bank term-deposit model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Download the UCI Bank Marketing CSV")

    train_parser = subparsers.add_parser("train", help="Train 3 models and pick a winner")
    train_parser.add_argument(
        "--azure",
        action="store_true",
        help="Log the MLflow runs to the Azure ML workspace from a laptop",
    )
    train_parser.add_argument(
        "--register",
        action="store_true",
        help="Register the winner in whatever MLflow server is already configured",
    )

    subparsers.add_parser("predict", help="Score sample_request.json with the local model")
    subparsers.add_parser("deploy", help="Register the model and deploy an online endpoint")
    subparsers.add_parser("invoke", help="Send the sample request to the live endpoint")
    subparsers.add_parser("teardown", help="Delete the endpoint to stop the hourly charge")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # A dictionary of name -> function is an alternative to a long if/elif.
    # The if/elif below is easier to read while you are learning.
    if args.command == "download":
        command_download()
    elif args.command == "train":
        command_train(use_azure=args.azure, register=args.register)
    elif args.command == "predict":
        command_predict()
    elif args.command == "deploy":
        command_deploy()
    elif args.command == "invoke":
        command_invoke()
    elif args.command == "teardown":
        command_teardown()
    else:
        parser.error(f"Unknown command: {args.command}")


# This block runs only when you execute `python main.py`.
# Importing main from a test does not immediately train or deploy.
if __name__ == "__main__":
    main()
