# =============================================================
# WHAT THIS FILE DOES:
# Talks to Azure Machine Learning: points MLflow at the workspace,
# registers the winning model, deploys a managed online endpoint,
# calls that endpoint once, and can delete it to stop the bill.
#
# HOW IT CONNECTS TO OTHER FILES:
# - config.py holds the resource names (group, workspace, endpoint)
# - train.py does the actual learning. This file only tells MLflow
#   where to send the notebook entries when you pass --azure.
# - scoring/score.py is copied into a small folder and uploaded.
#   Azure calls init() and run() in that copy.
#
# WHY AZURE IMPORTS ARE INSIDE FUNCTIONS:
# GitHub Actions runs the tests on a clean computer with no Azure
# login. If this file imported Azure at the top, importing it would
# fail there. Local tests do not need the cloud.
# =============================================================

import json
import os
import shutil
import subprocess
from pathlib import Path

from config import (
    AZURE_ML_LOCATION,
    AZURE_ML_WORKSPACE,
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    DEPLOYMENT_NAME,
    ENDPOINT_NAME,
    INSTANCE_COUNT,
    INSTANCE_TYPE,
    PROJECT_ROOT,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
)

# Tried in order. F2s_v2 is the cheap 2-CPU size. Some regions reject
# a size even when the docs list it, so deploy() walks this list.
INSTANCE_CANDIDATES = [
    INSTANCE_TYPE,
    "Standard_DS2_v2",
    "Standard_DS3_v2",
]

# The base container Azure builds our environment on top of.
# This is the CPU image the Azure ML docs still show in their examples.
# A newer Ubuntu tag exists, but a wrong tag fails only after a long
# image build. The conda file installs Python 3.12 on top of this image.
BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"


# --------------------------------------------------------------
# FUNCTION: azure_cli_path
# WHAT IT DOES: Finds the az command on this computer
# WHY WE NEED IT: On Windows the command is a .cmd file. Python
#                 sometimes cannot run the short name "az".
# INPUTS: None
# OUTPUTS: A full path to az or az.cmd
# --------------------------------------------------------------
def azure_cli_path():
    # On Windows, `az` is usually az.cmd, not az.exe. Python's subprocess
    # does not always resolve that short name. shutil.which does.
    found = shutil.which("az") or shutil.which("az.cmd")
    if found:
        return found

    windows_default = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    windows_default = windows_default / "Microsoft SDKs" / "Azure" / "CLI2" / "wbin" / "az.cmd"
    if windows_default.exists():
        return str(windows_default)

    raise FileNotFoundError(
        "The Azure CLI was not found. Install it and run `az login`, "
        "or set AZURE_SUBSCRIPTION_ID in a .env file."
    )


# --------------------------------------------------------------
# FUNCTION: subscription_id
# WHAT IT DOES: Finds which Azure subscription to use
# WHY WE NEED IT: The id is an account detail, not something we
#                 want to commit into a public repository
# INPUTS: None
# OUTPUTS: A subscription id string
# --------------------------------------------------------------
def subscription_id():
    if AZURE_SUBSCRIPTION_ID:
        return AZURE_SUBSCRIPTION_ID

    # An Azure ML job does not have the Azure CLI installed. The platform
    # puts the subscription id in an environment variable instead.
    from_job = os.getenv("AZUREML_ARM_SUBSCRIPTION")
    if from_job:
        return from_job

    # az is the Azure command-line tool. You already logged in with it
    # on a laptop. We ask it which subscription is active, and we do not
    # print the id into the repository.
    completed = subprocess.run(
        [azure_cli_path(), "account", "show", "--query", "id", "-o", "tsv"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


# --------------------------------------------------------------
# FUNCTION: get_client
# WHAT IT DOES: Builds the object that can create Azure ML resources
# WHY WE NEED IT: Every other function in this file needs that object
# INPUTS: None. It uses your existing `az login` session.
# OUTPUTS: An MLClient
#
# LEARNING NOTE: DefaultAzureCredential is a chain of ways to prove
# who you are (the CLI login, an environment variable, a managed
# identity in the cloud). On your laptop the CLI login is the one
# that works. The alternative is pasting a key into the code. Do not.
# --------------------------------------------------------------
def get_client():
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    return MLClient(
        credential=DefaultAzureCredential(exclude_interactive_browser_credential=True),
        subscription_id=subscription_id(),
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_ML_WORKSPACE,
    )


# --------------------------------------------------------------
# FUNCTION: tracking_uri
# WHAT IT DOES: Returns the MLflow address of this workspace
# WHY WE NEED IT: train.py logs to whatever address you give it.
#                 This function asks Azure for the right address
#                 instead of hard-coding one (those addresses contain
#                 your subscription id).
# INPUTS: client, from get_client()
# OUTPUTS: A string starting with azureml://
# --------------------------------------------------------------
def tracking_uri(client):
    workspace = client.workspaces.get(AZURE_ML_WORKSPACE)
    uri = workspace.mlflow_tracking_uri
    if not uri:
        raise RuntimeError(
            f"Workspace {AZURE_ML_WORKSPACE} did not return an MLflow tracking URI."
        )
    return uri


# --------------------------------------------------------------
# FUNCTION: stage_scoring_code
# WHAT IT DOES: Copies only the files the endpoint needs into a
#               clean folder
# WHY WE NEED IT: The project folder also contains the virtual
#                 environment, which is hundreds of megabytes.
#                 Uploading that would be slow and pointless.
# INPUTS: destination folder
# OUTPUTS: The destination path
# --------------------------------------------------------------
def stage_scoring_code(destination):
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for file_name in ["config.py", "features.py", "predict.py"]:
        shutil.copy(PROJECT_ROOT / file_name, destination / file_name)

    # Azure expects the scoring script inside the uploaded folder.
    # We copy it to the top of that folder so the path is just score.py.
    shutil.copy(PROJECT_ROOT / "scoring" / "score.py", destination / "score.py")
    return destination


# --------------------------------------------------------------
# FUNCTION: register_model
# WHAT IT DOES: Puts the saved pipeline into the Azure ML model registry
# WHY WE NEED IT: A deployment has to point at a versioned registry
#                 entry, not at a file that only exists on your laptop
# INPUTS: client, and the folder that holds pipeline.joblib
# OUTPUTS: The registered model object (name + version)
# --------------------------------------------------------------
def register_model(client, model_dir):
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model

    model_dir = Path(model_dir)
    if not (model_dir / "pipeline.joblib").exists():
        raise FileNotFoundError(
            f"{model_dir / 'pipeline.joblib'} is missing. Train before registering."
        )

    model = Model(
        name=REGISTERED_MODEL_NAME,
        path=str(model_dir),
        type=AssetTypes.CUSTOM_MODEL,
        description=(
            "Best of logistic regression, random forest, and gradient boosting "
            "on UCI Bank Marketing. Call duration is excluded (leakage)."
        ),
    )
    registered = client.models.create_or_update(model)
    print(f"Registered {registered.name} version {registered.version}")
    return registered


# --------------------------------------------------------------
# FUNCTION: _environment
# WHAT IT DOES: Describes the Python packages the endpoint container needs
# WHY WE NEED IT: The endpoint is a fresh computer. It does not have
#                 your laptop's virtual environment.
# INPUTS: None
# OUTPUTS: An Environment object (not yet built into an image)
# --------------------------------------------------------------
def _environment():
    from azure.ai.ml.entities import Environment

    return Environment(
        name="bank-term-deposit-sklearn",
        description="Scoring environment for the bank term-deposit model.",
        image=BASE_IMAGE,
        conda_file=str(PROJECT_ROOT / "azure" / "conda.yml"),
    )


# --------------------------------------------------------------
# FUNCTION: deploy
# WHAT IT DOES: Creates the endpoint and one deployment, then sends
#               100% of traffic to that deployment
# WHY WE NEED IT: A registered model just sits in storage until a
#                 deployment copies it onto a machine that can answer HTTP
# INPUTS: client, registered model
# OUTPUTS: A dictionary with the scoring URI (no secrets)
#
# LEARNING NOTE: begin_create_or_update().result() waits until Azure
# finishes. The first deployment also builds a container image, which
# often takes 10 to 20 minutes. That wait is normal.
# --------------------------------------------------------------
def deploy(client, registered_model):
    from azure.ai.ml.entities import (
        CodeConfiguration,
        ManagedOnlineDeployment,
        ManagedOnlineEndpoint,
    )

    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        auth_mode="key",
        description="Predict whether a client will subscribe to a term deposit.",
        location=AZURE_ML_LOCATION,
    )
    print(f"Creating endpoint {ENDPOINT_NAME} in {AZURE_ML_LOCATION}...")
    client.online_endpoints.begin_create_or_update(endpoint).result()

    code_dir = stage_scoring_code(PROJECT_ROOT / "azure" / "score_bundle")
    last_error = None

    for instance_type in INSTANCE_CANDIDATES:
        deployment = ManagedOnlineDeployment(
            name=DEPLOYMENT_NAME,
            endpoint_name=ENDPOINT_NAME,
            model=registered_model,
            environment=_environment(),
            code_configuration=CodeConfiguration(
                code=str(code_dir),
                scoring_script="score.py",
            ),
            instance_type=instance_type,
            instance_count=INSTANCE_COUNT,
        )
        print(f"Deploying on {instance_type}. This can take 15-25 minutes...")
        try:
            client.online_deployments.begin_create_or_update(deployment).result()
            chosen_type = instance_type
            break
        except Exception as error:
            last_error = error
            print(f"{instance_type} failed: {error}")
    else:
        raise RuntimeError(
            "Every instance type failed. The last error is above."
        ) from last_error

    # Traffic is the percentage of requests each deployment receives.
    # 100 means "send everything to blue". A second deployment named
    # green is how teams do a safe swap later. We only have one.
    live_endpoint = client.online_endpoints.get(ENDPOINT_NAME)
    live_endpoint.traffic = {DEPLOYMENT_NAME: 100}
    client.online_endpoints.begin_create_or_update(live_endpoint).result()

    details = {
        "endpoint_name": ENDPOINT_NAME,
        "deployment_name": DEPLOYMENT_NAME,
        "instance_type": chosen_type,
        "instance_count": INSTANCE_COUNT,
        "scoring_uri": live_endpoint.scoring_uri,
        "location": AZURE_ML_LOCATION,
        "resource_group": AZURE_RESOURCE_GROUP,
        "workspace": AZURE_ML_WORKSPACE,
        "model": f"{registered_model.name}:{registered_model.version}",
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "endpoint.json").write_text(
        json.dumps(details, indent=2),
        encoding="utf-8",
    )
    print(f"Scoring URI: {details['scoring_uri']}")
    return details


# --------------------------------------------------------------
# FUNCTION: invoke
# WHAT IT DOES: Sends sample_request.json to the live endpoint
# WHY WE NEED IT: A URL that exists is not the same as a URL that works
# INPUTS: client, optional path to a JSON request file
# OUTPUTS: The response text from the endpoint
# --------------------------------------------------------------
def invoke(client, request_file=None):
    path = Path(request_file) if request_file else PROJECT_ROOT / "sample_request.json"
    response = client.online_endpoints.invoke(
        endpoint_name=ENDPOINT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        request_file=str(path),
    )
    return response


# --------------------------------------------------------------
# FUNCTION: teardown_endpoint
# WHAT IT DOES: Deletes the online endpoint and its virtual machine
# WHY WE NEED IT: The machine bills every hour it exists, even if
#                 nobody sends a request. This is the off switch.
# INPUTS: client
# OUTPUTS: None
# --------------------------------------------------------------
def teardown_endpoint(client):
    print(f"Deleting endpoint {ENDPOINT_NAME}...")
    client.online_endpoints.begin_delete(name=ENDPOINT_NAME).result()
    print("Endpoint deleted. The workspace is still there (it is cheap).")
    print("To also delete the workspace and its storage, run:")
    print(f"  az group delete --name {AZURE_RESOURCE_GROUP} --yes")


if __name__ == "__main__":
    print("Use python main.py deploy, invoke, or teardown.")
