import argparse
import uuid
import os
from typing import Optional
from fedn.network.clients.fedn_client import ConnectToApiResult, FednClient


class FEDnNamespace(argparse.Namespace):
    api_url: str
    api_port: int
    token: str


def parse_args() -> FEDnNamespace:
    parser = argparse.ArgumentParser(description="Edge Client")
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.environ.get("FEDN_URL"),
        required=os.environ.get("FEDN_URL") is None,
        help="The FEDn API URL. Falls back to .env's FEDN_URL",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        required=False,
        help="The FEDn API Port (not required if using full url)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("FEDN_TOKEN"),
        required=os.environ.get("FEDN_TOKEN") is None,
        help="The FEDn API Token. Falls back to .env's FEDN_TOKEN",
    )

    args = parser.parse_args()
    return args


def get_api_url(api_url: str, api_port: int):
    url = f"{api_url}:{api_port}" if api_port else api_url
    if not url.endswith("/"):
        url += "/"

    return url


def on_train(in_model):
    training_metadata = {
        "num_examples": 1,
        "batch_size": 1,
        "epochs": 1,
        "lr": 1,
    }

    metadata = {"training_metadata": training_metadata}

    # Do your training here, out_model is your result...
    out_model = in_model

    return out_model, metadata


def on_validate(in_model):
    # Calculate metrics here...
    metrics = {
        "test_accuracy": 0.9,
        "test_loss": 0.1,
        "train_loss": 0.2,
        "implemented": False,
    }
    return metrics


def on_predict(in_model):
    # Do your prediction here...
    raise NotImplementedError("on_predict not supported.")
    prediction = {
        "prediction": 1,
        "confidence": 0.9,
        "implemented": False,
    }
    return prediction


def connnect_fedn_client(
    client_name: str,
    api_url: str,
    api_port: Optional[int] = None,
    token: Optional[str] = None,
):
    fedn_client = FednClient(
        train_callback=on_train,
        validate_callback=on_validate,
        predict_callback=on_predict,
    )

    url = get_api_url(api_url, api_port)

    fedn_client.set_name(client_name)

    client_id = str(uuid.uuid4())
    fedn_client.set_client_id(client_id)

    controller_config = {
        "name": client_name,
        "client_id": client_id,
        "package": "local",
        "preferred_combiner": "",
    }

    result, combiner_config = fedn_client.connect_to_api(url, token, controller_config)

    if result != ConnectToApiResult.Assigned:
        print("Failed to connect to API, exiting.")
        return

    result: bool = fedn_client.init_grpchandler(
        config=combiner_config, client_name=client_id, token=token
    )

    if not result:
        return

    fedn_client.run()
