# Fledge (FL + edge)

This repo is for my master thesis research, aimed at making federated training of LLMs possible on edge devices. The proposed method incorporates Split Federated Learning, quantization and PEFT methods in order to reduce the client side burden of training and enable fine tuning of LLMs on edge devices (with a target hadware size of mobile-devices).

## Getting started:
This project uses [FEDn](https://github.com/scaleoutsystems/fedn) for client orchestration and adapter aggregation. See thier getting started to learn more about how the framework works. You can create a project for free using their service or self host it.

Additionally, this project manages dependencies and packaging with [uv](https://docs.astral.sh/uv/). See their [installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to get started.

After cloning this repo, `cd fledge` and if `uv` is installed, you can download the dependencies for development via `uv sync`. If you are missing the required python version (`3.12`), uv can install it as well via `uv python install 3.12`. 

`uv` automatically creates a `.venv` that you can source to activate (if say your language server needs an environment), or you can run commands using the `uv run ...` command.

TODO: Talk about docker
TODO: Talk about general pipeline
TODO: Main run command (eventually)

## TODO: directory structure

## Running tests
Tests are written in `/tests` and should mirror the `/fledge` source directory 1:1. Tests can be ran using:
```bash
uv run pytest
```
Or a single test file:
```bash
pytest tests/<TARGET_TEST_FILE>.py
```
Or with a specific test function in mind:
```bash
uv run pytest -k "<THE_NAME_OF_THE_TARGET_FUNCTION>"
```
