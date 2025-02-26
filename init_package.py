import os
from fedn import APIClient


# TODO steps:
# Probably read a bunch of config stuff like model_id, number of blocks etc
# Create package.tgz (or use local package?? Not sure where that is configured)
# Init model (seed.npz) via build cmd with just adapter layers
# Push package and seed model to FEDn


if __name__ == "__main__":
    client = APIClient(host=os.environ.get("FEDN_URL"))
