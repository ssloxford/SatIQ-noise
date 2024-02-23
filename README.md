# SatIQ, Jamming Analysis

This reposistory contains the data collection and analysis code used in the paper "Sticky Fingers: Resilience of Satellite Fingerprinting Systems against Jamming Attacks".
This work uses the `SatIQ` system from the paper "Watch This Space: Securing Satellite Communication through Resilient Transmitter Fingerprinting", evaluating the resilience of the system against jamming attacks - the repo for this system can be found at https://github.com/ssloxford/SatIQ.

Additional materials:
- Paper (arXiv preprint): https://arxiv.org/abs/2402.05042
- Dataset: https://zenodo.org/record/10678124
- "Watch This Space" paper (arXiv preprint): https://arxiv.org/abs/2305.06947
- SatIQ model weights: https://zenodo.org/record/8298532

When using this data, please cite the following paper: "Sticky Fingers: Resilience of Satellite Fingerprinting Systems against Jamming Attacks".
The BibTeX entry is given below:
```
@inproceedings{smailesSticky2024,
  author = {Smailes, Joshua and Salkield, Edd and K{\"o}hler, Sebastian and Birnbach, Simon and Strohmeier, Martin and Martinovic, Ivan},
  title = {{Sticky Fingers}: {Resilience of Satellite Fingerprinting against Jamming Attacks}},
  year = {2024},
  booktitle = {Workshop on the Security of Space and Satellite Systems (SpaceSec)},
  location = {San Diego, USA},
  series = {SpaceSec '24}
}
```

If using the `SatIQ` model, please cite the following paper: "Watch This Space: Securing Satellite Communication through Resilient Transmitter Fingerprinting".
The BibTeX entry is given below:
```
@inproceedings{smailesWatch2023,
  author = {Smailes, Joshua and K{\"o}hler, Sebastian and Birnbach, Simon and Strohmeier, Martin and Martinovic, Ivan},
  title = {{Watch This Space}: {Securing Satellite Communication through Resilient Transmitter Fingerprinting}},
  year = {2023},
  publisher = {Association for Computing Machinery},
  booktitle = {Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security},
  location = {Copenhagen, Denmark},
  series = {CCS '23}
}
```


## Setup

To clone the repository:
```bash
git clone --recurse-submodules https://github.com/ssloxford/SatIQ-noise.git
cd SatIQ-noise
```

A Docker container is provided for ease of use, with all dependencies installed.
A recent version of Docker must be installed on your system to use this.

To run scripts locally, the following packages are required:
```
python3
```

The following Python packages are also required:
```
numpy
matplotlib
pandas
keras
h5py
zmq
tqdm
tensorflow
tensorflow-datasets
tensorflow-addons==0.13.0
scipy
seaborn
scikit-learn
notebook
```

A GPU is recommended (with all necessary drivers installed), although not required to run the model.


### Downloading Data

The full dataset is stored on Zenodo at the following URL: https://zenodo.org/record/10678124.

These can be downloaded from the site directly, but the following script may be preferable due to the large file size:
```bash
#!/bin/bash

for i in $(seq 0 32); do
  wget https://zenodo.org/records/10678124/files/${i}.tar.gz
done
```

> [!WARNING]
> These files are large (45GB total).
> Ensure you have enough disk space before downloading.

To extract the files:
```bash
#!/bin/bash

for i in $(seq 0 32); do
  tar xzf ${i}.tar.gz
done
```

See the instructions below on processing the resulting files for use.


## Usage

### TensorFlow Container

The script `tf-container.sh` provides a Docker container with the required dependencies for analysis.
Run the script from inside the repository's root directory to ensure volumes are correctly mounted.

If your machine has no GPUs:
- Modify `Dockerfile` to use the `tensorflow/tensorflow:latest` image.
- Modify `tf-container.sh`, removing `--gpus all`.


### SatIQ

See [the main SatIQ repository](https://github.com/ssloxford/SatIQ) for documentation and examples of the `SatIQ` system.


### Data Collection

The `data-collection` directory contains a `docker-compose` pipeline to receive signals from an SDR, extract Iridium messages, and save the data to a database file, while adding noise from a different SDR.
To run under its default configuration, connect a USRP N210 (via Ethernet) and a BladeRF (via USB) to the host machine, and run the following (from inside the `data-collection` directory):

```bash
./autorun.sh
```

Data will be stored in `data/noise`.

If different SDRs are used, the `iridium_extractor` configuration may need to be altered.
Change the `docker-compose.yml` to ensure the device is mounted in the container, and modify `iridium_extractor/iridium_extractor.py` to use the new device as a source.


### Analysis

The `analysis` directory contains Jupyter notebooks for processing the data and producing the plots and numbers used in the paper.
The notebooks may be opened without running to see the results in context, or executed to reproduce the results.

The TensorFlow Docker container should contain all the required dependencies to run the notebooks.
See [Setup](#Setup) for requirements to run outside docker.

Note that these may require a large amount of RAM. A GPU is also recommended.

The notebooks contain the following:
- `jamming.ipynb` contains the evaluation of jamming Iridium messages.
- `plots-data.ipynb` contains plots relating to the raw data (amount, distribution, etc).
- `plots-models.ipynb` contains the code for running the `SatIQ` model on the data.

> [!IMPORTANT]
> The `plots-models.ipynb` notebook expects the trained model weights to be located in `data/models/ae-triplet-final.h5`.
> This can be downloaded from https://zenodo.org/record/8298532.


## Contribute

This code and dataset have been made public to aid future research in this area.
However, this reposistory is no longer actively developed.
Any contributions (documentation, bug fixes, etc.) should be made as pull requests, and may be accepted.

