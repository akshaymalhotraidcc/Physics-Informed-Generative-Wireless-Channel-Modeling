This repo contains the source code for running the physics-based generative model described in "Physics-based Generative Models for Geometrically Consistent and Interpretable Wireless Channel Synthesis", accepted to IJCAI-25

## Requirements
- `DeepMIMOv3==0.2.8`
- `matplotlib==3.7.5`
- `numpy==1.24.1`
- `opt-einsum==3.4.0`
- `PyYAML==6.0.2`
- `scikit-learn==1.3.2`
- `scipy==1.10.1`
- `torch==2.4.1`
- `torchinfo==1.8.0`


## Process
1. Create the following directories at the root:
  - `./weighted_model_results/`
  - `./utilities/`
  - `./images/`
  - `./deepmimo_data/`
  - `./city_scenarios/`
  - `./channel_data/`
2. Download the scenario .zip files from the [DeepMIMO website](https://www.deepmimo.net/) and unzip in `./deepmimo_data/`.
3. Run `python3 get_scenario_params_v2.py --scenario SCENARIO_NAME --a_end END_NO` (`END_NO` defined by `user_rows` for the specific scenario) to generate dataset and dictionary.
  - Press `y` when prompted to generate dataset.
  - Press `y` when prompted to generate array response dictionary.
4. Run `train_vae.py` to train the VAE model.
