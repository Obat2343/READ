# Motion Planning algorithms
## Install

```sh
git clone git@github.com:Obat2343/IBC.git
mkdir git
cd git
```

Install followings in the git directory.

- Pyrep (<https://github.com/stepjam/PyRep>)
- CoppeliaSim (<https://www.coppeliarobotics.com/downloads.html>) # Please check the Pyrep repository to confirm the version of CoppeliaSim
- RLBench (<https://github.com/Obat2343/RLBench>)
- RotationConinuity (<https://github.com/papagina/RotationContinuity>)

Next, Install requirements

```sh
pip install -r requirements.txt
```

## Prepare Dataset

To create the dataset for training and testing, please run the following command.

```sh
python create_dataset.py --task_list TaskA TaskB
```

***

## R2Diff

### Download Pre-trained weights

```sh
mkdir result
cd result
```

Please download and unzip the file from https://drive.google.com/file/d/1ECP7Vsz7HkC7dbgYmnI7gG1zVZAlwaXM/view?usp=share_link

### Train

```sh
cd main
python R2Diff/Train_Diffusion.py --config_file ../config/RLBench_Diffusion.yaml
```

### Test

```sh
cd main
python R2Diff/Evaluate_Diffusion_on_sim.py --config_path ../config/Test_config.yaml --diffusion_path ../weights/RLBench/PickUpCup/Diffusion_frame_100_mode_6d_step_1000_start_1e-05_auto_rank_1/model/model_iter50000.pth --tasks PickUpCup --inf_method_list retrieve_from_SPE
```

***

## DMOEBM

### Train
Please train a VAE first.

```sh
cd main
python DMOEBM/Train_VAE.py --tasks PickUpCup --config_file ../RLBench_VAE.yaml
```

Then train EBM and DMO.

```sh
python DMOEBM/Train_EBM.py --tasks PickUpCup --config_file ../Transformer_EBM.yaml
```

```sh
python DMOEBM/Train_iterative_DMO.py --tasks PickUpCup --config_file ../RLBench_DMO.yaml
```

### Test

```sh
cd main
python DMOEBM/Evaluate_EBMDMO_on_sim.py --config_path ../Test_DMOEBM.yaml --EBM_path ../weights/RLBench/PickUpCup/EBM_aug_frame_100_mode_6d_first_Transformer_vae_256_and_random_second_none_inf_sort/model/model_iter50000.pth --DMO_path ../weights/RLBench/PickUpCup/DMO_iterative_5_frame_100_mode_6d_noise_Transformer_vae_256/model/model_iter100000.pth --tasks PickUpCup --inf_method_list DMO_keep
```
