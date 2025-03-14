# Motion Planning algorithms
## Install

```sh
git clone git@github.com:Obat2343/READ.git
mkdir git
cd git
```

Install followings in the git directory.

- Pyrep (<https://github.com/stepjam/PyRep>)
- CoppeliaSim (<https://www.coppeliarobotics.com/downloads.html>) # Please check the Pyrep repository to confirm the version of CoppeliaSim
- RLBench (<https://github.com/Obat2343/RLBench>)
- RotationConinuity (<https://github.com/papagina/RotationContinuity>)

## Prepare Dataset

To create the dataset for training and testing, please run the following command.

```sh
python create_dataset.py --task_list TaskA TaskB
```

***

### Train

Please train a VAE first.

```sh
cd main
python READ/Train_ACTOR.py --config_file ../config/Train_ACTOR.yaml
```
Then, train a diffusion model.

```sh
cd main
python READ/Train_READ.py --config_file ../config/Train_READ.yaml
```

### Test

```sh
cd main
python R2Diff/Evaluate_READ.py --config_file ../config/yamlfilename --model_path /path/to/pthfile --tasks PickUpCup --inf_method_list retrieve_from_SPE
```

***

```bibtex
@inproceedings{oba2024read,
  title={READ: Retrieval-Enhanced Asymmetric Diffusion for Motion Planning},
  author={Oba, Takeru and Walter, Matthew and Ukita, Norimichi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17974--17984},
  year={2024}
}
```
