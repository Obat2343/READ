# Motion Planning algorithms
## Install

```sh
git clone Anonymous:will/be/updated
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

Please train a VAE first.

```sh
cd main
python DF/Train_ACTOR.py --config_path ../config/Train_ACTOR.yaml
```

### Train

```sh
cd main
python DF/Train_Retrieval_IRSDE.py
```

### Test

```sh
cd main
python R2Diff/Evaluate_Retrieval_IRSDE.py --config_path ../config/yamlfilename --model_path /path/to/pthfile --tasks PickUpCup --inf_method_list retrieve_from_SPE
```

***

