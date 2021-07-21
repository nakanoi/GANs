# MANY KINDS OF GANs
## Modularized GANs
To make building network much more easier than ever for me, I build many kinds of GANs' modules.
I used `tensorflow version 2` and `tf.keras`.
I'm still adding other networks to this repo.

You can run these networks on google colaboratory.
Open and run `.ipynb` files on it.

These networks can train not only `tf.data.Datasets` but also your own images data. Check Here when you want to use images you prepared.

## Environments
|  | Version |
| ------ | ------ |
| Ubuntu | 20.04.2 LTS |
| conda | 4.10.3 |
| Python | 3.9.5 |
| tensorflow | 2.4.1 |
| numpy | 1.20.3 |
| matplotlib | 3.3.4 |

If you want to see more environment infomation, view conda_requirements.txt.
It has package list.

## GANs list

- GAN
-- `models/gan.py`
-- paper's link [here](https://arxiv.org/abs/1406.2661)
- Deep Convolutional GAN
-- `models/dcgan.py`
-- paper's link [here](https://arxiv.org/abs/1511.06434)
- Wasserstein GAN
-- `models/wgan.py`
-- paper's link [here](https://arxiv.org/abs/1701.07875)
- Wasserstein GAN with Gradient Penalty
-- `models/wgangp.py`
-- paper's link [here](https://arxiv.org/abs/1704.00028)
- Cycle GAN
-- `models/cyclegan.py`
-- paper's link [here](https://arxiv.org/abs/1703.10593)

## Usage
Each test file is already ready for train mnist images.
```sh
cd GANs
```
```python
python test_**.py
```

## Datasets
### tfds.data.Datasets
When you use [tensorflow's dataset collection](https://www.tensorflow.org/datasets), follow bellow procedures.

1. Install tensorflow datasets.
```sh
conda install -c anaconda tensorflow-datasets
or
pip install tensorflow-datasets
```

2. Change arguments like below in test file(test**.py). If you don't know which datasets you can use, check [here](https://www.tensorflow.org/datasets/catalog/overview#all_datasets)
```python
# Model
model = GAN(
    data_name='celeb_a',
    )

# Load Data
x_train = model.loader.load_tf_data(split='train')

# Train
model.train(
    BATCH_SIZE,
    MAX_EPOCHS,
    SHOW_EVERY_N,
    train_data=[x_train]
    )
```

3. Run & train your model.
```python
python test_**.py
```

### Your own images
On the other hand, if you prepared dataset by yourself, follow below as well.
1. Put your Dataset Folders on `GANs/datasets`. Folder name must be 
-- `train` & `test`
or
-- `trainA`, `trainB`, `testA` & `testB` when you train cycleGAN

2. Change arguments like below in test file(test**.py) as well as tfds.data version.
```python
# Model
model = GAN(
    # This variable 'data_name' must be same as dataset folder name.
    # In this case, folder path is 'GANs/datasets/mnist'
    # and 'mnist' folder has 'train' and 'test' folders.
    data_name='mnist', 
    )

# Load Data
x_train = model.loader.load_data()

# Train
model.train(
    BATCH_SIZE,
    MAX_EPOCHS,
    SHOW_EVERY_N,
    train_data=[x_train]
    )
```

## Preference
I got inspired from [@davidADSP ](https://github.com/davidADSP).
I would like to take this moment to say thank you so much.

## License

MIT

## Thank you for visiting my repository!
