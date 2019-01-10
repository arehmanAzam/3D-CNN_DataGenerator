# 3D-CNN_DataGenerator
DataGenerator for 3D-CNN in keras and tensorflow

This repo contains the code of data generator for 3DCNN architectures. The code was built using multiple resources but most of the content is taken from [Shervine Amidi](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)

## Getting Started
We expect to have proir knowledge of keras and tensorflow. Also see keras documentation for fitgenerator and ImageGenerator.

### Prerequisites

You can use any version of the packages but we have used the following versions: 

Packages      | Versions
------------- | -------------
Keras         | 2.1.3
Tensorflow    | 1.8.0
Numpy         | 15.4.0
Matplotlib    | 3.0.2

### Getting Started

Before getting to this point of work, your code might look like this 

```

# Load entire dataset
path_dataset_train='/home/activity/train'
path_dataset_val='/home/activity/val'

# Design model
model.Sequential()
.....
# Train model on your dataset
model.fit_generator()
```
#### 
In python headers where packages are imported code this line
```
from DataGenerator import *
from dataset import load_cached
```
then pass path of the dataset folders to make or load cache_files for future usage

```
dataset_train=load_cached(cache_path='give path for making or loading .pkl file ',in_dir=path_dataset_train)
dataset_val=load_cached(cache_path='give path for making or loading .pkl file ',in_dir=path_dataset_val)
class_names=dataset_train.class_names
```
verify it by 

```
print(class_names)
```

Now get images path and labels using 

```
image_path_train,cls_train,labels_train=dataset_train.get_training_set()
image_path_val, cls_val,labels_val=dataset_val.get_training_set()
num_classes= dataset_train.num_classes
```

Concatenate both paths with parameters

```
partition={'train': image_path_train, 'validation': image_path_val}
```

Create and change parameters according to your needs for this 3D generators.

```
number_of_batches=10
frame_width=400
frame_height=256
frames_chunk=18
params={'dim': (frames_chunk,frame_width,frame_height),
        'n_channels': 3,
        'real_batchsize_custom': number_of_batches,
        'batch_size': frames_chunk*number_of_batches,
        'n_classes': num_classes,
        'shuffle': False
        }
```

Get labels from the findLabels function

```
labels_training=findLabels(partition['train'], labels_train)
labels_validation=findLabels(partition['validation'], labels_val)
```


Now pass the parameters to the main DataGenerator Class

```
training_generator = DataGenerator(partition['train'], labels_training, **params)
validation_generator = DataGenerator(partition['validation'], labels_validation, **params)
```

Now pass the generators to your model.fit_generator() function for training the model

```
model.fit_generator(...
        generator=training_generator,
        validation_data=validation_generator,
        ..
        )
```
