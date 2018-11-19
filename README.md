# cats-vs-dogs
Classification examples using pytorch and keras to distinguish dogs and cats pictures

## How to run it

### You will need to:
- Download and install Python 3.6.1
- Download and install Nvidia CUDA Toolkit 9.0
- Download CuDNN for CUDA 9.0
To install CuDNN, copy the files to Nvidia CUDA toolkit(usually is located on C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0)
   - Copy cudnn\bin\cudnn64_5.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\
   - Copy cudnn\include\cudnn.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\
   - Copy cudnn\lib\x64\cudnn.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\
- Install Tensorflow via pip command prompt 'pip install --upgrade tensorflow-gpu'
- Visual C++ Redistributate 2015 x64

### Install requirements with pip or conda:

#### First create a new virtualenv

##### With Anaconda
```
conda create -n envname python=3.6 anaconda
```

##### Or
```
python3 -m venv envname
```

### Then activate it

##### With Anaconda
```
activate envname
```

##### Or (On Windows) 
```
envname\Scripts\activate.bat
```

### Now install the dependencies

##### With Anaconda
```
conda install --yes --file requirements.txt
```

##### With Pip
```
pip install -r requirements.txt
```

### To run Keras example
```
py kerasexp.py
```

### To run PyTorch example
```
py pytorchexp.py
```








