# Vision Transformer
<img src="https://img.shields.io/badge/-Python-3776AB.svg?logo=python&style=plastic">
<img src="https://img.shields.io/badge/-Docker-1488C6.svg?logo=docker&style=plastic">


## Command
```
git clone git@github.com:hanoi0126/vision-transformer.git
cd vision-transformer
```

```
docker build -t vit:v1.0 .
docker run -it --gpus all -v $PWD:workspace vit:v1.0 /bin/bash
```