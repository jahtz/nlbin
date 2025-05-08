# nlbin

Generate binary and normalized versions of a set of input images using OCRopus nlbin algorithm.

## Setup Docker

### Docker - CPU only

>[!WARNING]
> For large datasets, you should consider using a GPU build.

```shell
docker pull ghcr.io/jahtz/nlbin:latest
```

```shell
sudo docker run --rm -it -v $(pwd):/data ghcr.io/jahtz/nlbin:latest IMAGES... [OPTIONS]
```

### Docker - CUDA 12.5

>[!NOTE]
> For other CUDA or ROCm versions, see the build guide below.

```shell
docker pull ghcr.io/jahtz/nlbin:latest-cuda12
```

```shell
sudo docker run --rm -it --gpus all -v $(pwd):/data ghcr.io/jahtz/nlbin:latest-cuda12 IMAGES... [OPTIONS]
```

### Docker - CUDA11 / ROCm 4.3 / ROCm 5.0

>[!NOTE]
> This requires building your own docker image. Example: `CUDA 12.5`

1. Clone repository

    ```shell
    git clone https://github.com/jahtz/nlbin
    ```

2. Build the image

    ```shell
    docker build -f cuda12.Dockerfile -t nlbin .
    ```

3. Run with

    ```shell
    sudo docker run --rm -it --gpus all -v $(pwd):/data nlbin IMAGES... [OPTIONS]
    ```

## Setup (PIP)

### PIP

>[!TIP]
> Use a virtual enviroment, e.g. with [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix).

1. Clone repository

    ```shell
    git clone https://github.com/jahtz/nlbin
    ```

2. Install
    1. CPU only

        ```shell
        pip install nlbin/.
        ```

    2. GPU support

        ```shell
        pip install nlbin/.[cuda12]
        ```

        Supported versions: `cuda11`, `cuda12`, `rocm4-3`, `rocm5-0`

3. (Optional) Set `LD_LIBRARY_PATH` to the correct GPU runtime

    ```shell
    export LD_LIBRARY_PATH="/usr/local/<version>/lib64:$LD_LIBRARY_PATH"
    ```

4. Run

    ```shell
    nlbin IMAGES... [OPTIONS]
    ```

## Usage

```txt
$ nlbin --help
Usage: nlbin [OPTIONS] IMAGES...

  Normalize and binarize images using OCRopus nlbin algorithm.

  IMAGES: List of image file paths to process. Accepts individual files, glob
  wildcards, or directories.

  If you want to use your GPU, consider installing cupy. See README.md for
  further information.

Options:
  --help                      Show this message and exit.
  --version                   Show the version and exit.
  -o, --output DIRECTORY      Specify output directory for processed files.
                              Defaults to the parent directory of each input
                              file.
  -b, --binarize              Save the binary (black and white) version of
                              each image.
  -n, --normalize             Save the normalized version of each image.
  --bin-suffix TEXT           Specify suffix for binary output images.
                              [default: .ocropus.bin.png]
  --nrm-suffix TEXT           Specify suffix for normalized output images.
                              [default: .ocropus.nrm.png]
  --gpu / --cpu               Select computation device. Use '--gpu' for CUDA
                              or ROCm GPU acceleration (requires cupy). GPU
                              available: False  [default: cpu]
  --threshold FLOAT RANGE     Set binarization threshold.  [default: 0.5;
                              0.0<=x<=1.0]
  --zoom FLOAT RANGE          Zoom level for estimating page background.
                              [default: 0.5; 0.0<=x<=1.0]
  --scale FLOAT RANGE         Scale factor for defining the text region mask.
                              [default: 1.0; 0.0<=x<=1.0]
  --border FLOAT RANGE        Fraction of the image border to ignore in
                              processing.  [default: 0.1; 0.0<=x<=1.0]
  --percentage INTEGER RANGE  Percentile value for image filtering to enhance
                              contrast.  [default: 80; 0<=x<=100]
  --range INTEGER RANGE       Range for the percentile filter to adjust
                              brightness/contrast.  [default: 20; 0<=x<=100]
  --low INTEGER RANGE         Lower percentage threshold for black level
                              estimation.  [default: 5; 0<=x<=100]
  --high INTEGER RANGE        Upper percentage threshold for white level
                              estimation.  [default: 90; 0<=x<=100]
```

## ZPD

Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).
