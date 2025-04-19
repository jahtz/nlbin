# nlbin
Calculate binary and normalized versions of a set of input images using OCRopus nlbin algorithm.

## Setup
>[!NOTE]
> Tested Versions:
> - Python: `3.11.11`
> - CUDA: `12.5`

>[!IMPORTANT]
>The following setup process uses [PyEnv](https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix)

1. Clone repository
	```shell
	git clone https://github.com/jahtz/nlbin
	```

2. Create Virtual Environment
	```shell
	pyenv install 3.11.11
	pyenv virtualenv 3.11.11 nlbin
	pyenv activate nlbin
	```

3. Install nlbin
    ```shell
    pip install nlbin/.
    ```

4. (Optional) Select CUDA version
    ```shell
    export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH"
    ```

## Setup GPU
For GPU computation, **ONE** of the following _cupy_ packages is required.
You can uncomment in _requirements.txt_ or install manually:
- CUDA _11.x_: `pip install cupy-cuda11x`
- CUDA _12.x_: `pip install cupy-cuda12x`
- ROCm _4.3_: `pip install cupy-rocm-4-3`
- ROCm _5.0_: `pip install cupy-rocm-5-0`

## Usage
```
$ nlbin --help
                                                                                          
 Usage: nlbin [OPTIONS] IMAGES...                                                         
                                                                                          
 Normalize and binarize images using OCRopus nlbin algorithm.                             
 IMAGES: List of image file paths to process. Accepts individual files, wildcards, or     
 directories (with -g option for pattern matching).                                       
 If you want to use your GPU, consider installing cupy. See README.md for further         
 information.                                                                             
                                                                                          
╭─ Input ────────────────────────────────────────────────────────────────────────────────╮
│ *  IMAGES        (PATH) [required]                                                     │
│    --glob    -g  Glob pattern for matching images within directories. Only applicable  │
│                  when directories are passed in IMAGES.                                │
│                  (TEXT)                                                                │
│                  [default: *.png]                                                      │
│    --device  -d  Select computation device. Use `gpu` for NVIDIA or AMD GPU            │
│                  acceleration (requires cupy). GPU available: False                    │
│                  (cpu|gpu)                                                             │
│                  [default: cpu]                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ───────────────────────────────────────────────────────────────────────────────╮
│ --output      -o  Specify output directory for processed files. Defaults to the parent │
│                   directory of each input file.                                        │
│                   (DIRECTORY)                                                          │
│ --binarize    -B  Save the binary (black and white) version of each image.             │
│ --bin-suffix      Specify suffix for binary output images. (TEXT)                      │
│                   [default: .ocropus.bin.png]                                          │
│ --normalize   -N  Save the normalized version of each image.                           │
│ --nrm-suffix      Specify suffix for normalized output images. (TEXT)                  │
│                   [default: .ocropus.nrm.png]                                          │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Fine-tuning ──────────────────────────────────────────────────────────────────────────╮
│ --threshold     Set binarization threshold. (FLOAT RANGE) [default: 0.5; 0.0<=x<=1.0]  │
│ --zoom          Zoom level for estimating page background. (FLOAT RANGE)               │
│                 [default: 0.5; 0.0<=x<=1.0]                                            │
│ --scale         Scale factor for defining the text region mask. (FLOAT RANGE)          │
│                 [default: 1.0; 0.0<=x<=1.0]                                            │
│ --border        Fraction of the image border to ignore in processing. (FLOAT RANGE)    │
│                 [default: 0.1; 0.0<=x<=1.0]                                            │
│ --percentage    Percentile value for image filtering to enhance contrast.              │
│                 (INTEGER RANGE)                                                        │
│                 [default: 80; 0<=x<=100]                                               │
│ --range         Range for the percentile filter to adjust brightness/contrast.         │
│                 (INTEGER RANGE)                                                        │
│                 [default: 20; 0<=x<=100]                                               │
│ --low           Lower percentage threshold for black level estimation. (INTEGER RANGE) │
│                 [default: 5; 0<=x<=100]                                                │
│ --high          Upper percentage threshold for white level estimation. (INTEGER RANGE) │
│                 [default: 90; 0<=x<=100]                                               │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Help ─────────────────────────────────────────────────────────────────────────────────╮
│ --help         Show this message and exit.                                             │
│ --version      Show the version and exit.                                              │
│ --verbose  -v  Set verbosity level. `-v`: WARNING, `-vv`: INFO, `-vvv`: DEBUG.         │
│                (INTEGER)                                                               │
╰────────────────────────────────────────────────────────────────────────────────────────╯
```

## ZPD
Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).
