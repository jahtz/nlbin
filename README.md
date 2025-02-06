# nlbin

Calculate binary and normalized versions of a set of input images using OCRopus nlbin algorithm.

## Setup
>[!NOTE]
> Tested Python versions: `3.12.7`

>[!IMPORTANT]
>The following setup process uses [PyEnv](https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix)

1. Clone repository
	```shell
	git clone https://github.com/jahtz/nlbin
	```

2. Create Virtual Environment
	```shell
	pyenv install 3.13.1
	pyenv virtualenv 3.13.1 nlbin
	pyenv activate nlbin
	```

3. Install dependencies
	```shell
	pip install -r nlbin/requirements.txt
	```

## Setup GPU
>[!NOTE]
>CUDA or ROCm needs to be added to _LD\_LIBRARY\_PATH_, e.g.:<br>
>`export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH"`

For GPU computation, **ONE** of the following _cupy_ packages is required.
You can uncomment in _requirements.txt_ or install manually:
- CUDA _11.x_: `pip install cupy-cuda11x`
- CUDA _12.x_: `pip install cupy-cuda12x`
- ROCm _4.3_: `pip install cupy-rocm-4-3`
- ROCm _5.0_: `pip install cupy-rocm-5-0`

## Usage
```
> python nlbin --help
                                                                                    
 Usage: nlbin [OPTIONS] IMAGES...                                                       
                                                                                    
 Generate binary and normalized versions of input images.                                 
 IMAGES: List of image file paths to process. Accepts individual files, wildcards, or 
 directories (with -g option for pattern matching).                                       
                                                                                          
╭─ Input ────────────────────────────────────────────────────────────────────────────────╮
│ *  IMAGES    PATH  [required]                                                          │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────╮
│ --binary         -B             Save the binary (black and white) version of each      │
│                                 image.                                                 │
│ --normalized     -N             Save the normalized version of each image.             │
│ --glob           -g  TEXT       Glob pattern for matching images within directories.   │
│                                 Only applicable when directories are passed in IMAGES. │
│                                 [default: *.png]                                       │
│ --output         -o  DIRECTORY  Specify output directory for processed files. Defaults │
│                                 to the parent directory of each input file.            │
│ --device         -d  [cpu|gpu]  Select computation device. Use `gpu` for NVIDIA or AMD │
│                                 GPU acceleration (requires cupy).                      │
│                                 [default: cpu]                                         │
│ --keep-suffixes  -s             Preserve all filename suffixes except the last one. If │
│                                 not set, removes all suffixes.                         │
│ --bin-suffix         TEXT       Specify suffix for binary output images.               │
│                                 [default: .ocropus.bin.png]                            │
│ --nrm-suffix         TEXT       Specify suffix for normalized output images.           │
│                                 [default: .ocropus.nrm.png]                            │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Fine-tuning ──────────────────────────────────────────────────────────────────────────╮
│ --threshold     FLOAT RANGE    Set binarization threshold. [default: 0.5; 0.0<=x<=1.0] │
│ --zoom          FLOAT RANGE    Zoom level for estimating page background.              │
│                                [default: 0.5; 0.0<=x<=1.0]                             │
│ --scale         FLOAT RANGE    Scale factor for defining the text region mask.         │
│                                [default: 1.0; 0.0<=x<=1.0]                             │
│ --border        FLOAT RANGE    Fraction of the image border to ignore in processing.   │
│                                [default: 0.1; 0.0<=x<=1.0]                             │
│ --percentage    INTEGER RANGE  Percentile value for image filtering to enhance         │
│                                contrast.                                               │
│                                [default: 80; 0<=x<=100]                                │
│ --range         INTEGER RANGE  Range for the percentile filter to adjust               │
│                                brightness/contrast.                                    │
│                                [default: 20; 0<=x<=100]                                │
│ --low           INTEGER RANGE  Lower percentage threshold for black level estimation.  │
│                                [default: 5; 0<=x<=100]                                 │
│ --high          INTEGER RANGE  Upper percentage threshold for white level estimation.  │
│                                [default: 90; 0<=x<=100]                                │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Help ─────────────────────────────────────────────────────────────────────────────────╮
│ --help         Show this message and exit.                                             │
│ --version      Show the version and exit.                                              │
╰────────────────────────────────────────────────────────────────────────────────────────╯
```

## ZPD
Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).
