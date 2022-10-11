# NeuralizedCRF

A general framework to embed Markov Random Field into deep neural networks for structure prediction tasks, e.g. stereo matching, image colorization, image denoising etc. 

![Architecture](/vis/arch.png)

## Installation

Use the package manager pip or conda to install your tensorflow 2.2 GPU environment and other dependencies.

```bash
conda create -n tf python=3.8 cudatoolkit=10.1 cudnn tensoflow=2.2 matplotlib pillow 
```
Download corresponding datasets if you want to train your own model weights, pretrained weights download link:
<a id="raw-url" href="https://drive.google.com/drive/u/0/shared-with-me">Download Weights</a>


## Usage

We currently applied this framework on stereo matching task and image colorization task.

```bash
# train stereo matching model from scratch
python stereo.py

# training image colorization model from scratch
python colorization.py

# colorize your image using pretrained model
python colorization.py --colorize --img_dir="your/test/image/directory" --weights="weights/file/path"
```

### example output

Stereo Matching:

![Stereo](/vis/stereo-visual-comp.jpg)

Colorization:

![Colorization-bruce-lee](/vis/colorize-brucelee.jpg)

![Colorization-girl](/vis/colorize-girl.jpg)

![Colorization-chat](/vis/colorize-chat.jpg)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)