# Walking Bass Transcription

Algorithm for walking bass transcription in jazz ensemble recordings using Deep Neural Networks (DNN)
  - J. Abeßer, S. Balke, K. Frieler, M. Pfleiderer, M. Müller: Deep Learning for Jazz Walking Bass Transcription, AES conference on Semantic Audio, Erlangen, Germany, 2017

Audio examples can be found here:
  - http://www.audiolabs-erlangen.de/resources/MIR/2017-AES-WalkingBassTranscription/

This algorithm was used to create the beat-wise bass pitch values included in the Weimar Jazz Database
  - http://jazzomat.hfm-weimar.de/dbformat/dboverview.html

The script requires the Python packages
  - python 3.X
  - numpy
  - scipy
  - keras
  - librosa
  - pysoundfile
  - h5py

We recommend you to install *miniconda* (https://conda.io/miniconda.html).
You can create a suitable environment using
```
conda env create -f conda_environment.yml
```
and activated it via
```
source activate walking_bass_transcription
```

## Demo

Now you can run the transcription algorithm on a test file by calling
```
python transcriber.py
```
Please see the docstrings for further documentation.

![Sonic Visualizer Screenshot](data/Sonic_Visualizer_Screenshot.png "Bass saliency for excerpt from Art Pepper's solo on Anthropology")


Enjoy.
