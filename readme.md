This repo contains scripts for executing bird id models

The scripts work with any models that have the same shape as those used in Merlin Bird ID

|Model|Decription|Inputs|
|-|-|-|
|Photo id model|Classify birds in an image|224x224 RGB image (`float32[1,224,224,3]`)|
|Sound id model|Classify bird sounds in a spectrogram image|512x128 RGB image (`float32[1,128,512,3`]). The `float32` values should be normailzed to be between 0 and 1|
|Spectrogram generation model|Geneates spectrogram data for the sound id model. Run this over a 22050 Hz .wav file with hop size 128 and window size of 512|512 audio samples (`float32[512]`). The `float32` values should be normalized to be between -1 and 1|
|Geo model|Gives bird probabilities based on location and time of year|longitude (`float32`), week of year (`float32`), latitude (`float32`)|
