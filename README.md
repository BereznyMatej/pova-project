Name: Matej Berezny, Ondrej Valo

VUT FIT - Computer Vision (POVa) - Semantic image segmentation (project theme)

Set of scripts used for training 2D neural networks for CityScapes categorical segmentation. All scripts used for data manipulation are stored in `\dataset`, while scripts handling the network training are in `\unet`. 



## Training
Training process last for set number of epochs with best versions of model saved in `\pretrained weights`. If no improvement is being made in past 100 epochs, training gets terminated preemptively.
All training is handled by `UNetTrainer` and `Logger`, which are initiated automatically by launching script `train.py`.
If user wishes to further modify the training environment besides the options provided by command line params and 
.json files, (such as changing loss function, evaluation metric) `train.py` or `UNetTrainer` have to be modified.
```sh 
python3 train.py [-h] [--model MODEL] [--path DATA] [--load_weights]  
    
    '--help', '-h' Prints the short help message
    '--model STRING' '-m STRING' specifies the name under which will be the model saved in '\pretrained_weights'
    '--path STRING' '-p STRING' name of the preprocessed dataset
    '--load_weights' '-l' If set, loads the model weights from '\pretrained_weights' folder.

```
## Inference
Runs inference on one file from directory set in `--input`. Both have to be specified, along with the model name. If user wishes to also calculate dice coeff., label has to be provided in `'{INPUT_DIR}\labels'`

```sh
python3 predict.py [-h] [--path PATH] [--model MODEL] 

    '--help', '-h' Prints the short help message
    '--path DIR' '-p DIR' directory containing input files.
    '--output DIR' '-o DIR' where to save finished predictions.
	'--save_images' '-s' saves segmentation results into seg_results folder. 
    '--model STRING' '-m STRING' specifies the model used for inference. 
```

# Pretrained models
Directory `\pretrained_weights` contains most of the models used in thesis's experiments. best versions of models are marked as `BEST_{model_name}`.
