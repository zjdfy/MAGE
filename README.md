# MAGE: Multi-stage Generation with Sparse Observations

## Enviroment Setup
```
conda env create -f environment.yml
conda activate mage
```
The code was tested on Python 3.9 and PyTorch 2.5.1

## Dataset Preparation
Please download the AMASS dataset from (https://amass.is.tue.mpg.de/)(Choosing SMPL+H G, Dataset1 needs BMLrub(BioMotionLab_NTroje),CMU,HDM05(MPI_HDM05)).
                    SMPL/DMPLs  from (https://smpl.is.tue.mpg.de/download.php).
You need to register before downloading these datasets and models.
```
#head and wrists observations
python dataprocess/prepare_data.py --support_dir /path/to/your/smplh/dmpls --save_dir ./dataset/AMASS/ --root_dir /path/to/your/amass/dataset
#root joint as an additional input
python dataprocess/prepare_data_4joint.py --support_dir /path/to/your/smplh/dmpls --save_dir ./dataset/AMASS/ --root_dir /path/to/your/amass/dataset 
```
The generated dataset should look like this
```
./dataset/AMASS/
├── BioMotionLab_NTroje
├──── train/
├──── test/
├── CMU/
├──── train/
├──── test/
└── MPI_HDM05/
├──── train/
└──── test/
```
## Evaluation
To evaluate the model:
```
python test.py --model_path /path/to/your/model --timestep_respacing ddim4 --support_dir /path/to/your/smpls/dmpls --dataset_path ./dataset/AMASS/ --input_motion_length 120 --overlapping_test --sld_wind_size 108
# if you need visualize the result, please add these parameters 
--vis --output_dir video
```
## Training
To train the MAGE diffusion-model:
```
python train.py --save_dir /path/to/save/your/model --dataset amass --weight_decay 1e-4 --batch_size 512 --lr 3e-4 --latent_dim 512 --save_interval 1 --log_interval 1 --device 0 --input_motion_length 120 --diffusion_steps 1000 --num_workers 8 --motion_nfeat 132 --arch diffusion_DiffMLP --layers 36 --sparse_dim 54 --train_dataset_repeat_times 500 --lr_anneal_steps 600000 --overwrite
```