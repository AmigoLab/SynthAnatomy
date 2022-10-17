# !! The codebase will not actively maintained !!
# Morphology-preserving Autoregressive 3D Generative Modelling of the Brain 
This codebase was used in generating the results of the paper Morphology-preserving Autoregressive 3D Generative Modelling of the Brain  which was accepted at the MICCAI 2022 workshop SASHIMI.  

Preliminary work can be visualized [here](http://amigos.ai/thisbraindoesnotexist/).

# Pretrained models

To use the pretrained models you need to do the following:

0) Create a docker container based on the Dockerfile and requirements file found in the dcoker folder
1) Create a folder similar with the following structure where you replace 'experiment_name' with the name of your experiment and you chose either baseline_vqvae or performer depending on which weights you want to use:
```
<<experiment_name>>
├── baseline_vqvae/performer
    ├── checkpoints 
    ├── logs
    └── outputs
```
2) Download the weights of the desired model from the links below and put it the checkpoints folder:
* [VQ-VAE UKB](https://drive.google.com/file/d/1ETfWg0g1tEH98dKK2INl30Ol9ID5zC7w/view?usp=sharing)
* [VQ-VAE ADNI](https://drive.google.com/file/d/1PK_ur0WKC00jA22cBzGcbgMVacSzDNdy/view?usp=sharing)
* [Transformer UKB Young](https://drive.google.com/file/d/1R-70AH11i7CsRYnvygohTY5mxO6kUviG/view?usp=sharing)
* [Transformer UKB Old](https://drive.google.com/file/d/12ilo5aEwOBUqRWN6aRHIWOk4-q1Z_g3-/view?usp=sharing)
* [Transformer UKB Small Ventricles](https://drive.google.com/file/d/14I6MRSFDCNOf2_KbmOIOc_gQUZIvbvo1/view?usp=sharing)
* [Transformer UKB Big Ventricles](https://drive.google.com/file/d/1XaLSNjpthGNBzMIhOMq8vl4DFtYQx4Sj/view?usp=sharing)
* [Transformer ADNI Cognitively Normal](https://drive.google.com/file/d/1AjAA6jVTp3syh86xJ3sFphWAX2amx7Fz/view?usp=sharing)
* [Transformer ADNI Alzheimer Diseased](https://drive.google.com/file/d/1kbEqCF3UyazVAXC2PnM8pTlOe1xzbY2p/view?usp=sharing)
3) Rename the file to 'checkpoint_epoch=0.pt'
4) Use the corresponding script from the examples bellow and remember to:
* Replace the training/validation subjects with paths towards either folder filled with .nii.gz files or towards csv/tsv files that have a path column with the full paths towards the files.
* Replace the project_directory with the path were you created the folder from point 1
* Replace the experiment_name with the name of the experiment you created from point 1
5) Properly mount the paths towards the files and results folders and launch your docker container
6) Use the appropriate script for the model from bellow and change the mode to the desired one

# VQ-VAE

To extract the quantized latent representations of the images you need to run the same command as you used for training and replace the `--mode=Training` parameter with `--mode=extracting`. For decoding, you need to replace it with `--mode=decoding`.

Training script example for VQ-VAE.
```bash
python /project/run_vqvae.py run \
    --training_subjects="/path/to/training/data/tsv/" \
    --validation_subjects="/path/to/validation/data/tsv/" \
    --load_nii_canonical=False \
    --project_directory="/results/" \
    --experiment_name="example_run" \
    --mode='training' \
    --device='ddp' \
    --distributed_port=29500 \
    --amp=True \
    --deterministic=False \
    --cuda_benchmark=True \
    --seed=4 \
    --epochs=500 \
    --learning_rate=0.000165 \
    --gamma=0.99999 \
    --log_every=1 \
    --checkpoint_every=1 \
    --eval_every=1 \
    --loss='jukebox_perceptual' \
    --adversarial_component=True \
    --discriminator_network='baseline_discriminator' \
    --discriminator_learning_rate=5e-05 \
    --discriminator_loss='least_square' \
    --generator_loss='least_square' \
    --initial_factor_value=0 \
    --initial_factor_steps=25 \
    --max_factor_steps=50 \
    --max_factor_value=5 \
    --batch_size=8 \
    --normalize=True \
    --roi='((16,176), (16,240),(96,256))' \
    --eval_batch_size=8 \
    --num_workers=8 \
    --prefetch_factor=8 \
    --starting_epoch=172 \
    --network='baseline_vqvae' \
    --use_subpixel_conv=False \
    --use_slim_residual=True \
    --no_levels=4 \
    --downsample_parameters='((4,2,1,1),(4,2,1,1),(4,2,1,1),(4,2,1,1))' \
    --upsample_parameters='((4,2,1,0,1),(4,2,1,0,1),(4,2,1,0,1),(4,2,1,0,1))' \
    --no_res_layers=3 \
    --no_channels=256 \
    --codebook_type='ema' \
    --num_embeddings='(2048,)' \
    --embedding_dim='(32,)' \
    --decay='(0.5,)' \
    --commitment_cost='(0.25,)' \
    --max_decay_epochs=100 \
    --dropout=0.0 \
    --act='RELU'
```

# Transformer

To sample new images from the trained model you need to run the same command as you used for training and replace the `--mode=training` parameter with `--mode=inference`.

Training script example for Transformer.
```bash
python /project/run_vqvae.py run \
    --training_subjects="/path/to/training/data/tsv/" \
    --validation_subjects="/path/to/validation/data/tsv/" \
    --project_directory="/results/" \
    --experiment_name="example_run" \
    --mode='training' \
    --deterministic=False \
    --cuda_benchmark=True \
    --device='ddp' \
    --seed=4 \
    --epochs=2000 \
    --learning_rate=0.001 \
    --gamma='auto' \
    --log_every=1 \
    --checkpoint_every=1 \
    --eval_every=1 \
    --batch_size=6 \
    --eval_batch_size=6 \
    --num_workers=8 \
    --prefetch_factor=8 \
    --starting_epoch=0 \
    --network='performer' \
    --vocab_size=2048 \
    --ordering_type='raster_scan' \
    --transpositions_axes='((2, 0, 1),)' \
    --rot90_axes='((0, 1),)' \
    --transformation_order='(\"rotate_90\", \"transpose\")' \
    --n_embed=512 \
    --n_layers=24 \
    --n_head=16 \
    --local_attn_heads=8 \
    --local_window_size=420 \
    --feature_redraw_interval=1 \
    --generalized_attention=False \
    --emb_dropout=0.0 \
    --ff_dropout=0.0 \
    --attn_dropout=0.0 \
    --use_rezero=True \
    --spatial_position_emb='absolute'
```

# Acknowledgements

Work done through the collaboration between NVIDIA and KCL using [Cambridge-1](https://www.nvidia.com/en-us/industries/healthcare-life-sciences/cambridge-1/).

# Funding
- Jointly with UCL - Wellcome Flagship Programme (WT213038/Z/18/Z)
- Wellcome/EPSRC Centre for Medical Engineering (WT203148/Z/16/Z)
- EPSRC Research Council DTP(EP/R513064/1)

# Reference

If you use our work please cite:

```
@inproceedings{tudosiu2022morphology,
  title={Morphology-Preserving Autoregressive 3D Generative Modelling of the Brain},
  author={Tudosiu, Petru-Daniel and Pinaya, Walter Hugo Lopez and Graham, Mark S and Borges, Pedro and Fernandez, Virginia and Yang, Dai and Appleyard, Jeremy and Novati, Guido and Mehra, Disha and Vella, Mike and others},
  booktitle={International Workshop on Simulation and Synthesis in Medical Imaging},
  pages={66--78},
  year={2022},
  organization={Springer}
}
```
