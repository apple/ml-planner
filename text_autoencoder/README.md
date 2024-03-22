# text_autoencoder-pytorch
Text auto-encoder and classification model in PyTorch.  

**This repository is still developing.**

# Requirements
- python>=3.6
- torch>=1.4.0
- transformers==4.21.0

## Usage
### Model choices

**Encoder choices:**
- Convolution encoder
- Bert encoder
  
**Decoder choices:**
- Deconvolution decoder
- GPT2 decoder



### Train
#### Paragraph reconstruction
Download preprocessed tripadvisor data. [link](s3://yizhe-data/textcnn/trip) 
Then, run following command.


```shell
$ export PYTHONPATH="${PYTHONPATH}:/path/to/text_autoencoder
$ python text_autoencoder/autoencoder/train.py --train_pt_dir=trip/pts/train_pt_dir --dev_pt_dir=trip/pts/dev_pt_dir --enc_model <encoder choice> --dec_model <decoder choice> \
```

Specify download data path by `--train_pt_dir` and `--dev_pt_dir`.

About other parameters.

```
usage: autoencoder/train.py
  -h, --help            show this help message and exit
  --seed SEED
  --lr LR               initial learning rate
  --enc_lr ENC_LR
  --dec_lr DEC_LR
  --epochs EPOCHS       number of epochs for train
  --batch_size BATCH_SIZE
                        batch size for training
  --dropout DROPOUT     dropout ratio
  --save_dir SAVE_DIR   where to save the snapshot
  --train_pt_dir TRAIN_PT_DIR
  --dev_pt_dir DEV_PT_DIR
  --sentence_len SENTENCE_LEN
                        how many tokens in a sentence
  --enc_model ENC_MODEL 
                        encoder model
  --reg_layer {bn,ln,none}
                        regularization layer
  --dec_model DEC_MODEL 
                        encoder model
  --embed_dim EMBED_DIM 
                        number of embedding dimension
  --filter_size FILTER_SIZE
                        filter size of convolution
  --filter_shape FILTER_SHAPE
                        filter shape to use for convolution
  --latent_size LATENT_SIZE
                        size of latent variable
  --resume_ckpt RESUME_CKPT
  --exp_name EXP_NAME   experiment name
  --noiser {bart,bert,sub,none}
```

- Changing `latent_size` to value other than 768 will make the Bert and GPT2 model train from scratch. 
- `noiser` controls the noise added to the input. 

## Reference
[Deconvolutional Paragraph Representation Learning](https://arxiv.org/abs/1708.04729v3)  
Yizhe Zhang, Dinghan Shen, Guoyin Wang, Zhe Gan, Ricardo Henao, Lawrence Carin. NIPS 2017
[INSET: Sentence Infilling with INter-SEntential Transformer](https://arxiv.org/abs/1911.03892)  
Yichen Huang*, Yizhe Zhang*, Oussama Elachqar, Yu Cheng. ACL 2020
[Narrative Incoherence Detection](https://arxiv.org/pdf/2012.11157)  
Deng Cai, Yizhe Zhang, Yichen Huang, Wai Lam, Bill Dolan. eprint arXiv:2012.11157