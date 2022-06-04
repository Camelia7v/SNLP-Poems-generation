# GenROPO: Generating Romanian poems

Project done during "Statistical Natural Language Processing"
course.

**Working team**

1.  Lupancu Viorica-Camelia (MLC)
    
2.  Mocanu Ada-Astrid (MSAI)
    
3.  Plătică Alexandru-Gabriel (MLC)
    
4.  Știrbu Alexandru-Ilie (MSAI)

Team coordinator: Trandabăț Diana

* ### [Project presentation](https://docs.google.com/presentation/d/1OEmU0-l95MiG9l12FkKJu3kIpJxWoWl6cnZatlOcF84/edit#slide=id.g13142c097a7_0_6)
* ### [Project description](https://drive.google.com/file/d/11PV2W1yGDhHCcWqtjPE_JAp1Vk2aP6V8/view?usp=sharing)

## State-of-the-art

* https://arxiv.org/abs/1909.09534 - Creative GANs for generating poems, lyrics, and metaphors - a GAN approach for generating poems in English

* https://aclanthology.org/P17-1016.pdf - Automatically Generating Rhythmic Verse with Neural Networks - a softer method based on a LSTM (English) and applied poetic directions

* https://arxiv.org/pdf/2102.04114.pdf - GENERATE AND REVISE: REINFORCEMENT LEARNING IN NEURAL POETRY - a method based on a bi-LSTM, including several experiments and postprocessing techniques (English)

* https://aclanthology.org/D14-1074.pdf - Chinese Poetry Generation with Recurrent Neural Networks - poem generator experiments on convolutional and recurrent networks, tested using BLEU score and human volunteers.

## Conclusions
 The common point of all these papers consists in Neural Network type approaches, more precisely a Deep Learning one.
 
 We will try to use a GAN architecture in which the Generator is a Transformer and the Discriminator is a simple network.

 We consider each word a token meant to be generated by the Generator along with other tokens in order to create a strophe/stanza.
 
 Our goal is to generate plausible poems in Romanian language.
 
## Work per module
* Collecting data Module: Lupancu Camelia

* Encoding data Module: Platica Alexandru

* Processing data (Tools and Generator): Mocanu Astrid

* Processing data (Discriminator and GAN): Stirbu Alexandru

* Postprocessing & Testing: Stirbu Alexndru & Mocanu Astrid

* Documentation & Presentation: Mocanu Astrid & Lupancu Camelia

 
## Arhitecture

![Untitled Diagram](https://user-images.githubusercontent.com/62291817/166889698-3337c92a-32e5-4906-9447-ba8c3cc4e9c1.svg)

