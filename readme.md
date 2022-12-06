# NFT Maker

The aim to this program is to use a Deep Convolutional GAN (Generative Adversarial Network) using [Tensorflow](https://www.tensorflow.org/tutorials/generative/dcgan).


This program is made possible thanks to another repository: [NFTGrabber](https://github.com/ericmurphyxyz/nftgrabber)


 ## How to run?
 
 - First, download some NFT images:
    - install dependencies: `npm start`
    - run `npm start [link]` *(ex:"npm start https://rarible.com/boredapeyachtclub")*
    
 - Then use `avif_converter.py` to convert the images in PNG
 
 - Finally run `gan_2_tf_module.py` to run the gan
 
 In an epoch, the discriminator is trained wuth the database, then the generator is trained using the discriminator
 
 At the end of each epoch, an image is saved to the [generated_images](generated_images) folder


## Author

Romain Dodet

- Twitter🐦: @Romain_Dodet_

Hope it will be helpful :)
