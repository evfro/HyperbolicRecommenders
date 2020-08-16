# Hyperbolic (ordinary and variational) autoencoders for recommender systems
Accompanying code for the paper

## Results
<p align="middle">
  <img src="assets/netflix.png" />
  <img src="assets/ml20m.png"/> 
  <img src="assets/pinterest.png" />   
  <img src="assets/ML1M_.png" /> 
</p>


## Data
To reproduce our code, please put the data files in the following order:

data
  * recvae
      * ml20m
  * troublinganalysis
      * mvae
          * netflix
      * neumf
          * ml1m
          * pinterest

Also, please install geoopt package [geoopt](https://github.com/geoopt) for Riemannian optimization and [hyptorch](https://github.com/leymir/hyperbolic-image-embeddings) for computations in hyperbolic spaces.

## Wandb
In our experiments, we have used [wandb](http://wandb.com) framework for result tracking. Our test scripts are based on wandb configs.

## Acknowledgments
In our code we have used the following repositories:
* [mvae](https://github.com/oskopek/mvae)
* [geoopt](https://github.com/geoopt)
* [hyptorch](https://github.com/leymir/hyperbolic-image-embeddings)
