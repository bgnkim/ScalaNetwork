ScalaNetwork 0.1.0
==================

A **Neural Network implementation** with Scala & [Breeze](https://github.com/scalanlp/breeze)

# Features

## Network
ScalaNetwork supports following layered neural network implementation:

* *Fully-connected* Neural Network : ![equation](http://www.sciweavers.org/tex2img.php?eq=f%28Wx%2Bb%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
* *Fully-connected* Rank-3 Tensor Network : ![equation](http://www.sciweavers.org/tex2img.php?eq=f%28v_1Q%5E%7B%5B1%3Ak%5D%7Dv_2%20%2B%20L%5E%7B%5B1%3Ak%5D%7Dv%2Bb%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
* *Fully-connected* Auto Encoder
* *Fully-connected* Stacked Auto Encoder

## Training Methodology
ScalaNetwork supports following training methodologies:

* Stochastic Gradient Descent w/ L1-, L2-regularization, Momentum.
* [AdaGrad](http://www.magicbroom.info/Papers/DuchiHaSi10.pdf)
* [AdaDelta](http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf)

## Activation Function
ScalaNetwork supports following activation functions:

* Sigmoid : ![equation](http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7B1%7D%7B1%2B%20e%5E%7B-x%7D%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
* HyperbolicTangent : ![equation](http://www.sciweavers.org/tex2img.php?eq=%5Ctanh%28x%29)
* Rectifier : ![equation](http://www.sciweavers.org/tex2img.php?eq=%5Cmax%5C%7Bx%2C%200%5C%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
* Softplus : ![equation](http://www.sciweavers.org/tex2img.php?eq=%5Clog%281%2Be%5Ex%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

# Usage

`Network.apply(Activation, Int*)` generates fully-connected network:

```scala
// 2(input) -> 4(hidden) -> 1(output)
val network = Network(Sigmoid, 2, 4, 1)
//Training with Squared Error
val trainer = new BasicTrainer(network, SquaredErr)
//Training gives validation error 
val err = trainer.trainWithValidation(set, validation)
```

Also you can use `new BasicNetwork(Seq[Layer], Probability)` to generate a basic network,
and `new AutoEncoder(Reconstructable, Probability)` to generate a single-layered autoencoder.

For further usage, please read scaladocs.

# Blueprint

ScalaNetwork will support these implementations:

* Recursive Auto Encoder (RAE)
* Unfolded Recursive Auto Encoder (URAE)
* Recursive Neural Tensor Network (RNTN)

Also ScalaNetwork will support these features:

* Input-dependent Weight

## Current Status

Next version(v0.2) will support RAE, URAE
