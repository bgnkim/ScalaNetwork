ScalaNetwork 0.10.2
====================

A *Neural Network implementation* with Scala, [Breeze](https://github.com/scalanlp/breeze) & [Spark](http://spark.apache.org)

Spark Network follows [GPL v2 license](http://choosealicense.com/licenses/gpl-2.0/).

# Features

## Network

ScalaNetwork supports following layered neural network implementation:

* *Fully-connected* Neural Network : f(Wx + b)
* *Fully-connected* Rank-3 Tensor Network : f(v<sub>1</sub><sup>T</sup>Q<sup>[1:k]</sup>v<sub>2</sub> + L<sup>[1:k]</sup>v + b)
* *Fully-connected* Auto Encoder
* *Fully-connected* Stacked Auto Encoder

Also you can implement following Recursive Network via training tools.

* Traditional *Recursive* Auto Encoder (RAE)
* Standard *Recursive* Auto Encoder (RAE)
* Unfolding *Recursive* Auto Encoder (RAE) <sup>[EXPERIMENTAL]</sup>

## Training Methodology

ScalaNetwork supports following training methodologies:

* Stochastic Gradient Descent w/ L1-, L2-regularization, Momentum.
* [AdaGrad](http://www.magicbroom.info/Papers/DuchiHaSi10.pdf)
* [AdaDelta](http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf)

ScalaNetwork supports following environments:

* Single-Threaded Training Environment.
* Spark-based Distributed Environment, with modified version of Downpour SGD in [DistBelief](http://research.google.com/archive/large_deep_networks_nips2012.html)

Also you can add negative examples with `Trainer.setNegativeSampler()`.

## Activation Function

ScalaNetwork supports following activation functions:

* Linear
* Sigmoid
* HyperbolicTangent
* Rectifier
* Softplus
* HardSigmoid
* HardTanh

# Usage

Here is some examples for basic usage. If you want to extend this package or use it more precisely, please refer [ScalaDoc](http://nearbydelta.github.io/ScalaNetwork/api/#kr.ac.kaist.ir.deep.package)

## Download

Currently ScalaNetwork supports Scala version 2.10 ~ 2.11.

* Stable Release is 0.10.2
 
If you are using SBT, add a dependency as described below:

```scala
libraryDependencies += "kr.ac.kaist.ir" %% "scalanetwork" % "0.10.2"
```

If you are using Maven, add a dependency as described below:
```xml
<dependency>
  <groupId>kr.ac.kaist.ir</groupId>
  <artifactId>scalanetwork_${your.scala.version}</artifactId>
  <version>0.10.1</version>
</dependency>
```

## Simple Example
`Network.apply(Activation, Int*)` generates fully-connected network:

```scala
// Define 2 -> 4 -> 1 Layered, Fully connected network.
val net = Network(Sigmoid, 2, 4, 1)
// Define Manipulation Type. VectorType, AEType, RAEType, StandardRAEType, and URAEType.
val operation = new VectorType(
   corrupt = GaussianCorruption(variance = 0.1)
)
// Define Training Style. SingleThreadTrainStyle, MultiThreadTrainStyle, & DistBeliefTrainStyle
val style = new SingleThreadTrainStyle(
  net = net,
  algorithm = new StochasticGradientDescent(l2decay = 0.0001),
  make = operation,
  param = SimpleTrainingCriteria(miniBatch = 8))
// Define Trainer
val train = new Trainer(
  style = style,
  stops = StoppingCriteria(maxIter = 100000))
// Do Train
train.train(set, valid)
```

## Network Creation

To create network, you can choose one of the followings:

* Most simplest : Using sugar syntax, `Network.apply`

```scala
// Network(Activation, SizeOfLayer1, SizeOfLayer2, SizeOfLayer3, ...)
Network(Sigmoid, 2, 4, 1)
Network(HyperbolicTangent, 4, 10, 7)
Network(Rectifier, 30, 10, 5)
Network(Softplus, 100, 50, 30, 10, 1)
```

* If you want different activation functions for each layer,

```scala
val layer1 = new BasicLayer(10 -> 7, Sigmoid)
val layer2 = new SplitTensorLayer((3, 4) -> 2, Rectifier)
new BasicNetwork(Seq(layer1, layer2), 0.95)
```

Second argument of Basic Network indicates presence probability, 
i.e. 1 - (neuron drop-out probability for drop-out training). Default is 1.

* If you want single-layer AutoEncoder,

```scala
val layer = new ReconBasicLayer(10 -> 7, Sigmoid)
new AutoEncoder(layer, 0.95)
```

AutoEncoder only accepts `Reconstructable` type. Currently, `ReconBasicLayer` is only supported one. 
(Tensor layer version is planned)

* If you want to stack autoencoders,

```scala
val net1 = new AutoEncoder(...)
val net2 = new AutoEncoder(...)
new StackedAutoEncoder(Seq(net1, net2))
```

Note that StackedAutoEncoder does not get any presence probability.

## Training

### Algorithm & Training Criteria
Before choose Training Style, you must specify algorithm and training criteria.

```scala
/* Algorithms */
new StochasticGradientDescent(rate=0.8, l1decay=0.0, l2decay=0.0001, momentum=0.0001)
new AdaGrad(rate=0.6, l1decay=0.0, l2decay=0.0001)
new AdaDelta(l1decay=0.0, l2decay=0.0001, decay=0.95, epsilon=1e-6)
```
```scala
/* Training Criteria */
import scala.concurrent.duration._
SimpleTrainingCriteria(miniBatch=100, validationSize=20, negSamplingRatio=0)
DistBeliefCriteria(miniBatch=100, validationSize=20, negSamplingRatio=0, submitInterval=1.seconds,
  updateStep=2, fetchStep=10, numCores=1)
```

Validation size sets the number of elements used for validation phrase.

### Input Options
Also you can specify input operations or options.

```scala
/* Corruptions */
NoCorruption
DroppingCorruption(presence=0.95)
GaussianCorruption(mean=0, variance=0.1)
```
```scala
/* Objective Functions */
SquaredErr
CrossEntropyErr // Which is Logistic Err
```
```scala
/* Manipulation Type : Vector input, Vector output */
// General Neural Network type
new VectorType(corrupt, objective)
// General AutoEncoder type
new AEType(corrupt, objective)

/* Manipulation Type : Tree input, Null output (AutoEncoder) */
// Train network as RAE style. 
// Every internal node regarded as reconstruction its direct children (not all leaves).
new RAEType(corrupt, objective)
new StandardRAEType(corrupt, objective)
// Experimental: Train network as URAE style. 
// With same structure, network should reconstruct all leaves from root.
new URAEType(corrupt, objective)
```

### Training Style
You can choose the training style of the network.

```scala
/* Styles */
new SingleThreadTrainStyle(net, algorithm, mnpl:ManipulationType, param)
new MultiThreadTrainStyle(net, sparkContext, algorithm, mnpl:ManipulationType, param:DistBeliefCriteria)
new DistBeliefTrainStyle(net, sparkContext, algorithm, mnpl:ManipulationType, param:DistBeliefCriteria)
```

### Training
Training is done by `Trainer` class.

```scala
/* Stopping Criteria */
StoppingCriteria(maxIter = 100000, waitAfterUpdate=2,
  improveThreshold=0.95, lossThreshold=1e-4, validationFreq=1.0f)

/* Trainer */
new Trainer(style = style, stops = StoppingCriteria(), name = "Trainer")
```

* **waitAfterUpdate** indicates wating time from the improvement. If network output improved on 100-th iteration,
  the trainer waits until `Max(validationEpoch, 100 * patienceStep)`.
* **Improve Threshold** indicates bottom line for improvement. 
  To be regarded as improved, loss should be less than (best loss) * improveThreshold
* **Loss threshold** indicates maximum loss can be accepted.
* **Validation Frequency** sets the number of iterations between validations. (1 iteration does train all training examples)

Training is done by `train` method.

```scala
// If training and validation set are the same
trainer.train(Seq[(IN, OUT)])
trainer.train(Int => Seq[(IN, OUT)]) // With generator.

// If they are different
trainer.train(Seq[(IN, OUT)], Seq[(IN, OUT)])
trainer.train(Int => Seq[(IN, OUT)], Int => Seq[(IN, OUT)])

// If you are using RDD
trainer.train(RDD[(IN, OUT)])
trainer.train(RDD[(IN, OUT)], RDD[(IN, OUT)])
```

If you are using RDD, ScalaNetwork automatically caches your input sequence.

Also you can add negative examples, using `trainer.setNegativeTrainingReference()`
