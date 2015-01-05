package kr.ac.kaist.ir.deep.network

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.trainer._
import org.specs2.mutable.Specification

/**
 * Created by bydelta on 2015-01-04.
 */
class BasicNetTest extends Specification{
  val network = Network(Sigmoid, 2, 1)
  "BasicNetwork" should {
    "have right size" in {
      network.neurons must haveLength(0)
      network.synapses must haveLength(2)
      network.outNeurons must haveLength(1)
    }
  }

  var trainset = List[(NeuronVector, NeuronVector)]()
  for (i1 ← 0 to 1; i2 ← 0 to 1) {
    val x = Map(1 → i1.toDouble, 2 → i2.toDouble) → Map(6 → (if(i1 == i2) 0.0 else 1.0))
    trainset = x :: trainset
  }

  "SGD" should {
    val net = Network(Sigmoid, 2, 3, 1)
    val trainer = new StochasticTrainer(net, algorithm = new GradientDescent(l1decay = 0.0, l2decay = 0.0, momentum = 0.0), error = CrossEntropyErr, stops = StoppingCriteria(), param = TrainingCriteria(dropout = 0.0))
    "properly trained" in {
      trainer.train(trainset) must be_<(0.3)
      println(net.toJSON) must haveClass[Unit]
    }

    examplesBlock {
      for (i ← 0 until 4) {
        val item = trainset(i)
        val out = net(item._1)
        f"example $i : $item -> $out" in {
          SquaredErr(item._2, out) must be_<(0.04)
        }
      }
    }
  }

/*  "AdaGrad" should {
    val net = Network(Sigmoid, 2, 3, 1)
    val trainer = new StochasticTrainer(net, algorithm = new AdaGrad(), error = SquaredErr)
    "properly trained" in {
      trainer.train(trainset) must be_<(0.3)
    }

    examplesBlock {
      for (i ← 0 until 4) {
        val item = trainset(i)
        val out = net(item._1)
        f"example $i : $item -> $out" in {
          SquaredErr(item._2, out) must be_<(0.04)
        }
      }
    }
  }

  "AdaDelta" should {
    val net = Network(Sigmoid, 2, 3, 1)
    val trainer = new StochasticTrainer(net, algorithm = new AdaDelta(), error = SquaredErr)
    "properly trained" in {
      trainer.train(trainset) must be_<(0.3)
    }

    examplesBlock {
      for (i ← 0 until 4) {
        val item = trainset(i)
        val out = net(item._1)
        f"example $i : $item -> $out" in {
          SquaredErr(item._2, out) must be_<(0.04)
        }
      }
    }
  } */
}
