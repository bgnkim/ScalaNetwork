package kr.ac.kaist.ir.deep.network

import breeze.linalg.DenseMatrix
import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.layer.{Layer, ReconBasicLayer}
import kr.ac.kaist.ir.deep.trainer._
import org.specs2.mutable.Specification

/**
 * Network Tester Package.
 *
 * Created by bydelta on 2015-01-06.
 */
class NetworkTester extends Specification {
  val layer = new ReconBasicLayer(4 → 5, Sigmoid)
  "ReconBasicLayer" should {
    "have 3 weights" in {
      layer.W must have size 3
      layer.dW must have size 3
    }

    "have Reconstruction bias" in {
      layer.W(0).rows must_== 4
      layer.W(0).cols must_== 1
    }

    "have Weight" in {
      layer.W(1).rows must_== 5
      layer.W(1).cols must_== 4
    }

    "have encoder bias" in {
      layer.W(2).rows must_== 5
      layer.W(2).cols must_== 1
    }

    "can be reconstructed" in {
      val json = layer.toJSON
      val l2 = Layer(json)
      l2 must haveClass[ReconBasicLayer]
    }
  }

  "Non-drop encoder training" should {
    val set = (0 to 1) flatMap {
      x ⇒ (0 to 1) map {
        y ⇒ {
          val m = DenseMatrix.create[Scalar](4, 1, Array(x, y, y, x))
          m
        }
      }
    }

    val encoder = new AutoEncoder(layer, 0.9995)
    val trainer = new BasicTrainer(net = encoder,
      algorithm = new StochasticGradientDescent(rate = 0.8, l2decay = 0.0001),
      param = SimpleTrainingCriteria(miniBatch = 8),
      stops = StoppingCriteria(maxIter = 100000))

    "properly trained" in {
      trainer.trainAutoencoder(set) must be_<(0.3)
    }
  }

  "XOR network" should {

    val set = (0 to 1) flatMap {
      x ⇒ (0 to 1) flatMap {
        y ⇒ (0 until 100) map {
          _ ⇒ {
            val m = DenseMatrix.create[Scalar](2, 1, Array(x, y))
            val n = DenseMatrix.create[Scalar](1, 1, Array(if (x == y) 0.0 else 1.0))
            (m, n)
          }
        }
      }
    }

    val valid = (0 to 1) flatMap {
      x ⇒ (0 to 1) map {
        y ⇒ {
          val m = DenseMatrix.create[Scalar](2, 1, Array(x, y))
          val n = DenseMatrix.create[Scalar](1, 1, Array(if (x == y) 0.0 else 1.0))
          (m, n)
        }
      }
    }

    val net = Network(Sigmoid, 2, 4, 1)
    val train = new BasicTrainer(net = net,
      algorithm = new AdaDelta(l2decay = 0.0001, decay = 0.95, epsilon = 1e-4),
      //algorithm = new AdaGrad(l2decay = 0.0001),
      //algorithm = GradientDescent(rate = 0.8, l2decay = 0.0001),
      corrupt = GaussianCorruption(variance = 0.1),
      param = SimpleTrainingCriteria(miniBatch = 8),
      stops = StoppingCriteria(maxIter = 100000))

    "properly trained" in {
      train.train(set, valid) must be_<(0.3)
    }
  }
}
