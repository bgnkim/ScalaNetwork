package kr.ac.kaist.ir.deep.network

import breeze.linalg.DenseMatrix
import kr.ac.kaist.ir.deep.function.{CrossEntropyErr, Scalar, Sigmoid, SquaredErr}
import kr.ac.kaist.ir.deep.layer.{Layer, ReconBasicLayer}
import kr.ac.kaist.ir.deep.trainer._
import org.specs2.mutable.Specification

/**
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
      layer.W(1).rows must_== 3
      layer.W(1).cols must_== 4
    }

    "have encoder bias" in {
      layer.W(2).rows must_== 3
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
      x ⇒ (0 to 1) flatMap {
        y ⇒ (0 until 100) map {
          _ ⇒ {
            val m = DenseMatrix.create[Scalar](4, 1, Array(x, y, y, x))
            (m, m)
          }
        }
      }
    }
    val valid = (0 to 1) flatMap {
      x ⇒ (0 to 1) map {
        y ⇒ {
          val m = DenseMatrix.create[Scalar](4, 1, Array(x, y, y, x))
          (m, m)
        }
      }
    }

    val encoder = new AutoEncoder(layer)
    val trainer = new StochasticTrainer(encoder, new GradientDescent(), error = CrossEntropyErr)

    "properly trained" in {
      trainer.trainWithValidation(set, valid) must be_<(0.4)
    }

    examplesBlock {
      (0 until valid.size) foreach {
        i ⇒ {
          val in = valid(i)._1
          val out = valid(i)._1 >>: encoder
          s"Example $i, ${in.toArray.toSeq} ? ${out.toArray.toSeq}" in {
            SquaredErr(in, out) must be_<(0.02)
          }
        }
      }
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
    val train = new StochasticTrainer(net, new GradientDescent(), param = TrainingCriteria(batch = 1))

    "properly trained" in {
      train.trainWithValidation(set, valid) must be_<(0.4)
    }

    examplesBlock {
      (0 until valid.size) foreach {
        i ⇒ {
          val in = valid(i)._1
          val out = valid(i)._1 >>: net
          s"Example $i, ${in.toArray.toSeq} ? ${out.toArray.toSeq}" in {
            SquaredErr(valid(i)._2, out) must be_<(0.02)
          }
        }
      }
    }
  }
}
