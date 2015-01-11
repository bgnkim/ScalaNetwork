package kr.ac.kaist.ir.deep.network

import breeze.linalg.DenseMatrix
import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.layer.{Layer, Rank3TensorLayer}
import kr.ac.kaist.ir.deep.trainer._
import org.specs2.mutable.Specification

/**
 * Network Tester Package.
 *
 * Created by bydelta on 2015-01-06.
 */
class TensorNetworkTester extends Specification {
  val layer = new Rank3TensorLayer((2, 1) → 4, Sigmoid)
  "Rank3TensorLayer" should {
    "have 3 weights" in {
      layer.W must have size 9
      layer.dW must have size 9
    }

    "have bias" in {
      layer.W(0).rows must_== 4
      layer.W(0).cols must_== 1
    }

    "have Linear weight" in {
      layer.W(1).rows must_== 1
      layer.W(1).cols must_== 3
      layer.W(2).rows must_== 1
      layer.W(2).cols must_== 3
      layer.W(3).rows must_== 1
      layer.W(3).cols must_== 3
      layer.W(4).rows must_== 1
      layer.W(4).cols must_== 3
    }

    "have Quadratic weight" in {
      layer.W(5).rows must_== 2
      layer.W(5).cols must_== 1
      layer.W(6).rows must_== 2
      layer.W(6).cols must_== 1
      layer.W(7).rows must_== 2
      layer.W(7).cols must_== 1
      layer.W(8).rows must_== 2
      layer.W(8).cols must_== 1
    }

    "can be reconstructed" in {
      val json = layer.toJSON
      val l2 = Layer(json)
      l2 must haveClass[Rank3TensorLayer]
    }
  }

  "Boolean network" should {
    val set = (0 to 1) flatMap {
      x ⇒ (0 to 1) flatMap {
        y ⇒ (0 to 1) flatMap {
          z ⇒ (0 until 100) map {
            _ ⇒ {
              val m = DenseMatrix.create[Scalar](3, 1, Array(x, y, z))
              val n = DenseMatrix.create[Scalar](4, 1, Array((x + y + 1 - z) / 3.0, (x + 1 - y + z) / 3.0, (z + 1 - x + y) / 3.0, (x + y + z) / 3.0))
              (m, n)
            }
          }
        }
      }
    }

    val valid = (0 to 1) flatMap {
      x ⇒ (0 to 1) flatMap {
        y ⇒ (0 to 1) map {
          z ⇒ {
            val m = DenseMatrix.create[Scalar](3, 1, Array(x, y, z))
            val n = DenseMatrix.create[Scalar](4, 1, Array((x + y + 1 - z) / 3.0, (x + 1 - y + z) / 3.0, (z + 1 - x + y) / 3.0, (x + y + z) / 3.0))
            (m, n)
          }
        }
      }
    }

    val net = new BasicNetwork(Seq(layer))
    val train = new BasicTrainer(net = net,
      algorithm = new AdaDelta(l2decay = 0.001, l1decay = 0.0), //GradientDescent(rate = 0.8, l2decay = 0.0001),
      param = SimpleTrainingCriteria(miniBatch = 8),
      stops = StoppingCriteria(maxIter = 100000))

    "properly trained" in {
      train.train(set, valid) must be_<(0.4)
    }
  }
}
