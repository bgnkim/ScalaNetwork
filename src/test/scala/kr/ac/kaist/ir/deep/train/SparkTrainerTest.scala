package kr.ac.kaist.ir.deep.train

import breeze.linalg.DenseMatrix
import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.layer.SplitTensorLayer
import kr.ac.kaist.ir.deep.network.BasicNetwork
import org.apache.spark.{SparkConf, SparkContext}
import org.specs2.mutable.Specification

/**
 * Test for spark-based local training.
 */
class SparkTrainerTest extends Specification {

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

  "SparkTrainer(Local)" should {
    val layer = new SplitTensorLayer((2, 1) → 4, Sigmoid)
    val net = new BasicNetwork(Seq(layer))
    val conf = new SparkConf().setMaster("local[6]").setAppName("SparkTrainer Test")
    val sc = new SparkContext(conf)
    val style = new DistBeliefTrainStyle[ScalarMatrix, ScalarMatrix](
      net = net,
      sc = sc,
      algorithm = new StochasticGradientDescent(rate = 0.3, l2decay = 0.00001),
      param = DistBeliefCriteria(miniBatch = 16, fetchStep = 10, updateStep = 2, numCores = 6))
    val train = new Trainer(
      style = style,
      stops = StoppingCriteria(maxIter = 100000))

    "properly trained" in {
      train.train(set, valid) must be_<(0.4)
    }
  }
}
