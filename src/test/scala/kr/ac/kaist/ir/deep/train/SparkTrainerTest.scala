package kr.ac.kaist.ir.deep.train

import breeze.linalg.DenseMatrix
import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.layer.SplitTensorLayer
import kr.ac.kaist.ir.deep.network.BasicNetwork
import org.apache.log4j.{ConsoleAppender, Level, Logger, PatternLayout}
import org.apache.spark.{SparkConf, SparkContext}
import org.specs2.mutable.Specification

/**
 * Test for spark-based local training.
 */
class SparkTrainerTest extends Specification {
  val console = new ConsoleAppender()
  val PATTERN = "%d %p %C{1} %m%n"
  console.setLayout(new PatternLayout(PATTERN))
  console.setThreshold(Level.INFO)
  console.activateOptions()
  Logger.getLogger("kr.ac").addAppender(console)
  Logger.getLogger("org.spark").setLevel(Level.FATAL)


  val set = (0 to 1) flatMap {
    x ⇒ (0 to 1) flatMap {
      y ⇒ (0 to 1) flatMap {
        z ⇒ (0 until 100) map {
          _ ⇒ {
            val m = DenseMatrix.create[Scalar](3, 1, Array(x, y, z))
            val n = DenseMatrix.create[Scalar](4, 1, Array((x + y + 1 - z) / 3.0f, (x + 1 - y + z) / 3.0f, (z + 1 - x + y) / 3.0f, (x + y + z) / 3.0f))
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
          val n = DenseMatrix.create[Scalar](4, 1, Array((x + y + 1 - z) / 3.0f, (x + 1 - y + z) / 3.0f, (z + 1 - x + y) / 3.0f, (x + y + z) / 3.0f))
          (m, n)
        }
      }
    }
  }

  "SparkTrainer(Local)" should {
    val layer = new SplitTensorLayer((2, 1) → 4, Sigmoid)
    val net = new BasicNetwork(IndexedSeq(layer))
    val conf = new SparkConf().setMaster("local[6]").setAppName("SparkTrainer Test")
    val sc = new SparkContext(conf)
    val style = new DistBeliefTrainStyle[ScalarMatrix, ScalarMatrix](
      net = net,
      sc = sc,
      algorithm = new StochasticGradientDescent(rate = 0.7f, l2decay = 0.00001f),
      param = DistBeliefCriteria(miniBatch = 16, fetchStep = 10, updateStep = 2, numCores = 6, validationSize = 10))
    val train = new Trainer(
      style = style,
      stops = StoppingCriteria(maxIter = 100000))

    "properly trained" in {
      train.train(set, valid) must be_<(0.4f)
    }
  }
}
