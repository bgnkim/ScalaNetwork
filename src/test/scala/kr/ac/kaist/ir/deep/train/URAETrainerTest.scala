package kr.ac.kaist.ir.deep.train

import breeze.linalg.DenseMatrix
import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.layer.ReconBasicLayer
import kr.ac.kaist.ir.deep.network.AutoEncoder
import kr.ac.kaist.ir.deep.rec.{DAG, InternalNode, TerminalNode}
import org.apache.log4j.{ConsoleAppender, Level, Logger, PatternLayout}
import org.specs2.mutable.Specification

/**
 * Test for spark-based local training.
 */
class URAETrainerTest extends Specification {
  val console = new ConsoleAppender()
  val PATTERN = "%d %p %C{1} %m%n"
  console.setLayout(new PatternLayout(PATTERN))
  console.setThreshold(Level.INFO)
  console.activateOptions()
  Logger.getLogger("kr.ac").addAppender(console)


  val set = (0 to 1) flatMap {
    x ⇒ (0 to 1) flatMap {
      y ⇒ (0 to 1) flatMap {
        z ⇒ (0 until 100) map {
          _ ⇒ {
            val nodeX = new TerminalNode(DenseMatrix.create[Scalar](2, 1, Array(x, y)))
            val nodeY = new TerminalNode(DenseMatrix.create[Scalar](2, 1, Array(y, z)))
            val nodeZ = new TerminalNode(DenseMatrix.create[Scalar](2, 1, Array(z, x)))
            val int1 = new InternalNode(Seq(nodeX, nodeY))
            val int2 = new InternalNode(Seq(int1, nodeZ))
            (new DAG(Seq(int2)), null)
          }
        }
      }
    }
  }

  val valid = (0 to 1) flatMap {
    x ⇒ (0 to 1) flatMap {
      y ⇒ (0 to 1) map {
        z ⇒ {
          val nodeX = new TerminalNode(DenseMatrix.create[Scalar](2, 1, Array(x, y)))
          val nodeY = new TerminalNode(DenseMatrix.create[Scalar](2, 1, Array(y, z)))
          val nodeZ = new TerminalNode(DenseMatrix.create[Scalar](2, 1, Array(z, x)))
          val int1 = new InternalNode(Seq(nodeX, nodeY))
          val int2 = new InternalNode(Seq(int1, nodeZ))
          (new DAG(Seq(int2)), null)
        }
      }
    }
  }

  "URAETrainer(Local)" should {
    val layer = new ReconBasicLayer(4 → 2, Sigmoid)
    val net = new AutoEncoder(layer)
    val style = new SingleThreadTrainStyle(
      net = net,
      algorithm = new StochasticGradientDescent(rate = 0.7, l2decay = 0.00001),
      make = new URAEType(),
      param = SimpleTrainingCriteria(miniBatch = 16, validationSize = 10))
    val train = new Trainer(
      style = style,
      stops = StoppingCriteria(maxIter = 100000))

    "properly trained" in {
      train.train(set, valid) must be_<(0.4)
    }
  }
}
