package kr.ac.kaist.ir.deep.network

import breeze.linalg.DenseMatrix
import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.layer.{Layer, ReconBasicLayer}
import kr.ac.kaist.ir.deep.train._
import org.apache.log4j.{ConsoleAppender, Level, Logger, PatternLayout}
import org.specs2.mutable.Specification

/**
 * Network Tester Package.
 *
 * Created by bydelta on 2015-01-06.
 */
class NetworkTester extends Specification {
  val console = new ConsoleAppender()
  val PATTERN = "%d %p %C{1} %m%n"
  console.setLayout(new PatternLayout(PATTERN))
  console.setThreshold(Level.INFO)
  console.activateOptions()
  val logger = Logger.getRootLogger
  logger.addAppender(console)

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
          m → null
        }
      }
    }

    val encoder = new AutoEncoder(layer, 0.9995f)
    val style = new SingleThreadTrainStyle(
      net = encoder,
      make = new AEType(),
      algorithm = new StochasticGradientDescent(rate = 0.8f, l2decay = 0.0001f),
      param = SimpleTrainingCriteria(miniBatch = 8))
    val trainer = new Trainer(style = style,
      stops = StoppingCriteria(maxIter = 100000))

    "properly trained" in {
      trainer.train(set, set) must be_<(0.3f)
    }
  }

  "XOR network" should {

    val set = (0 to 1) flatMap {
      x ⇒ (0 to 1) flatMap {
        y ⇒ (0 until 100) map {
          _ ⇒ {
            val m = DenseMatrix.create[Scalar](2, 1, Array(x, y))
            val n = DenseMatrix.create[Scalar](1, 1, Array(if (x == y) 0.0f else 1.0f))
            (m, n)
          }
        }
      }
    }

    val valid = (0 to 1) flatMap {
      x ⇒ (0 to 1) map {
        y ⇒ {
          val m = DenseMatrix.create[Scalar](2, 1, Array(x, y))
          val n = DenseMatrix.create[Scalar](1, 1, Array(if (x == y) 0.0f else 1.0f))
          (m, n)
        }
      }
    }

    val net = Network(Sigmoid, 2, 4, 1)
    val operation = new VectorType(
      corrupt = GaussianCorruption(variance = 0.1)
    )
    val style = new SingleThreadTrainStyle(
      net = net,
      algorithm = new AdaDelta(decay = 0.9f, l2decay = 0.0f),
      make = operation,
      //algorithm = new AdaGrad(l2decay = 0.0001),
      //algorithm = GradientDescent(rate = 0.8, l2decay = 0.0001),
      param = SimpleTrainingCriteria(miniBatch = 8))
    val train = new Trainer(
      style = style,
      stops = StoppingCriteria(maxIter = 100000))

    "properly trained" in {
      train.train(set, valid) must be_<(0.3f)
    }
  }
}
