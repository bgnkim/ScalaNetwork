package kr.ac.kaist.ir.deep

import breeze.linalg.sum
import breeze.numerics.{abs, pow}
import breeze.stats.distributions.Gaussian
import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.network.{AutoEncoder, Network}

import scala.annotation.tailrec

/**
 * Package for training
 *
 * Created by bydelta on 2015-01-02.
 */
package object trainer {
  /** Type of Corruption */
  type Corruption = ScalarMatrix ⇒ ScalarMatrix

  /**
   * Trait : Weight update
   */
  trait WeightUpdater extends ((Seq[ScalarMatrix], Seq[ScalarMatrix]) ⇒ Unit) with Serializable {
    /** Decay factor for L1-reg */
    protected val l1decay: Scalar
    /** Decay factor for L2-reg */
    protected val l2decay: Scalar

    /**
     * Compute weight-loss of given neuron objects
     * @param objs to be computed
     * @return weight loss of the set
     */
    def loss(objs: Seq[ScalarMatrix]) = {
      objs.foldLeft(0.0) {
        (err, obj) ⇒ {
          val l1loss = sum(abs(obj)) * l1decay
          val l2loss = sum(pow(obj, 2)) * l2decay
          err + l1loss + l2loss
        }
      }
    }
  }

  /**
   * Trait: Trainer
   */
  trait Trainer extends Serializable {
    /** Stopping Criteria */
    protected val stops: StoppingCriteria

    /**
     * Train given sequence, and validate with given sequence.
     * @param set to be trained
     * @return Training error (loss)
     */
    def train(set: Seq[(ScalarMatrix, ScalarMatrix)]): Scalar = trainWithValidation(set, set)

    /**
     * Train given sequence, and validate with another sequence.
     * @param set to be used for training
     * @param valid to be used for validation
     * @return Training error (loss)
     */
    def trainWithValidation(set: Seq[(ScalarMatrix, ScalarMatrix)], valid: Seq[(ScalarMatrix, ScalarMatrix)]): Scalar

    /**
     * Train given sequence. Randomly split "train" and "validation" set per each mini-batch 
     * @param set to be used for training
     * @param split to be used for validation
     * @return Training error (loss)
     */
    def trainWithSplit(set: Seq[(ScalarMatrix, ScalarMatrix)], split: Probability): Scalar

    /**
     * Train autoencoder with given sequence.
     * @param set to be used for input & reconstruction.
     * @return Training error (loss)
     */
    def trainAutoencoder(set: Seq[ScalarMatrix]): Scalar
  }

  /**
   * Input Corruption: Drop input as zero.
   *
   * If network uses drop-out training, we recommend that you do not use this.
   *
   * @param presence probability of not-dropped.
   */
  case class DroppingCorruption(presence: Double = 0.95) extends Corruption {
    /**
     * Do corruption
     * @param v1 to be corrupted
     * @return corrupted vector
     */
    override def apply(v1: ScalarMatrix): ScalarMatrix =
      v1 mapValues { x ⇒ if (Math.random() > presence) 0.0 else x}
  }

  /**
   * Input Corruption: Gaussian
   * @param mean of noise
   * @param variance of noise
   */
  case class GaussianCorruption(mean: Double = 0.0, variance: Double = 0.1) extends Corruption {
    /**
     * Gaussian Distribution
     */
    private val distro = Gaussian distribution(Double.box(mean), Double.box(variance))

    /**
     * Do corruption
     * @param v1 to be corrupted
     * @return corrupted vector
     */
    override def apply(v1: ScalarMatrix): ScalarMatrix =
      v1 mapValues { x ⇒ x + distro.draw()}
  }

  /**
   * Input Corruption: None
   */
  case object NoCorruption extends Corruption {

    /**
     * Identity.
     * @param v1 to be corrupted
     * @return the vector
     */
    override def apply(v1: ScalarMatrix) = v1
  }

  /**
   * Criteria: When to stop training
   * @param maxIter is maximum iteration count
   * @param patience is default patience count
   * @param patienceStep is default step for patience
   * @param improveThreshold is threshold for marked as "improved"
   * @param lossThreshold is maximum-tolerant loss value.
   * @param validationFreq is step count for validation
   */
  case class StoppingCriteria(maxIter: Int = 100000,
                              patience: Int = 5000,
                              patienceStep: Int = 2,
                              improveThreshold: Double = 0.995,
                              lossThreshold: Double = 0.0001,
                              validationFreq: Int = 10)

  /**
   * Criteria: How to train
   * @param batch is size of mini-batch
   * @param dropout is weight drop-out probability
   */
  case class TrainingCriteria(batch: Int = 10, dropout: Double = 0.001)


  /**
   * Algorithm: Stochastic Gradient Descent
   * @param rate for learning
   * @param l1decay for L1-reg
   * @param l2decay for L2-reg
   * @param momentum for momentum adaptive learning
   */
  class StochasticGradientDescent(rate: Double = 0.03,
                                  protected override val l1decay: Double = 0.0000,
                                  protected override val l2decay: Double = 0.0001,
                                  momentum: Double = 0.0001)
    extends WeightUpdater {
    /** Last update */
    private var lastDelta = Seq[ScalarMatrix]()

    /**
     * Run the algorithm
     * @param delta matrices of accumulation
     * @param weight matrices of current
     */
    override def apply(delta: Seq[ScalarMatrix], weight: Seq[ScalarMatrix]): Unit = {
      lastDelta = delta.indices map {
        id ⇒ {
          val w: ScalarMatrix = weight(id)
          val deltaW: ScalarMatrix = delta(id)

          val deltaL1: ScalarMatrix = w mapValues { x ⇒ if (x > 0) l1decay else if (x < 0) -l1decay else 0.0}
          val deltaL2: ScalarMatrix = w :* (l2decay * 2)
          val deltaL: ScalarMatrix = deltaL1 + deltaL2
          val deltaLoss: ScalarMatrix = deltaW + deltaL
          val adapted: ScalarMatrix = deltaLoss :* rate

          val dw: ScalarMatrix = if (lastDelta.nonEmpty) {
            val moment: ScalarMatrix = lastDelta(id) * momentum
            moment - adapted
          } else {
            -adapted
          }

          w += dw
          deltaW := 0.0
          dw
        }
      }
    }
  }

  /**
   * Algorithm: AdaGrad
   * @param rate for learning
   * @param l1decay for L1-reg
   * @param l2decay for L2-reg
   */
  class AdaGrad(rate: Double = 0.6,
                protected override val l1decay: Double = 0.0000,
                protected override val l2decay: Double = 0.0001)
    extends WeightUpdater {
    /** History of update */
    private var history = Seq[ScalarMatrix]()

    /**
     * Run the algorithm
     * @param delta matrices of accumulation
     * @param weight matrices of current
     */
    override def apply(delta: Seq[ScalarMatrix], weight: Seq[ScalarMatrix]): Unit = {
      if (history.isEmpty) {
        history = delta map {
          matx ⇒ ScalarMatrix $0(matx.rows, matx.cols)
        }
      }

      delta.indices map {
        id ⇒ {
          val w = weight(id)
          val deltaW = delta(id)

          val deltaL1: ScalarMatrix = w mapValues { x ⇒ if (x > 0) l1decay else if (x < 0) -l1decay else 0.0}
          val deltaL2: ScalarMatrix = w * (l2decay * 2)
          val deltaL: ScalarMatrix = deltaL1 + deltaL2
          val deltaLoss: ScalarMatrix = deltaW + deltaL

          history(id) :+= pow(deltaLoss, 2)

          val adapted = history(id) mapValues { x ⇒ rate / Math.sqrt(x)}
          val d: ScalarMatrix = deltaLoss :* adapted

          w -= d
        }
      }
    }
  }

  /**
   * Algorithm: AdaDelta
   * @param l1decay for L1-reg
   * @param l2decay for L2-reg
   * @param decay for AdaDelta history decay factor
   * @param epsilon for AdaDelta base factor
   */
  class AdaDelta(protected override val l1decay: Double = 0.01,
                 protected override val l2decay: Double = 0.01,
                 private val decay: Double = 0.95,
                 private val epsilon: Double = 1e-6)
    extends WeightUpdater {
    /** History of updates */
    private var grad = Seq[ScalarMatrix]()
    private var delL2 = Seq[ScalarMatrix]()

    /**
     * Run the algorithm
     * @param delta matrices of accumulation
     * @param weight matrices of current
     */
    override def apply(delta: Seq[ScalarMatrix], weight: Seq[ScalarMatrix]): Unit = {
      if (grad.isEmpty) {
        grad = delta map {
          matx ⇒ ScalarMatrix $0(matx.rows, matx.cols)
        }
        delL2 = grad map {
          _.copy
        }
      }

      delta.indices map {
        id ⇒ {
          val w = weight(id)
          val deltaW = delta(id)

          val deltaL1: ScalarMatrix = w mapValues { x ⇒ if (x > 0) l1decay else if (x < 0) -l1decay else 0.0}
          val deltaL2: ScalarMatrix = w * (l2decay * 2)
          val deltaL: ScalarMatrix = deltaL1 + deltaL2
          val deltaLoss: ScalarMatrix = deltaW + deltaL

          val avgGradL2 = grad(id)
          val avgDeltaL2 = delL2(id)

          avgGradL2 :*= decay
          avgGradL2 += (pow(deltaLoss, 2) :* (1.0 - decay))

          val adjusted = avgGradL2 mapPairs { (key, grad2) ⇒ Math.sqrt(avgDeltaL2(key) + epsilon) / Math.sqrt(grad2 + epsilon)}
          val d: ScalarMatrix = deltaLoss :* adjusted

          w -= d

          avgDeltaL2 :*= decay
          avgDeltaL2 += (pow(d, 2) :* (1.0 - decay))
        }
      }
    }
  }

  /**
   * Trainer : Stochastic-Style
   * @param net to be trained
   * @param algorithm to be applied
   * @param error to compute loss
   * @param corrupt to corrupt input
   * @param param of training criteria
   * @param stops of stopping criteria
   */
  class BasicTrainer(private val net: Network,
                     private val algorithm: WeightUpdater,
                     private val error: Objective = SquaredErr,
                     private val corrupt: Corruption = NoCorruption,
                     private val param: TrainingCriteria = TrainingCriteria(),
                     protected override val stops: StoppingCriteria = StoppingCriteria(),
                     private val debug: Boolean = false)
    extends Trainer {
    /** Stochastic Generator Base */
    protected val randomSetGenerator = (set: Seq[(ScalarMatrix, ScalarMatrix)]) ⇒ () ⇒ set((Math.random() * set.size).toInt)
    /** Training Set */
    protected var trainingSet: () ⇒ (ScalarMatrix, ScalarMatrix) = null
    /** Validation Set */
    protected var testSet: () ⇒ Seq[(ScalarMatrix, ScalarMatrix)] = null
    /** Best Parameter History */
    protected var bestParam: Seq[ScalarMatrix] = null
    protected var bestIter: Int = 0

    /**
     * Train given sequence, and validate with another sequence.
     * @param set to be used for training
     * @param valid to be used for validation
     * @return Training error (loss)
     */
    def trainWithValidation(set: Seq[(ScalarMatrix, ScalarMatrix)], valid: Seq[(ScalarMatrix, ScalarMatrix)]) = {
      trainingSet = randomSetGenerator(set)
      testSet = () ⇒ valid

      saveParams()
      val err = trainBatch()
      restoreParams()
      printValidation()

      err
    }

    /**
     * Train given sequence. Randomly split "train" and "validation" set per each mini-batch 
     * @param set to be used for training
     * @param split to be used for validation
     * @return Training error (loss)
     */
    override def trainWithSplit(set: Seq[(ScalarMatrix, ScalarMatrix)], split: Probability): Scalar = {
      testSet = () ⇒ {
        val sets = set groupBy { _ ⇒ Math.random() > split}
        trainingSet = randomSetGenerator(sets(true))
        sets(false)
      }

      // initialize for first batch.
      testSet()

      saveParams()
      val err = trainBatch()
      restoreParams()
      printValidation()

      err
    }

    /**
     * Train autoencoder with given sequence.
     * @param set to be used for input & reconstruction.
     * @return Training error (loss)
     */
    override def trainAutoencoder(set: Seq[ScalarMatrix]): Scalar = {
      val pairs = set map { x ⇒ x → x}
      trainingSet = randomSetGenerator(pairs)
      testSet = () ⇒ pairs

      saveParams()
      val err = trainBatch(isAutoEncoder = true)
      restoreParams()
      printValidation(isAutoEncoder = true)

      err
    }

    /**
     * Store best parameters
     */
    protected final def saveParams() = {
      bestParam = net.W map {
        _.copy
      }
    }

    /**
     * Restore best parameters
     */
    protected final def restoreParams() = {
      bestParam.indices foreach {
        id ⇒ net.W(id) := bestParam(id)
      }
    }

    /**
     * Calculate validation error
     * @param isAutoEncoder true if it is autoencoder training
     * @return validation error
     */
    protected def validationError(isAutoEncoder: Boolean = false) = {
      val t = testSet()
      t.foldLeft(0.0) { (err, item) ⇒ {
        val in = item._1
        val out = if (isAutoEncoder) in >>: net else net(in)
        if (debug) {
          println(s"IN ${in.mkString} : EXP ${item._2.mkString} = OUT : ${out.mkString}")
        }
        err + error(item._2, out)
      }
      } / t.size
    }

    /**
     * Print validation result
     * @param isAutoEncoder true if it is autoencoder training
     */
    protected def printValidation(isAutoEncoder: Boolean = false) = {
      println(s"BEST ITERATION ${bestIter} : W = ${net.W map (_.mkString) mkString " | "}")

      val t = testSet()
      t map {
        item ⇒ {
          val in = item._1
          val out = if (isAutoEncoder) net.asInstanceOf[AutoEncoder].reconstruct(in) else net(in)
          println(s"IN ${in.mkString} : EXP ${item._2.mkString} = OUT : ${out.mkString}")
        }
      }
    }


    /**
     * Tail Recursive : Train each batch
     * @param iter indicates current iteration
     * @param prevloss indicates previous loss
     * @param patience indicates current patience
     * @param isAutoEncoder is a flag for autoencoder.
     * @return Total Loss when train is finished
     */
    @tailrec
    protected final def trainBatch(iter: Int = 0,
                                   prevloss: Double = Double.MaxValue,
                                   patience: Int = stops.patience,
                                   isAutoEncoder: Boolean = false): Scalar = {
      (0 until param.batch) foreach { _ ⇒ {
        val pair = trainingSet()
        val out = corrupt(pair._1) >>: net
        val err: ScalarMatrix = error.derivative(pair._2, out) / param.batch.toDouble
        net ! err
      }
      }
      algorithm(net.dW, net.W)

      var nPatience = patience

      val nLoss = if ((iter + 1) % stops.validationFreq == 0) {
        if (debug) {
          println(s"ITERATION $iter : W = ${net.W map (_.mkString) mkString " | "}")
        }
        val train = validationError(isAutoEncoder)
        val weight = algorithm loss net.W
        if (train + weight < prevloss * stops.improveThreshold) {
          nPatience = Math.max(patience, iter * stops.patienceStep)
          bestIter = iter
          saveParams()
          println(f"Iteration $iter%6d, Validation = $train%.5f, WeightLoss = $weight%.5f")
          train + weight
        } else {
          prevloss
        }
      } else {
        prevloss
      }

      if (iter < param.batch * stops.maxIter && nPatience > iter && nLoss > stops.lossThreshold) {
        trainBatch(iter + 1, nLoss, nPatience, isAutoEncoder)
      } else {
        println(f"Finished $iter%6d, Error = $nLoss%.5f")
        nLoss
      }
    }
  }

}
