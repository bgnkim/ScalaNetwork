package kr.ac.kaist.ir.deep

import breeze.linalg.sum
import breeze.numerics.{abs, pow}
import breeze.stats.distributions.Gaussian
import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.network.Network

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
  }

  /**
   * Input Corruption: Drop input as zero.
   *
   * If network uses drop-out training, we recommend that you do not use this.
   *
   * @param dropRate probability
   */
  case class DroppingCorruption(dropRate: Double = 0.01) extends Corruption {
    /**
     * Do corruption
     * @param v1 to be corrupted
     * @return corrupted vector
     */
    override def apply(v1: ScalarMatrix): ScalarMatrix =
      v1 mapValues { x ⇒ if (Math.random() < dropRate) 0.0 else x}
  }

  /**
   * Input Corruption: Gaussian
   * @param mean of noise
   * @param variance of noise
   */
  case class GaussianCorruption(mean: Double = 0.0, variance: Double = 1.0) extends Corruption {
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
   * @param validationFreq is step count for validation
   */
  case class StoppingCriteria(maxIter: Int = 10000, patience: Int = 5000, patienceStep: Int = 2, improveThreshold: Double = 0.995, validationFreq: Int = 10)

  /**
   * Criteria: How to train
   * @param batch is size of mini-batch
   * @param dropout is weight drop-out probability
   */
  case class TrainingCriteria(batch: Int = 10, dropout: Double = 0.001)


  /**
   * Algorithm: Gradient Descent
   * @param rate for learning
   * @param l1decay for L1-reg
   * @param l2decay for L2-reg
   * @param momentum for momentum adaptive learning
   */
  class GradientDescent(rate: Double = 0.03,
                        protected override val l1decay: Double = 0.000,
                        protected override val l2decay: Double = 0.000,
                        momentum: Double = 0.000)
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
                protected override val l1decay: Double = 0.001,
                protected override val l2decay: Double = 0.001)
    extends WeightUpdater {
    /** History of update per each weight */
    private var history: Seq[Scalar] = Seq()
    /** Constant function */
    private val one = (_: Any) ⇒ 1.0

    /**
     * Run the algorithm
     * @param delta matrices of accumulation
     * @param weight matrices of current
     */
    override def apply(delta: Seq[ScalarMatrix], weight: Seq[ScalarMatrix]): Unit = {
      val grad = delta.indices map {
        id ⇒ {
          val w = weight(id)
          val deltaW = delta(id)

          val deltaL1: ScalarMatrix = w mapValues { x ⇒ if (x > 0) l1decay else if (x < 0) -l1decay else 0.0}
          val deltaL2: ScalarMatrix = w * (l2decay * 2)
          val deltaL: ScalarMatrix = deltaL1 + deltaL2
          val deltaLoss: ScalarMatrix = deltaW + deltaL
          deltaLoss
        }
      }

      val adjusted = grad.indices map { id ⇒ rate / history.applyOrElse(id, one)}
      history = grad.indices map { id ⇒ history.applyOrElse(id, one) + sum(pow(grad(id), 2))}

      delta.indices foreach {
        id ⇒ {
          val dw: ScalarMatrix = grad(id) :* adjusted(id)

          weight(id) -= dw
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
  class AdaDelta(protected override val l1decay: Double = 0.001,
                 protected override val l2decay: Double = 0.001,
                 private val decay: Double = 0.95,
                 private val epsilon: Double = 1e-6)
    extends WeightUpdater {
    /** History of Delta */
    private var avgDeltaL2 = Seq[Scalar]()
    /** History of Gradient */
    private var avgGradL2 = Seq[Scalar]()
    /** Constant function */
    private val zero = (_: Any) ⇒ 0.0
    private val meanSq = (x: ScalarMatrix) ⇒ sum(pow(x, 2)) / x.size

    /**
     * Run the algorithm
     * @param delta matrices of accumulation
     * @param weight matrices of current
     */
    override def apply(delta: Seq[ScalarMatrix], weight: Seq[ScalarMatrix]): Unit = {
      val grad = delta.indices map {
        id ⇒ {
          val w = weight(id)
          val deltaW = delta(id)

          val deltaL1: ScalarMatrix = w mapValues { x ⇒ if (x > 0) l1decay else if (x < 0) -l1decay else 0.0}
          val deltaL2: ScalarMatrix = w * (l2decay * 2)
          val deltaL: ScalarMatrix = deltaL1 + deltaL2
          val deltaLoss: ScalarMatrix = deltaW + deltaL
          deltaLoss
        }
      }

      avgGradL2 = grad.indices map { id ⇒ decay * avgGradL2.applyOrElse(id, zero) + (1.0 - decay) * meanSq(grad(id))}

      val adjusted = grad.indices map { id ⇒ Math.sqrt(avgDeltaL2.applyOrElse(id, zero) + epsilon) / Math.sqrt(avgGradL2(id) + epsilon)}

      avgDeltaL2 = delta.indices map {
        id ⇒ {
          val d: ScalarMatrix = grad(id) :* adjusted(id)

          weight(id) -= d
          decay * avgDeltaL2.applyOrElse(id, zero) + (1.0 - decay) * meanSq(d)
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
  class StochasticTrainer(private val net: Network,
                          private val algorithm: WeightUpdater,
                          private val error: Objective = SquaredErr,
                          private val corrupt: Corruption = NoCorruption,
                          private val param: TrainingCriteria = TrainingCriteria(),
                          protected override val stops: StoppingCriteria = StoppingCriteria())
    extends Trainer {
    /** Stochastic Generator Base */
    protected val randomSetGenerator = (set: Seq[(ScalarMatrix, ScalarMatrix)]) ⇒ () ⇒ set((Math.random() * set.size).toInt)
    /** Generator */
    protected var generate: () ⇒ (ScalarMatrix, ScalarMatrix) = null
    /** Validation Set */
    protected var validation: Seq[(ScalarMatrix, ScalarMatrix)] = null
    /** Best Parameter History */
    protected var bestParam: Seq[ScalarMatrix] = null

    /**
     * Train given sequence, and validate with another sequence.
     * @param set to be used for training
     * @param valid to be used for validation
     * @return Training error (loss)
     */
    def trainWithValidation(set: Seq[(ScalarMatrix, ScalarMatrix)], valid: Seq[(ScalarMatrix, ScalarMatrix)]) = {
      generate = randomSetGenerator(set)
      validation = valid

      saveParams()
      val err = trainBatch()
      restoreParams()
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
     * Tail Recursive : Train each batch
     * @param iter indicates current iteration
     * @param prevloss indicates previous loss
     * @param patience indicates current patience
     * @return Total Loss when train is finished
     */
    @tailrec
    protected final def trainBatch(iter: Int = 0, prevloss: Double = Double.MaxValue, patience: Int = stops.patience): Scalar = {
      (0 until param.batch) foreach { _ ⇒ {
        val pair = generate()
        val out = corrupt(pair._1) >>: net
        val err: ScalarMatrix = error.derivative(pair._2, out) / param.batch.toDouble
        net ! err
      }
      }
      algorithm(net.dW, net.W)

      var nPatience = patience

      val nLoss = if ((iter + 1) % stops.validationFreq == 0) {
        val train = validation.foldLeft(0.0) { (err, item) ⇒ err + error(item._2, item._1 >>: net)} / validation.size
        val weight = algorithm loss net.W
        if (train + weight < prevloss * stops.improveThreshold) {
          nPatience = Math.max(patience, iter * stops.patienceStep)
          saveParams()
          println(f"Iteration $iter%6d, Validation = $train%.5f, WeightLoss = $weight%.5f")
          train + weight
        } else {
          prevloss
        }
      } else {
        prevloss
      }

      if (iter < param.batch * stops.maxIter && nPatience > iter) {
        trainBatch(iter + 1, nLoss, nPatience)
      } else {
        println(f"Finished $iter%6d, Error = $nLoss%.5f")
        nLoss
      }
    }
  }

}
