package kr.ac.kaist.ir.deep

import breeze.linalg.sum
import breeze.numerics.{abs, pow}
import breeze.stats.distributions.Gaussian
import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.network.{AutoEncoder, Network}
import org.apache.log4j.Logger

/**
 * Package for training
 *
 * Created by bydelta on 2015-01-02.
 */
package object trainer {

  /** Type of Corruption */
  trait Corruption extends (ScalarMatrix ⇒ ScalarMatrix) with Serializable

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
    /** Algorithm */
    protected val algorithm: WeightUpdater
    /** Network */
    protected val net: Network
    /** Objective function */
    protected val error: Objective
    /** Corruption function */
    protected val corrupt: Corruption
    /** Training parameters */
    protected val param: TrainingCriteria
    /** Training Set */
    protected var trainingSet: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)] = null
    /** Validation Set */
    protected var testSet: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)] = null
    /** Best Parameter History */
    @transient protected var bestParam: Seq[ScalarMatrix] = null
    @transient protected var bestIter: Int = 0
    /** Logger */
    @transient protected val logger = Logger.getLogger(this.getClass)

    /**
     * Implicit weight operation 
     * @param w to be applied
     */
    implicit class WeightOp(w: Seq[ScalarMatrix]) extends Serializable {
      /**
       * Sugar: Weight update 
       * @param dw is a amount of update
       */
      def -=(dw: Seq[ScalarMatrix]) = algorithm(dw, w)
    }

    /**
     * Train given sequence, and validate with given sequence.
     * @param set to be trained (Random Sequence Generator)
     * @return Training error (loss)
     */
    def train(set: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)]): Scalar = train(set, set)

    /**
     * Train given sequence, and validate with given sequence.
     * @param set to be trained (Full Seq)
     * @return Training error (loss)
     */
    def train(set: Seq[(ScalarMatrix, ScalarMatrix)]): Scalar = {
      val index = () ⇒ Math.floor(Math.random() * set.size).toInt
      val randomizer = (n: Int) ⇒ (0 until n) map { _ ⇒ set(index())}
      train(randomizer, randomizer)
    }

    /**
     * Train given sequence, and validate with another sequence.
     * @param set to be used for training (Full Seq)
     * @param validation to be used for validation (Full Seq)
     * @return Training error (loss)
     */
    def train(set: Seq[(ScalarMatrix, ScalarMatrix)],
              validation: Seq[(ScalarMatrix, ScalarMatrix)]): Scalar = {
      val index = () ⇒ Math.floor(Math.random() * set.size).toInt
      val randomizer = (n: Int) ⇒ (0 until n) map { _ ⇒ set(index())}
      val topN = (n: Int) ⇒ validation.slice(0, n)
      train(randomizer, topN)
    }

    /**
     * Train given sequence, and validate with another sequence.
     *
     * @param set to be used for training (Randomized Sequence Generator)
     * @param validation to be used for validation (Sequence Generator)
     * @return Training error (loss)
     */
    def train(set: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)],
              validation: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)]) = {
      trainingSet = set
      testSet = validation

      saveParams()
      val err = trainBatch()
      restoreParams()
      printValidation()

      err
    }

    /**
     * Train autoencoder with given sequence.
     * @param set to be used for input & reconstruction. (Full Seq)
     * @return Training error (loss)
     */
    def trainAutoencoder(set: Seq[ScalarMatrix]): Scalar = {
      val index = () ⇒ Math.floor(Math.random() * set.size).toInt
      val randomizer = (n: Int) ⇒ (0 until n) map { _ ⇒ set(index())}
      trainAutoencoder(randomizer)
    }

    /**
     * Train autoencoder with given sequence.
     * @param set to be used for input & reconstruction. (Randomized Sequence Generator)
     * @return Training error (loss)
     */
    def trainAutoencoder(set: Int ⇒ Seq[ScalarMatrix]): Scalar = {
      trainingSet = set andThen { seq ⇒ seq map { item ⇒ item → item}}
      testSet = trainingSet

      saveParams()
      val err = trainBatch(isAutoEncoder = true)
      restoreParams()
      printValidation(isAutoEncoder = true)

      err
    }

    /**
     * Calculate validation error
     * @param isAutoEncoder true if it is autoencoder training
     * @return validation error
     */
    protected def validationError(isAutoEncoder: Boolean = false) = {
      val t = testSet(param.validationSize)
      t.foldLeft(0.0) { (err, item) ⇒ {
        val in = item._1
        val out = if (isAutoEncoder) in >>: net else net(in)
        logger.debug(s"IN ${in.mkString} : EXP ${item._2.mkString} = OUT : ${out.mkString}")
        err + error(item._2, out)
      }
      } / t.size
    }

    /**
     * Print validation result
     * @param isAutoEncoder true if it is autoencoder training
     */
    protected def printValidation(isAutoEncoder: Boolean = false) = {
      logger.info(s"BEST ITERATION $bestIter : W = ${net.W map (_.mkString) mkString " | "}")

      val t = testSet(param.validationSize)
      t.par foreach {
        item ⇒ {
          val in = item._1
          val out = if (isAutoEncoder) net.asInstanceOf[AutoEncoder].reconstruct(in) else net(in)
          logger.info(s"IN ${in.mkString} : EXP ${item._2.mkString} = OUT : ${out.mkString}")
        }
      }
    }

    /**
     * Store best parameters
     */
    protected final def saveParams() = {
      bestParam = net.W.copy
    }

    /**
     * Restore best parameters
     */
    protected final def restoreParams() = {
      net.W := bestParam
    }

    /**
     * Tail Recursive : Train each batch
     * @param iter indicates current iteration
     * @param prevloss indicates previous loss
     * @param patience indicates current patience
     * @param isAutoEncoder is a flag for autoencoder.
     * @return Total Loss when train is finished
     */
    protected def trainBatch(iter: Int = 0,
                             prevloss: Double = Double.MaxValue,
                             patience: Int = stops.patience,
                             isAutoEncoder: Boolean = false): Scalar
  }

  /**
   * Input Corruption: Drop input as zero.
   *
   * If network uses drop-out training, we recommend that you do not use this.
   *
   * @param presence probability of not-dropped. (default 95% = 0.95)
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
   * @param mean of noise (default 0.0)
   * @param variance of noise (default 0.1)
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
   * @param maxIter is maximum mini-batch iteration count (default 100,000)
   * @param patience is default patience count (default 5,000)
   * @param patienceStep is default step for patience (default x2)
   * @param improveThreshold is threshold for marked as "improved" (default 95% = 0.95)
   * @param lossThreshold is maximum-tolerant loss value. (default 0.0001)
   * @param validationFreq is step count for validation (default 100)
   */
  case class StoppingCriteria(maxIter: Int = 100000,
                              patience: Int = 5000,
                              patienceStep: Int = 2,
                              improveThreshold: Double = 0.995,
                              lossThreshold: Double = 0.0001,
                              validationFreq: Int = 100)
    extends Serializable

  /**
   * Trait : Training Criteria 
   */
  trait TrainingCriteria extends Serializable {
    /** Size of mini-batch */
    val miniBatch: Int
    /** Size of validation */
    val validationSize: Int
  }

  /**
   * Criteria: How to train
   * @param miniBatch is size of mini-batch (default 100)
   * @param validationSize is size of validation set to be generated (default 20)
   */
  case class SimpleTrainingCriteria(override val miniBatch: Int = 100,
                                    override val validationSize: Int = 20) extends TrainingCriteria

  /**
   * Criteria: How to train (for DistBelief-style online trainer, [[kr.ac.kaist.ir.deep.trainer.SparkTrainer]])
   * @param miniBatch is size of mini-batch (default 100)
   * @param validationSize is size of validation set to be generated (default 20)
   * @param updateStep is number of "numCores mini-batches" between update (default 2)
   * @param fetchStep is number of "numCores mini-batches" between fetching (default 10)
   * @param numCores is number of v-cores in the spark cluster. (default 1)
   */
  case class DistBeliefCriteria(override val miniBatch: Int = 100,
                                override val validationSize: Int = 20,
                                updateStep: Int = 2,
                                fetchStep: Int = 10,
                                numCores: Int = 1) extends TrainingCriteria

}
