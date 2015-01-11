package kr.ac.kaist.ir.deep

import breeze.linalg.sum
import breeze.numerics.{abs, pow}
import breeze.stats.distributions.Gaussian
import kr.ac.kaist.ir.deep.function._

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
     * @param set to be used for training (Randomized Sequence Generator)
     * @param validation to be used for validation (Sequence Generator)
     * @return Training error (loss)
     */
    def train(set: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)],
              validation: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)]): Scalar

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
     * Train autoencoder with given sequence.
     * @param set to be used for input & reconstruction. (Randomized Sequence Generator)
     * @return Training error (loss)
     */
    def trainAutoencoder(set: Int ⇒ Seq[ScalarMatrix]): Scalar

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
    extends Serializable

  /**
   * Criteria: How to train
   * @param miniBatch is size of mini-batch
   * @param validationSize is size of validation set to be generated
   */
  case class TrainingCriteria(miniBatch: Int = 100,
                              validationSize: Int = 20) extends Serializable

  /**
   * Criteria: How to train (for DistBelief-style online trainer, [[kr.ac.kaist.ir.deep.trainer.SparkTrainer]])
   * @param miniBatch is size of mini-batch
   * @param validationSize is size of validation set to be generated
   * @param updateStep is number of "numCores mini-batches" between update
   * @param fetchStep is number of "numCores mini-batches" between fetching
   * @param numCores is number of v-cores in the spark cluster.
   */
  case class DistBeliefCriteria(miniBatch: Int = 100,
                                validationSize: Int = 20,
                                updateStep: Int = 1,
                                fetchStep: Int = 1,
                                numCores: Int = 1)
}
