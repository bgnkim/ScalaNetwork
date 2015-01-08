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
}
