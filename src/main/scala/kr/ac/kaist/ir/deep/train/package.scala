package kr.ac.kaist.ir.deep

import breeze.stats.distributions.Gaussian
import kr.ac.kaist.ir.deep.function._

/**
 * Package for training
 *
 * Created by bydelta on 2015-01-02.
 */
package object train {

  /** Type of Corruption */
  trait Corruption extends (ScalarMatrix ⇒ ScalarMatrix) with Serializable

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
   * Criteria: How to train (for [[kr.ac.kaist.ir.deep.train.style.SingleThreadTrainStyle]])
   * @param miniBatch is size of mini-batch (default 100)
   * @param validationSize is size of validation set to be generated (default 20)
   */
  case class SimpleTrainingCriteria(override val miniBatch: Int = 100,
                                    override val validationSize: Int = 20) extends TrainingCriteria

  /**
   * Criteria: How to train (for [[kr.ac.kaist.ir.deep.train.style.DistBeliefTrainStyle]])
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
}
