package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network.Network

/**
 * __Trait__ that describes style of training
 *
 * This trait controls how to train, i.e. __Single-threaded__ or __Distributed__.
 *
 * @tparam IN the type of input
 * @tparam OUT the type of output
 */
trait TrainStyle[IN, OUT] extends Serializable {
  /** Training Pair Type */
  type Pair = (IN, OUT)
  /** Training parameters */
  protected[train] val param: TrainingCriteria
  /** Network */
  protected[train] val net: Network
  /** Algorithm */
  protected[train] val algorithm: WeightUpdater
  /** Training Set */
  protected[train] var trainingSet: Int ⇒ Seq[Pair] = null

  /**
   * Fetch weights
   *
   * @param iter current iteration
   */
  protected[train] def fetch(iter: Int): Unit

  /**
   * Do mini-batch
   *
   * @param make Set of input operations
   */
  protected[train] def batch(make: ManipulationType[IN, OUT]): Unit

  /**
   * Send update of weights
   *
   * @param iter current iteration
   */
  protected[train] def update(iter: Int): Unit

  /**
   * Implicit weight operation
   *
   * @param w Sequence of weight to be applied
   */
  implicit class WeightOp(w: Seq[ScalarMatrix]) extends Serializable {
    /**
     * Sugar: Weight update
     *
     * @param dw A amount of update i.e. __ΔWeight__
     */
    def -=(dw: Seq[ScalarMatrix]) = algorithm(dw, w)
  }

}
