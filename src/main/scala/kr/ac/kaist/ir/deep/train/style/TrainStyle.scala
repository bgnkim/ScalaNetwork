package kr.ac.kaist.ir.deep.train.style

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.train.TrainingCriteria
import kr.ac.kaist.ir.deep.train.op.InputOp

/**
 * Trait: Training Style
 *
 * This trait controls how to train : Single-threaded or Distributed.
 */
trait TrainStyle[IN] extends Serializable {
  /** Training Pair Type */
  type Pair = (IN, ScalarMatrix)
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
   * @param iter is current iteration
   */
  protected[train] def fetch(iter: Int): Unit

  /**
   * Do mini-batch
   * @param op : set of input operations
   */
  protected[train] def batch(op: InputOp[IN]): Unit

  /**
   * Send update of weights
   * @param iter is current iteration
   */
  protected[train] def update(iter: Int): Unit

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

}
