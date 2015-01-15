package kr.ac.kaist.ir.deep.function

import breeze.linalg.sum
import breeze.numerics._

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
