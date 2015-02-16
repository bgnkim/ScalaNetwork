package kr.ac.kaist.ir.deep.fn

import scala.collection.mutable.ArrayBuffer

/**
 * __Algorithm__: Stochastic Gradient Descent
 *
 * Basic Gradient Descent rule with mini-batch training.
 *
 * @param rate the learning rate `(Default 0.03)`
 * @param l1decay L,,1,, regularization factor `(Default 0.0)`
 * @param l2decay L,,2,, regularization factor `(Default 0.0001)`
 * @param momentum Momentum factor for adaptive learning `(Default 0.0001)`
 *
 * @example {{{val algorithm = new StochasticGradientDescent(l2decay = 0.0001)}}}
 */
class StochasticGradientDescent(rate: Double = 0.03,
                                protected override val l1decay: Double = 0.0000,
                                protected override val l2decay: Double = 0.0001,
                                momentum: Double = 0.0001)
  extends WeightUpdater {
  /** the last update of parameters */
  private val lastDelta = ArrayBuffer[ScalarMatrix]()

  /**
   * Execute the algorithm for given __sequence of Δweight__ and sequence of __weights__
   *
   * @param delta the __sequence of accumulated Δweight__
   * @param weight the __sequence of current weights__
   */
  override def apply(delta: IndexedSeq[ScalarMatrix], weight: IndexedSeq[ScalarMatrix]): Unit = {
    if (lastDelta.isEmpty) {
      lastDelta.sizeHint(delta)
      var i = 0
      while (i < delta.size) {
        val matx = delta(i)
        lastDelta += ScalarMatrix.$0(matx.rows, matx.cols)
        i += 1
      }
    }

    var id = delta.size - 1
    while (id >= 0) {
      val w = weight(id)
      val deltaW = delta(id)

      val deltaL1 = w mapValues { x ⇒ if (x > 0) l1decay else if (x < 0) -l1decay else 0.0}
      val deltaL2 = w :* (l2decay * 2)
      val deltaLoss: ScalarMatrix = deltaW + deltaL1 + deltaL2
      val adapted: ScalarMatrix = deltaLoss :* rate

      val dw = if (lastDelta.nonEmpty) {
        val moment: ScalarMatrix = lastDelta(id) :* momentum
        moment - adapted
      } else {
        -adapted
      }

      w += dw
      deltaW := 0.0
      lastDelta.update(id, dw)
      id -= 1
    }
  }
}