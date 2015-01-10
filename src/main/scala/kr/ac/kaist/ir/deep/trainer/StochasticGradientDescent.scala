package kr.ac.kaist.ir.deep.trainer

import kr.ac.kaist.ir.deep.function._

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
    lastDelta = delta.indices.par map {
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