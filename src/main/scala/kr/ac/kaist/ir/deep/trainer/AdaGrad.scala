package kr.ac.kaist.ir.deep.trainer

import breeze.numerics._
import kr.ac.kaist.ir.deep.function._

/**
 * Algorithm: AdaGrad
 * @param rate for learning
 * @param l1decay for L1-reg
 * @param l2decay for L2-reg
 */
class AdaGrad(rate: Double = 0.6,
              protected override val l1decay: Double = 0.0000,
              protected override val l2decay: Double = 0.0001)
  extends WeightUpdater {
  /** History of update */
  private var history = Seq[ScalarMatrix]()

  /**
   * Run the algorithm
   * @param delta matrices of accumulation
   * @param weight matrices of current
   */
  override def apply(delta: Seq[ScalarMatrix], weight: Seq[ScalarMatrix]): Unit = {
    if (history.isEmpty) {
      history = delta map {
        matx ⇒ ScalarMatrix $0(matx.rows, matx.cols)
      }
    }

    delta.indices map {
      id ⇒ {
        val w = weight(id)
        val deltaW = delta(id)

        val deltaL1: ScalarMatrix = w mapValues { x ⇒ if (x > 0) l1decay else if (x < 0) -l1decay else 0.0}
        val deltaL2: ScalarMatrix = w * (l2decay * 2)
        val deltaL: ScalarMatrix = deltaL1 + deltaL2
        val deltaLoss: ScalarMatrix = deltaW + deltaL

        history(id) :+= pow(deltaLoss, 2)

        val adapted = history(id) mapValues { x ⇒ rate / Math.sqrt(x)}
        val d: ScalarMatrix = deltaLoss :* adapted

        w -= d
      }
    }
  }
}
