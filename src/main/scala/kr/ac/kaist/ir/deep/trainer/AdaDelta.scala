package kr.ac.kaist.ir.deep.trainer

import breeze.numerics._
import kr.ac.kaist.ir.deep.function._

/**
 * Algorithm: AdaDelta
 * @param l1decay for L1-reg
 * @param l2decay for L2-reg
 * @param decay for AdaDelta history decay factor
 * @param epsilon for AdaDelta base factor
 */
class AdaDelta(protected override val l1decay: Double = 0.01,
               protected override val l2decay: Double = 0.01,
               private val decay: Double = 0.95,
               private val epsilon: Double = 1e-6)
  extends WeightUpdater {
  /** History of updates */
  private var grad = Seq[ScalarMatrix]()
  private var delL2 = Seq[ScalarMatrix]()

  /**
   * Run the algorithm
   * @param delta matrices of accumulation
   * @param weight matrices of current
   */
  override def apply(delta: Seq[ScalarMatrix], weight: Seq[ScalarMatrix]): Unit = {
    if (grad.isEmpty) {
      grad = delta map {
        matx ⇒ ScalarMatrix $0(matx.rows, matx.cols)
      }
      delL2 = grad map {
        _.copy
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

        val avgGradL2 = grad(id)
        val avgDeltaL2 = delL2(id)

        avgGradL2 :*= decay
        avgGradL2 += (pow(deltaLoss, 2) :* (1.0 - decay))

        val adjusted = avgGradL2 mapPairs { (key, grad2) ⇒ Math.sqrt(avgDeltaL2(key) + epsilon) / Math.sqrt(grad2 + epsilon)}
        val d: ScalarMatrix = deltaLoss :* adjusted

        w -= d

        avgDeltaL2 :*= decay
        avgDeltaL2 += (pow(d, 2) :* (1.0 - decay))
      }
    }
  }
}
