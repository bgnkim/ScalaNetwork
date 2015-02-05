package kr.ac.kaist.ir.deep.fn

import breeze.numerics._

import scala.collection.mutable.ArrayBuffer

/**
 * __Algorithm__: AdaGrad algorithm.
 *
 * If you are trying to use this algorithm for your research, you should add a reference to [[http://www.magicbroom.info/Papers/DuchiHaSi10.pdf AdaGrad paper]].
 *
 * @param rate the learning rate `(Default 0.6)`
 * @param l1decay L,,1,, regularization factor `(Default 0.0)`
 * @param l2decay L,,2,, regularization factor `(Default 0.0001)`
 *
 *
 * @example {{{val algorithm = new AdaGrad(l2decay = 0.0001)}}}
 */
class AdaGrad(rate: Double = 0.6,
              protected override val l1decay: Double = 0.0000,
              protected override val l2decay: Double = 0.0001)
  extends WeightUpdater {
  /** accumulated history of parameter updates */
  private val history = ArrayBuffer[ScalarMatrix]()

  /**
   * Execute the algorithm for given __sequence of Δweight__ and sequence of __weights__
   *
   * @param delta the __sequence of accumulated Δweight__
   * @param weight the __sequence of current weights__
   */
  override def apply(delta: IndexedSeq[ScalarMatrix], weight: IndexedSeq[ScalarMatrix]): Unit = {
    if (history.isEmpty) {
      delta foreach {
        matx ⇒ history += ScalarMatrix.$0(matx.rows, matx.cols)
      }
    }

    delta.indices.par foreach {
      id ⇒
        val w = weight(id)
        val deltaW = delta(id)

        val deltaL1 = w mapValues { x ⇒ if (x > 0) l1decay else if (x < 0) -l1decay else 0.0}
        val deltaL2 = w * (l2decay * 2)
        val deltaLoss = deltaW + deltaL1 + deltaL2

        history(id) :+= pow(deltaLoss, 2)

        val adapted = history(id) mapValues { x ⇒ rate / Math.sqrt(x)}
        val d = deltaLoss :* adapted

        w -= d
    }
  }
}
