package kr.ac.kaist.ir.deep.fn.alg

import breeze.numerics._
import kr.ac.kaist.ir.deep.fn._

/**
 * __Algorithm__: AdaDelta algorithm
 *
 * If you are trying to use this algorithm for your research, you should add a reference to [[http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf AdaDelta techinical report]].
 *
 * @param l1decay L,,1,, regularization factor `(Default 0.0)`
 * @param l2decay L,,2,, regularization factor `(Default 0.0001)`
 * @param decay AdaDelta history decay factor `(Default 95% = 0.95)`
 * @param epsilon AdaDelta base factor `(Default 1e-6)`
 *
 * @example {{{val algorithm = new AdaDelta(l2decay = 0.0001)}}}
 */
class AdaDelta(protected override val l1decay: Double = 0.0000,
               protected override val l2decay: Double = 0.0001,
               private val decay: Double = 0.95,
               private val epsilon: Double = 1e-6)
  extends WeightUpdater {
  /** accumulated history of gradients */
  private var gradSq = Seq[ScalarMatrix]()
  /** accumulated history of parameter updates */
  private var deltaSq = Seq[ScalarMatrix]()

  /**
   * Execute the algorithm for given __sequence of Δweight__ and sequence of __weights__
   *
   * @param delta the __sequence of accumulated Δweight__
   * @param weight the __sequence of current weights__
   */
  override def apply(delta: Seq[ScalarMatrix], weight: Seq[ScalarMatrix]): Unit = {
    if (gradSq.isEmpty) {
      gradSq = delta map {
        matx ⇒ ScalarMatrix $0(matx.rows, matx.cols)
      }
      deltaSq = gradSq.copy
    }

    delta.indices.par foreach {
      id ⇒
        val w = weight(id)
        val deltaW = delta(id)

        val deltaL1 = w mapValues { x ⇒ if (x > 0) l1decay else if (x < 0) -l1decay else 0.0}
        val deltaL2 = w * (l2decay * 2)
        val deltaLoss = deltaW + deltaL1 + deltaL2

        val gradSq_id = gradSq(id)
        val deltaSq_id = deltaSq(id)

        gradSq_id :*= decay
        gradSq_id += (pow(deltaLoss, 2) :* (1.0 - decay))

        val adjusted = ScalarMatrix $0(gradSq_id.rows, gradSq_id.cols)
        gradSq_id foreachKey { key ⇒ adjusted.update(key, Math.sqrt(deltaSq_id(key) + epsilon) / Math.sqrt(gradSq_id(key) + epsilon))}

        val d = deltaLoss :* adjusted

        w -= d

        deltaSq_id :*= decay
        deltaSq_id += (pow(d, 2) :* (1.0 - decay))
    }
  }
}
