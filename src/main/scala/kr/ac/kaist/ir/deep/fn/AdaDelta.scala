package kr.ac.kaist.ir.deep.fn

import breeze.numerics._

import scala.collection.mutable.ArrayBuffer

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
class AdaDelta(protected override val l1decay: Scalar = 0.0000f,
               protected override val l2decay: Scalar = 0.0001f,
               private val decay: Scalar = 0.95f,
               private val epsilon: Scalar = 1e-6f)
  extends WeightUpdater {
  /** accumulated history of gradients */
  private val gradSq = ArrayBuffer[ScalarMatrix]()
  /** accumulated history of parameter updates */
  private val deltaSq = ArrayBuffer[ScalarMatrix]()

  /**
   * Execute the algorithm for given __sequence of Δ(weight)__ and sequence of __weights__
   *
   * @param delta the __sequence of accumulated Δ(weight)__
   * @param weight the __sequence of current weights__
   */
  override def apply(delta: IndexedSeq[ScalarMatrix], weight: IndexedSeq[ScalarMatrix]): Unit = {
    if (gradSq.isEmpty) {
      gradSq.sizeHint(delta)
      deltaSq.sizeHint(delta)

      var i = 0
      while (i < delta.length) {
        val matx = delta(i)
        i += 1

        gradSq += ScalarMatrix.$0(matx.rows, matx.cols)
        deltaSq += ScalarMatrix.$0(matx.rows, matx.cols)
      }
    }

    var id = delta.length - 1
    while (id >= 0) {
      val w = weight(id)
      val deltaW = delta(id)

      val deltaL1 = w mapValues { x ⇒ if (x > 0) l1decay else if (x < 0) -l1decay else 0.0f}
      val deltaL2 = w * (l2decay * 2)
      val deltaLoss: ScalarMatrix = deltaW + deltaL1 + deltaL2

      val gradSq_id = gradSq(id)
      val deltaSq_id = deltaSq(id)

      gradSq_id :*= decay
      gradSq_id += (pow(deltaLoss, 2.0f) :* (1.0f - decay))

      val adjusted = ScalarMatrix $0(gradSq_id.rows, gradSq_id.cols)
      val iter = gradSq_id.keysIterator
      while (iter.hasNext) {
        val key = iter.next()
        val value = Math.sqrt(deltaSq_id(key) + epsilon) / Math.sqrt(gradSq_id(key) + epsilon)
        adjusted.update(key, value.toFloat)
      }

      val d = deltaLoss :* adjusted

      w -= d

      deltaSq_id :*= decay
      deltaSq_id += (pow(d, 2.0f) :* (1.0f - decay))

      id -= 1
    }
  }
}
