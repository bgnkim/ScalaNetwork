package kr.ac.kaist.ir.deep.fn

import breeze.linalg.sum
import breeze.numerics._

import scala.collection.mutable.ArrayBuffer

/**
 * __Trait__ that describes the algorithm for weight update
 *
 * Because each weight update requires history, we recommend to make inherited one as a class. 
 */
trait WeightUpdater extends ((IndexedSeq[ScalarMatrix], IndexedSeq[ScalarMatrix]) ⇒ Unit) with Serializable {
  /** Decay factor for L,,1,, regularization */
  protected val l1decay: Scalar
  /** Decay factor for L,,2,, regularization */
  protected val l2decay: Scalar

  /**
   * Execute the algorithm for given __sequence of Δweight__ and sequence of __weights__
   *
   * @param delta the __sequence of accumulated Δweight__
   * @param weight the __sequence of current weights__
   */
  override def apply(delta: IndexedSeq[ScalarMatrix], weight: IndexedSeq[ScalarMatrix]): Unit

  /**
   * Compute weight-loss of given weight parameters
   *
   * @param seq the __sequence__ of weight matrices
   * @return the total weight loss of this sequence
   */
  def loss(seq: Seq[ScalarMatrix]) = {
    var i = 0
    var err = 0.0f
    while (i < seq.size) {
      val obj = seq(i)
      val l1loss = sum(abs(obj)) * l1decay
      val l2loss = sum(pow(obj, 2)) * l2decay
      err += l1loss + l2loss
      i += 1
    }
    err
  }
}

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

    delta.indices.par.foreach {
      id ⇒
        val w = weight(id)
        val deltaW = delta(id)
        val gradSq_id = gradSq(id)
        val deltaSq_id = deltaSq(id)

        val keys = w.keysIterator

        while (keys.hasNext) {
          val (r, c) = keys.next()
          val x = w(r, c)
          val l1 = if (x > 0) l1decay else if (x < 0) -l1decay else 0.0f
          val d = (l2decay * 2 * x) + l1 + deltaW(r, c)

          val gSq = gradSq_id(r, c) * decay + d * d * (1.0f - decay)
          val dSq = deltaSq_id(r, c)

          val rate = Math.sqrt(dSq + epsilon) / Math.sqrt(gSq + epsilon)
          gradSq_id.update(r, c, gSq)

          val dw = d * rate.toFloat
          w.update(r, c, x - dw)

          deltaSq_id.update(r, c, dSq * decay + dw * dw * (1.0f - decay))
        }
    }
  }
}


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
class AdaGrad(rate: Scalar = 0.6f,
              protected override val l1decay: Scalar = 0.0000f,
              protected override val l2decay: Scalar = 0.0001f)
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
      history.sizeHint(delta)

      var i = 0
      while (i < delta.size) {
        val matx = delta(i)
        i += 1
        history += ScalarMatrix.$0(matx.rows, matx.cols)
      }
    }

    delta.indices.par.foreach {
      id ⇒
        val w = weight(id)
        val deltaW = delta(id)
        val hW = history(id)

        val keys = w.keysIterator

        while (keys.hasNext) {
          val (r, c) = keys.next()
          val x = w(r, c)
          val l1 = if (x > 0) l1decay else if (x < 0) -l1decay else 0.0f
          val d = (l2decay * 2 * x) + l1 + deltaW(r, c)
          val h = hW(r, c) + d * d

          hW.update(r, c, h)
          w.update(r, c, x - d * (rate / Math.sqrt(h)).toFloat)
        }
    }
  }
}

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
class StochasticGradientDescent(rate: Scalar = 0.03f,
                                protected override val l1decay: Scalar = 0.0000f,
                                protected override val l2decay: Scalar = 0.0001f,
                                momentum: Scalar = 0.0001f)
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
    if (momentum > 0 && lastDelta.isEmpty) {
      lastDelta.sizeHint(delta)
      var i = 0
      while (i < delta.size) {
        val matx = delta(i)
        lastDelta += ScalarMatrix.$0(matx.rows, matx.cols)
        i += 1
      }
    }

    delta.indices.par.foreach {
      id ⇒
        val w = weight(id)
        val deltaW = delta(id)
        val hW = if (momentum > 0) lastDelta(id) else null

        w foreachPair {
          case ((r, c), x) ⇒
            val l1 = if (x > 0) l1decay else if (x < 0) -l1decay else 0.0f
            val d = (l2decay * 2 * x + l1 + deltaW(r, c)) * rate
            deltaW.update(r, c, 0f)
            if (hW != null) {
              val dw = -d + hW(r, c) * momentum
              w.update(r, c, x + dw)
              hW.update(r, c, dw)
            } else {
              w.update(r, c, x - d)
            }
        }
    }
  }
}