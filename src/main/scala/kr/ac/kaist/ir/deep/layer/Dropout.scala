package kr.ac.kaist.ir.deep.layer

import kr.ac.kaist.ir.deep.fn._
import play.api.libs.json.{JsObject, Json}

/**
 * __Layer__ that drop-outs its input.
 *
 * This layer has a function of "pipeline" with drop-out possibility.
 * Because dropping out neurons occurr in the hidden layer, we need some intermediate pipe that handle this feature.
 * This layer only conveys its input to its output synapse if that output is alive.
 */
trait Dropout extends Layer {
  /* On-off matrix */
  protected var onoff: ScalarMatrix = null
  /** The probability of the neuron is alive. `(Default: 1.0, 100%)` */
  private var presence: Probability = 1.0f

  /**
   * Set presence probability
   * @param p Probability to be set
   * @return Layer extended with dropout operta
   */
  def withProbability(p: Probability) = {
    presence = p
    this
  }

  /**
   * Forward computation
   *
   * @param x input matrix
   * @return output matrix
   */
  abstract override def apply(x: ScalarMatrix): ScalarMatrix =
    if (presence >= 1.0) super.apply(x)
    else super.apply(x) :* presence.safe

  /**
   * Translate this layer into JSON object (in Play! framework)
   *
   * @return JSON object describes this layer
   */
  abstract override def toJSON: JsObject = super.toJSON ++ Json.obj("Dropout" → presence)

  /**
   * Sugar: Forward computation. Calls apply(x)
   *
   * @param x input matrix
   * @return output matrix
   */
  abstract override def passedBy(x: ScalarMatrix): ScalarMatrix =
    if (presence >= 1.0) super.passedBy(x)
    else {
      onoff = ScalarMatrix $01(x.rows, x.cols, presence.safe)
      super.passedBy(x) :* onoff
    }

  /**
   * <p>Backward computation.</p>
   *
   * @note Because this layer only mediates two layers, this layer just remove propagated error for unused elements. 
   *
   * @param delta Sequence of delta amount of weight. The order must be the reverse of [[W]]
   * @param error to be propagated ( <code>dG / dF</code> is propagated from higher layer )
   * @return propagated error (in this case, <code>dG/dx</code> )
   */
  abstract override def updateBy(delta: Iterator[ScalarMatrix], error: ScalarMatrix): ScalarMatrix =
    if (presence >= 1) super.updateBy(delta, error)
    else super.updateBy(delta, error :* onoff)
}