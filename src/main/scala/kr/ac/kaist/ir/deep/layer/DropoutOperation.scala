package kr.ac.kaist.ir.deep.layer

import kr.ac.kaist.ir.deep.fn._
import play.api.libs.json.{JsObject, Json}

/**
 * __Layer__ that drop-outs its input.
 *
 * This layer has a function of "pipeline" with drop-out possibility.
 * Because dropping out neurons occurr in the hidden layer, we need some intermediate pipe that handle this feature.
 * This layer only conveys its input to its output synapse if that output is alive.
 *
 * @param presence The probability of the neuron is alive. `(Default: 1.0, 100%)`
 */
class DropoutOperation(protected val presence: Probability = 1.0f) extends Layer {
  /**
   * weights for update
   *
   * @return weights
   */
  override val W: IndexedSeq[ScalarMatrix] = IndexedSeq.empty
  /**
   * accumulated delta values
   *
   * @return delta-weight
   */
  override val dW: IndexedSeq[ScalarMatrix] = IndexedSeq.empty
  /** Null activation */
  protected override val act = null
  /* On-off matrix */
  protected var onoff: ScalarMatrix = null

  /**
   * Forward computation
   *
   * @param x input matrix
   * @return output matrix
   */
  override def apply(x: ScalarMatrix): ScalarMatrix =
    if (presence >= 1.0) x
    else x :* presence.safe

  /**
   * Translate this layer into JSON object (in Play! framework)
   *
   * @return JSON object describes this layer
   */
  override def toJSON: JsObject = Json.obj(
    "type" → "DropoutOp",
    "presence" → presence.safe
  )

  /**
   * Sugar: Forward computation. Calls apply(x)
   *
   * @param x input matrix
   * @return output matrix
   */
  override protected[deep] def into_:(x: ScalarMatrix): ScalarMatrix =
    if (presence >= 1.0) x
    else {
      onoff = ScalarMatrix $01(x.rows, x.cols, presence.safe)
      x :* onoff
    }

  /**
   * <p>Backward computation.</p>
   *
   * @note Because this layer only mediates two layers, this layer just remove propagated error for unused elements. 
   *
   * @param error to be propagated ( <code>dG / dF</code> is propagated from higher layer )
   * @param input of this layer (in this case, <code>x = entry of dX / dw</code>)
   * @param output of this layer (in this case, <code>y</code>)
   * @return propagated error (in this case, <code>dG/dx</code> )
   */
  protected[deep] override def updateBy(error: ScalarMatrix, input: ScalarMatrix, output: ScalarMatrix): ScalarMatrix =
    if (presence >= 1) error
    else error :* onoff
}