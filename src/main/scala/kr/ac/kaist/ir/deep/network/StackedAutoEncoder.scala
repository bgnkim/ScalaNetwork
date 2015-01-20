package kr.ac.kaist.ir.deep.network

import kr.ac.kaist.ir.deep.fn._
import play.api.libs.json.{JsObject, Json}

/**
 * __Network__: Stack of autoencoders. 
 *
 * @param encoders __Sequence of AutoEncoders__ to be stacked.
 */
class StackedAutoEncoder(private val encoders: Seq[AutoEncoder]) extends Network {
  /** Not used for this network */
  @deprecated protected override val presence: Probability = 0.0

  /**
   * All accumulated delta weights of layers
   *
   * @return all accumulated delta weights
   */
  override def dW: Seq[ScalarMatrix] = encoders flatMap (_.dW)

  /**
   * All weights of layers
   *
   * @return all weights of layers
   */
  override def W: Seq[ScalarMatrix] = encoders flatMap (_.W)

  /**
   * Serialize network to JSON
   *
   * @return JsObject of this network
   */
  override def toJSON: JsObject =
    Json.obj(
      "type" → "StackedAutoEncoder",
      "stack" → Json.arr(encoders map (_.toJSON))
    )

  /**
   * Compute output of neural network with given input (without reconstruction)
   * If drop-out is used, to average drop-out effect, we need to multiply output by presence probability.
   *
   * @param in an input vector
   * @return output of the vector
   */
  override def apply(in: ScalarMatrix): ScalarMatrix =
    encoders.foldLeft(in) {
      (x, enc) ⇒ enc(x)
    }

  /**
   * Sugar: Forward computation for training. Calls apply(x)
   *
   * @param x input matrix
   * @return output matrix
   */
  protected[deep] override def >>:(x: ScalarMatrix): ScalarMatrix =
    encoders.foldLeft(x) {
      (in, enc) ⇒ in >>: enc
    }

  /**
   * Backpropagation algorithm
   *
   * @param err backpropagated error from error function
   */
  protected[deep] override def !(err: ScalarMatrix): ScalarMatrix =
    encoders.foldRight(err) {
      (enc, err) ⇒ enc ! err
    }
}
