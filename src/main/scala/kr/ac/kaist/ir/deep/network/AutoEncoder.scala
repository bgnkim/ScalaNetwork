package kr.ac.kaist.ir.deep.network

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.layer.Reconstructable
import play.api.libs.json.Json

/**
 * __Network__: Single-layer Autoencoder
 *
 * @param layer A __reconstructable__ layer for this network
 * @param presence the probability of non-dropped neurons (for drop-out training). `(default : 100% = 1.0)`
 */
class AutoEncoder(val layer: Reconstructable,
                  private val presence: Probability = 1.0f)
  extends Network {
  /**
   * All weights of layers
   *
   * @return all weights of layers
   */
  override val W: IndexedSeq[ScalarMatrix] = layer.W
  /**
   * All accumulated delta weights of layers
   *
   * @return all accumulated delta weights
   */
  override val dW: IndexedSeq[ScalarMatrix] = layer.dW

  /**
   * Compute output of neural network with given input (without reconstruction)
   * If drop-out is used, to average drop-out effect, we need to multiply output by presence probability.
   *
   * @param in an input vector
   * @return output of the vector
   */
  override def apply(in: ScalarMatrix): ScalarMatrix = layer(in)

  /**
   * Serialize network to JSON
   *
   * @return JsObject of this network
   */
  override def toJSON = Json.obj(
    "type" → this.getClass.getSimpleName,
    "presence" → presence.safe,
    "layers" → Json.arr(layer.toJSON)
  )

  /**
   * Reconstruct the given hidden value
   *
   * @param x hidden value to be reconstructed.
   * @return reconstruction value.
   */
  def reconstruct(x: ScalarMatrix): ScalarMatrix = layer.decodeBy(x)

  /**
   * Backpropagation algorithm
   *
   * @param err backpropagated error from error function
   */
  override def updateBy(err: ScalarMatrix) = encode_!(decode_!(err))

  /**
   * Backpropagation algorithm for decoding phrase
   *
   * @param err backpropagated error from error function
   */
  def decode_!(err: ScalarMatrix) = {
    layer decodeUpdateBy err
  }

  /**
   * Backpropagation algorithm for encoding phrase
   *
   * @param err backpropagated error from error function
   */
  def encode_!(err: ScalarMatrix) = {
    layer updateBy err
  }

  /**
   * Forward computation for training.
   * If drop-out is used, we need to drop-out entry of input vector.
   *
   * @param x input matrix
   * @return output matrix
   */
  override def passedBy(x: ScalarMatrix): ScalarMatrix = decode(encode(x))

  /**
   * Encode computation for training.
   * If drop-out is used, we need to drop-out entry of input vector.
   *
   * @param x input matrix
   * @return hidden values
   */
  def encode(x: ScalarMatrix): ScalarMatrix = {
    layer.passedBy(x)
  }

  /**
   * Decode computation for training.
   * If drop-out is used, we need to drop-out entry of input vector.
   *
   * @param x hidden values
   * @return output matrix
   */
  def decode(x: ScalarMatrix): ScalarMatrix = {
    layer.decodeBy(x)
  }

  /**
   * Sugar: Forward computation for validation. Calls apply(x)
   *
   * @param x input matrix
   * @return output matrix
   */
  override def of(x: ScalarMatrix): ScalarMatrix = {
    layer.decodeFrom(layer(x))
  }
}

