package kr.ac.kaist.ir.deep.network

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.layer.Reconstructable
import play.api.libs.json.Json

/**
 * Network: Single-layer Autoencoder
 * @param layer for this network
 * @param presence is the probability of non-dropped neurons (for drop-out training). (default : 100% = 1.0)
 */
class AutoEncoder(private val layer: Reconstructable,
                  protected[deep] override val presence: Probability = 1.0)
  extends Network {
  /** Collected input & output of each layer */
  protected[deep] var input: Seq[ScalarMatrix] = Seq()

  /**
   * All weights of layers
   * @return all weights of layers
   */
  override def W = layer.W

  /**
   * All accumulated delta weights of layers
   * @return all accumulated delta weights
   */
  override def dW = layer.dW

  /**
   * Compute output of neural network with given input (without reconstruction)
   * If drop-out is used, to average drop-out effect, we need to multiply output by presence probability.
   *
   * @param in is an input vector
   * @return output of the vector
   */
  override def apply(in: ScalarMatrix): ScalarMatrix = layer(in) :* presence.safe

  /**
   * Serialize network to JSON
   * @return JsObject
   */
  override def toJSON = Json.obj(
    "type" → "AutoEncoder",
    "presence" → presence.safe,
    "layers" → Json.arr(layer.toJSON)
  )

  /**
   * Backpropagation algorithm
   * @param err backpropagated error from error function
   */
  protected[deep] override def !(err: ScalarMatrix) = encode_!(decode_!(err))

  /**
   * Backpropagation algorithm for decoding phrase
   * @param err backpropagated error from error function
   */
  protected[deep] def decode_!(err: ScalarMatrix) = {
    val output = input.head
    val hidden = input.tail.head
    input = input.tail.tail

    layer rec_!(err, hidden, output)
  }

  /**
   * Backpropagation algorithm for encoding phrase
   * @param err backpropagated error from error function
   */
  protected[deep] def encode_!(err: ScalarMatrix) = {
    val hidden = input.head
    val in = input.tail.head
    input = input.tail.tail

    layer !(err, in, hidden)
  }

  /**
   * Forward computation for training.
   * If drop-out is used, we need to drop-out entry of input vector.
   *
   * @param x of input matrix
   * @return output matrix
   */
  protected[deep] override def >>:(x: ScalarMatrix): ScalarMatrix = decode(encode(x))
  
  /**
   * Encode computation for training.
   * If drop-out is used, we need to drop-out entry of input vector.
   *
   * @param x of input matrix
   * @return hidden values
   */
  protected[deep] def encode(x: ScalarMatrix): ScalarMatrix = {
    val in = x.copy
    if (presence < 1.0)
      in :*= ScalarMatrix $01(x.rows, x.cols, presence.safe)
    val hidden = in >>: layer
    input = Seq(hidden, in) ++: input
    hidden
  }

  /**
   * Decode computation for training.
   * If drop-out is used, we need to drop-out entry of input vector.
   *
   * @param x of hidden values
   * @return output matrix
   */
  protected[deep] def decode(x: ScalarMatrix): ScalarMatrix = {
    val hidden = x.copy
    if (presence < 1.0)
      hidden :*= ScalarMatrix $01(hidden.rows, hidden.cols, presence.safe)
    val output = hidden rec_>>: layer
    input = Seq(output, hidden) ++: input
    output
  }

  /**
   * Sugar: Forward computation for validation. Calls apply(x)
   *
   * @param x of input matrix
   * @return output matrix
   */
  override protected[deep] def on(x: ScalarMatrix): ScalarMatrix = reconstruct(x)

  /**
   * Reconstruct the input
   *
   * @param x to be reconstructed.
   * @return reconstruction of x.
   */
  def reconstruct(x: ScalarMatrix): ScalarMatrix = {
    val h = (x >>: layer) :* presence
    (h rec_>>: layer) :* presence
  }
}
