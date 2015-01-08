package kr.ac.kaist.ir.deep.network

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.layer.Reconstructable
import play.api.libs.json.Json

class AutoEncoder(private val layer: Reconstructable, protected[deep] override val presence: Probability = 1.0) extends Network {
  /** Collected input & output of each layer */
  private var input: ScalarMatrix = null
  private var hidden: ScalarMatrix = null
  private var output: ScalarMatrix = null

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
   * Reconstruct the input
   *
   * @param x to be reconstructed.
   * @return reconstruction of x.
   */
  def reconstruct(x: ScalarMatrix): ScalarMatrix = {
    val h = (x >>: layer) :* presence
    (h rec_>>: layer) :* presence
  }

  /**
   * Backpropagation algorithm
   * @param err backpropagated error from error function
   */
  protected[deep] override def !(err: ScalarMatrix) = {
    val reconErr = layer rec_!(err, hidden, output)
    layer !(reconErr, input, hidden)
  }

  /**
   * Forward computation for training.
   * If drop-out is used, we need to drop-out entry of input vector.
   *
   * @param x of input matrix
   * @return output matrix
   */
  protected[deep] override def >>:(x: ScalarMatrix): ScalarMatrix = {
    input = x.copy
    if (presence < 1.0)
      input :*= ScalarMatrix $01(x.rows, x.cols, presence.safe)
    hidden = input >>: layer
    if (presence < 1.0)
      hidden :*= ScalarMatrix $01(hidden.rows, hidden.cols, presence.safe)
    output = hidden rec_>>: layer
    output
  }
}
