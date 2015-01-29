package kr.ac.kaist.ir.deep.network

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.layer.Layer
import play.api.libs.json.{JsArray, Json}

/**
 * __Network__: A basic network implementation
 * @param layers __Sequence of layers__ of this network
 */
class BasicNetwork(private val layers: Seq[Layer])
  extends Network {
  /** Collected input & output of each layer */
  protected[deep] var input: Seq[ScalarMatrix] = Seq()

  /**
   * All weights of layers
   *
   * @return all weights of layers
   */
  override def W = layers flatMap (_.W)

  /**
   * All accumulated delta weights of layers
   *
   * @return all accumulated delta weights
   */
  override def dW = layers flatMap (_.dW)

  /**
   * Compute output of neural network with given input
   * If drop-out is used, to average drop-out effect, we need to multiply output by presence probability.
   *
   * @param in an input vector
   * @return output of the vector
   */
  override def apply(in: ScalarMatrix): ScalarMatrix =
    // We don't have to store this value
    layers.indices.foldLeft(in) {
      (input, id) ⇒ layers(id)(input)
    }

  /**
   * Serialize network to JSON
   *
   * @return JsObject of this network
   */
  override def toJSON = Json.obj(
    "type" → "BasicNetwork",
    "layers" → JsArray(layers map (_.toJSON))
  )

  /**
   * Backpropagation algorithm
   *
   * @param err backpropagated error from error function
   */
  protected[deep] override def !(err: ScalarMatrix) = {
    val error = layers.indices.foldRight(err) {
      (id, e) ⇒
        val l = layers(id)
        val out = input.head
        input = input.tail
        val in = input.head
        l !(e, in, out)
    }

    // Clean-up last entry
    input = input.tail
    error
  }

  /**
   * Forward computation for training.
   * If drop-out is used, we need to drop-out entry of input vector.
   *
   * @param x input matrix
   * @return output matrix
   */
  protected[deep] override def >>:(x: ScalarMatrix): ScalarMatrix = {
    // We have to store this value
    input = layers.indices.foldLeft(Seq(x.copy)) {
      (seq, id) ⇒ (seq.head >>: layers(id)) +: seq
    } ++: input
    input.head
  }
}
