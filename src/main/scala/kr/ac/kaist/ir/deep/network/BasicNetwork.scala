package kr.ac.kaist.ir.deep.network

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.layer.Layer
import play.api.libs.json.{JsArray, Json}

/**
 * __Network__: A basic network implementation
 * @param layers __Sequence of layers__ of this network
 */
class BasicNetwork(val layers: IndexedSeq[Layer])
  extends Network {
  /**
   * All weights of layers
   *
   * @return all weights of layers
   */
  override val W: IndexedSeq[ScalarMatrix] = layers flatMap (_.W)
  /**
   * All accumulated delta weights of layers
   *
   * @return all accumulated delta weights
   */
  override val dW: IndexedSeq[ScalarMatrix] = layers flatMap (_.dW)
  /** Collected input & output of each layer */
  protected[deep] var input: Seq[ScalarMatrix] = Seq()

  /**
   * Compute output of neural network with given input
   * If drop-out is used, to average drop-out effect, we need to multiply output by presence probability.
   *
   * @param in an input vector
   * @return output of the vector
   */
  override def apply(in: ScalarMatrix): ScalarMatrix = {
    // We don't have to store this value
    var id = 0
    var input = in
    while (id < layers.size) {
      input = layers(id)(input)
      id += 1
    }
    input
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
  override def updateBy(err: ScalarMatrix) = {
    var id = layers.size - 1
    var error = err
    while (id >= 0) {
      val l = layers(id)
      val out = input.head
      input = input.tail
      val in = input.head
      error = l updateBy(error, in, out)
      id -= 1
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
  override def into_:(x: ScalarMatrix): ScalarMatrix = {
    // We have to store this value
    var id = 0
    input = x.copy +: input
    while (id < layers.size) {
      input = (input.head into_: layers(id)) +: input
      id += 1
    }
    input.head
  }
}
