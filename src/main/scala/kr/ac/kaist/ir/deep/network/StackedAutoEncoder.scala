package kr.ac.kaist.ir.deep.network

import kr.ac.kaist.ir.deep.fn.ScalarMatrix
import play.api.libs.json.{JsObject, Json}

import scala.collection.mutable.ArrayBuffer

/**
 * __Network__: Stack of autoencoders. 
 *
 * @param encoders __Sequence of AutoEncoders__ to be stacked.
 */
class StackedAutoEncoder(val encoders: Seq[AutoEncoder]) extends Network {
  /**
   * All accumulated delta weights of layers
   *
   * @return all accumulated delta weights
   */
  override val dW: IndexedSeq[ScalarMatrix] = {
    val matrices = ArrayBuffer[ScalarMatrix]()
    encoders.flatMap(_.dW).foreach(matrices += _)
    matrices
  }

  /**
   * All weights of layers
   *
   * @return all weights of layers
   */
  override val W: IndexedSeq[ScalarMatrix] = {
    val matrices = ArrayBuffer[ScalarMatrix]()
    encoders.flatMap(_.W).foreach(matrices += _)
    matrices
  }

  /**
   * Serialize network to JSON
   *
   * @return JsObject of this network
   */
  override def toJSON: JsObject =
    Json.obj(
      "type" → this.getClass.getSimpleName,
      "stack" → Json.arr(encoders map (_.toJSON))
    )

  /**
   * Compute output of neural network with given input (without reconstruction)
   * If drop-out is used, to average drop-out effect, we need to multiply output by presence probability.
   *
   * @param in an input vector
   * @return output of the vector
   */
  override def apply(in: ScalarMatrix): ScalarMatrix = {
    var i = 0
    var x = in
    while (i < encoders.size) {
      x = encoders(i)(x)
      i += 1
    }
    x
  }

  /**
   * Sugar: Forward computation for training. Calls apply(x)
   *
   * @param x input matrix
   * @return output matrix
   */
  override def into_:(x: ScalarMatrix): ScalarMatrix = {
    var i = 0
    var in = x
    while (i < encoders.size) {
      in = in into_: encoders(i)
      i += 1
    }
    in
  }

  /**
   * Backpropagation algorithm
   *
   * @param err backpropagated error from error function
   */
  override def updateBy(err: ScalarMatrix): ScalarMatrix = {
    var i = encoders.size - 1
    var x = err
    while (i >= 0) {
      x = encoders(i) updateBy err
      i -= 1
    }
    x
  }
}


