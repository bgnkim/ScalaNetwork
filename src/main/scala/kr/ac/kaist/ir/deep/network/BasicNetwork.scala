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
   * Compute output of neural network with given input
   * If drop-out is used, to average drop-out effect, we need to multiply output by presence probability.
   *
   * @param in an input vector
   * @return output of the vector
   */
  override def apply(in: ScalarMatrix): ScalarMatrix = {
    layers.foldLeft(in) {
      case (v, l) ⇒ l apply v
    }
  }

  /**
   * Serialize network to JSON
   *
   * @return JsObject of this network
   */
  override def toJSON = Json.obj(
    "type" → this.getClass.getSimpleName,
    "layers" → JsArray(layers map (_.toJSON))
  )

  /**
   * Backpropagation algorithm
   *
   * @param delta Sequence of delta amount of weight. The order must be the reverse of [[W]]
   * @param err backpropagated error from error function
   */
  override def updateBy(delta: Iterator[ScalarMatrix], err: ScalarMatrix): ScalarMatrix = {
    layers.foldRight(err) {
      case (l, e) ⇒ l updateBy(delta, e)
    }
  }

  /**
   * Forward computation for training.
   * If drop-out is used, we need to drop-out entry of input vector.
   *
   * @param x input matrix
   * @return output matrix
   */
  override def passedBy(x: ScalarMatrix): ScalarMatrix = {
    layers.foldLeft(x) {
      case (v, l) ⇒ l passedBy v
    }
  }
}

