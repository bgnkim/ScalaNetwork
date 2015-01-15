package kr.ac.kaist.ir.deep.network

import kr.ac.kaist.ir.deep.function._
import play.api.libs.json.JsObject

/**
 * Trait: Network interface
 */
trait Network extends (ScalarMatrix ⇒ ScalarMatrix) with Serializable {
  /**
   * Probability that each input neuron of each layer is not dropped (For neuron drop-out)
   */
  protected val presence: Probability

  /**
   * All weights of layers
   * @return all weights of layers
   */
  def W: Seq[ScalarMatrix]

  /**
   * All accumulated delta weights of layers
   * @return all accumulated delta weights
   */
  def dW: Seq[ScalarMatrix]

  /**
   * Serialize network to JSON
   * @return JsObject
   */
  def toJSON: JsObject

  /**
   * Backpropagation algorithm
   * @param err backpropagated error from error function
   */
  protected[deep] def !(err: ScalarMatrix): ScalarMatrix

  /**
   * Sugar: Forward computation for training. Calls apply(x)
   *
   * @param x of input matrix
   * @return output matrix
   */
  protected[deep] def >>:(x: ScalarMatrix): ScalarMatrix
}



