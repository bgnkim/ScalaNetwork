package kr.ac.kaist.ir.deep.layer

import kr.ac.kaist.ir.deep.fn._

/**
 * __Trait__ of Layer that can be used for autoencoder
 */
trait Reconstructable extends Layer {
  protected var decX: ScalarMatrix = _
  protected var decdFdX: ScalarMatrix = _
  /**
   * Reconstruction
   *
   * @param x hidden layer output matrix
   * @return tuple of reconstruction output
   */
  def decodeFrom(x: ScalarMatrix): ScalarMatrix

  /**
   * Sugar: reconstruction
   *
   * @param x hidden layer output matrix
   * @return tuple of reconstruction output
   */
  def decodeBy(x: ScalarMatrix): ScalarMatrix = {
    decX = x
    val out = decodeFrom(x)
    decdFdX = act.derivative(out)
    out
  }

  /**
   * Sugar: reconstruction
   *
   * @param x hidden layer output matrix
   * @return tuple of reconstruction output
   */
  @deprecated
  def decodeBy_:(x: ScalarMatrix): ScalarMatrix = decodeBy(x)

  /**
   * Backpropagation of reconstruction. For the information about backpropagation calculation, see [[kr.ac.kaist.ir.deep.layer.Layer]]
   *
   * @param delta Sequence of delta amount of weight. The order must be the reverse of [[W]]
   * @param error error matrix to be propagated
   * @return propagated error
   */
  protected[deep] def decodeUpdateBy(delta: Iterator[ScalarMatrix], error: ScalarMatrix): ScalarMatrix
}
