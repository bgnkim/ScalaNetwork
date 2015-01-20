package kr.ac.kaist.ir.deep.layer

import kr.ac.kaist.ir.deep.fn._

/**
 * Trait of Layer : Reconstructable
 */
trait Reconstructable extends Layer {
  /**
   * Sugar: Forward computation + reconstruction
   *
   * @param x of hidden layer output matrix
   * @return tuple of reconstruction output
   */
  def rec_>>:(x: ScalarMatrix): ScalarMatrix

  /**
   * Backpropagation of reconstruction. For the information about backpropagation calculation, see [[kr.ac.kaist.ir.deep.layer.Layer]]
   * @param error to be propagated
   * @param input of this layer
   * @param output is final reconstruction output of this layer
   * @return propagated error
   */
  protected[deep] def rec_!(error: ScalarMatrix, input: ScalarMatrix, output: ScalarMatrix): ScalarMatrix
}
