package kr.ac.kaist.ir.deep.fn.act

import breeze.linalg.DenseMatrix
import kr.ac.kaist.ir.deep.fn._

/**
 * __Activation Function__: Linear
 *
 * @note `linear(x) = x`
 * @example
  * {{{val fx = Linear(0.0)
 *       val diff = Linear.derivative(fx)}}}
 */
object Linear extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, diagonal matrix__
   */
  override def derivative(fx: ScalarMatrix): ScalarMatrix = DenseMatrix.eye[Double](fx.rows)

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = x.copy
}