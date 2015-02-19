package kr.ac.kaist.ir.deep.fn

import breeze.numerics._

/**
 * __Activation Function__: Softplus
 *
 * @note `softplus(x) = log[1 + exp(x)]`
 * @example
 * {{{val fx = Softplus(0.0)
 *        val diff = Softplus.derivative(fx)}}}
 */
object Softplus extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, diagonal matrix__
   */
  override def derivative(fx: ScalarMatrix): ScalarMatrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = ScalarMatrix $0(fx.rows, fx.rows)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    var r = 0
    while (r < fx.rows) {
      val expx = exp(fx(r, 0))
      res.update((r, r), (expx - 1.0f) / expx)
      r += 1
    }
    res
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val expx = exp(x)
    val plus1 = expx :+ 1.0f
    log(plus1)
  }
}