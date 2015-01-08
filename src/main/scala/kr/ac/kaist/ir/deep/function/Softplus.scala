package kr.ac.kaist.ir.deep.function

import breeze.numerics._

/**
 * Activation Function: Softplus
 *
 * Softplus(x) = log(1 + e ** x)
 */
object Softplus extends Activation {
  /**
   * Compute derivative of this function
   * @param fx is output of this function
   * @return differentiation value at f(x)=fx, which is Square, diagonal matrix
   */
  override def derivative(fx: ScalarMatrix): ScalarMatrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = ScalarMatrix $0(fx.rows, fx.rows)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    (0 until fx.rows) foreach { r ⇒ {
      val expx = exp(fx(r, 0))
      res.update((r, r), (expx - 1.0) / expx)
    }
    }
    res
  }

  /**
   * Compute mapping at x
   * @param x is input scalar.
   * @return f(x)
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val expx: ScalarMatrix = exp(x)
    val plus1: ScalarMatrix = expx :+ 1.0
    log(plus1)
  }
}