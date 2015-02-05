package kr.ac.kaist.ir.deep.fn

/**
 * __Activation Function__: Rectifier
 *
 * @note `rectifier(x) = x if x > 0, otherwise 0`
 * @example
 * {{{val fx = Rectifier(0.0)
 *        val diff = Rectifier.derivative(fx)}}}
 */
object Rectifier extends Activation {
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
    (0 until fx.rows).par foreach {
      r ⇒
        val x = fx(r, 0)
        res.update((r, r), if (x > 0) 1.0 else 0.0)
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
    val res = x.copy
    x foreachKey { key ⇒ if (x(key) < 0) res.update(key, 0.0)}
    res
  }
}