package kr.ac.kaist.ir.deep.fn

/**
 * __Activation Function__: Hard version of Tanh (Hyperbolic Tangent)
 *
 * @note `tanh(x) = sinh(x) / cosh(x)`, hard version approximates tanh as piecewise linear function.
 * @example
 * {{{val fx = HardTanh(0.0)
 *         val diff = HardTanh.derivative(fx) }}}
 */
object HardTanh extends Activation {
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
      val x = fx(r, 0)
      res.update((r, r), if (x == 1.0f || x == -1.0f) 0.0f else 1.0f)
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
    val res = x.copy
    val iter = x.keysIterator
    while (iter.hasNext) {
      val key = iter.next()
      val v = x(key)
      if (v < -1) res.update(key, -1.0f)
      else if (v > 1) res.update(key, 1.0f)
    }
    res
  }
}