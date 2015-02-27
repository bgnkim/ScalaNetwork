package kr.ac.kaist.ir.deep.fn

/**
 * __Activation Function__: Hard version of Sigmoid
 *
 * @note `sigmoid(x) = 1 / [exp(-x) + 1]`, hard version approximates tanh as piecewise linear function 
 *       (derived from relationship between tanh & sigmoid, and tanh & hard tanh.) 
 * @example
 * {{{val fx = HardSigmoid(0.0)
 *         val diff = HardSigmoid.derivative(fx) }}}
 */
object HardSigmoid extends Activation {
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
      res.update((r, r), if (x == 0.0f || x == 1.0f) 0.0f else 0.25f)
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
      if (v < -2) res.update(key, 0.0f)
      else if (v > 2) res.update(key, 1.0f)
      else res.update(key, 0.25f * v + 0.5f)
    }
    res
  }
}