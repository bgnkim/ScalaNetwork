package kr.ac.kaist.ir.deep.function

/**
 * Activation Function: Rectifier
 *
 * Rectifier(x) = x if x > 0, otherwise 0
 */
object Rectifier extends Activation {
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
      val x = fx(r, 0)
      res.update((r, r), if (x > 0) 1.0 else 0.0)
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
    val res = x.copy
    x foreachKey { key ⇒ if (x(key) < 0) res.update(key, 0.0)}
    res
  }
}