package kr.ac.kaist.ir.deep.fn

import breeze.numerics._

/**
 * __Activation Function__: Tanh (Hyperbolic Tangent)
 *
 * @note `tanh(x) = sinh(x) / cosh(x)`
 * @example
 * {{{val fx = HyperbolicTangent(0.0)
 *        val diff = HyperbolicTangent.derivative(fx) }}}
 */
object HyperbolicTangent extends Activation {
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
      res.update((r, r), 1.0f - x * x)
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
  override def apply(x: ScalarMatrix): ScalarMatrix = tanh(x)

  /**
   * Initialize the weight matrix
   *
   * @param fanIn the number of __fan-in__ ''i.e. the number of neurons in previous layer''
   * @param fanOut the number of __fan-out__ ''i.e. the number of neurons in next layer''
   * @param rows the number of __rows of resulting matrix__ `(Default 0)`
   * @param cols the number of __cols of resulting matrix__ `(Default 0)`
   * @return the initialized weight matrix
   */
  override def initialize(fanIn: Int, fanOut: Int, rows: Int = 0, cols: Int = 0): ScalarMatrix = {
    val range = Math.sqrt(6.0 / (fanIn + fanOut)).toFloat
    val pmMatx: ScalarMatrix = ScalarMatrix.of(if (rows > 0) rows else fanOut, if (cols > 0) cols else fanIn) :- 0.5f
    pmMatx :* (2.0f * range)
  }
}