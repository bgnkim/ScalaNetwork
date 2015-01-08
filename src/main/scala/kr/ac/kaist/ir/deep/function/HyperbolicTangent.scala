package kr.ac.kaist.ir.deep.function

import breeze.numerics._

/**
 * Activation Function: Tanh
 */
object HyperbolicTangent extends Activation {
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
      res.update((r, r), 1.0 - x * x)
    }
    }
    res
  }

  /**
   * Compute mapping at x
   * @param x is input scalar.
   * @return f(x)
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = tanh(x)

  /**
   * Initialize Weight matrix
   * @param fanIn is a weight vector indicates fan-in
   * @param fanOut is a count of fan-out
   * @return weight matrix
   */
  override def initialize(fanIn: Int, fanOut: Int, rows: Int = 0, cols: Int = 0): ScalarMatrix = {
    val range = Math.sqrt(6.0 / (fanIn + fanOut))
    val pmMatx: ScalarMatrix = ScalarMatrix.of(if (rows > 0) rows else fanOut, if (cols > 0) cols else fanIn) :- 0.5
    pmMatx :* (2.0 * range)
  }
}