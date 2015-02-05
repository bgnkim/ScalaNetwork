package kr.ac.kaist.ir.deep.fn

import breeze.linalg.sum

/**
 * __Objective Function__: Sum of Squared Error
 *
 * @example
 * {{{val output = net(input)
 *        val err = SquaredErr(real, output)
 *        val diff = SquaredErr.derivative(real, output)
 * }}}
 */
object SquaredErr extends Objective {
  /**
   * Compute differentiation value of this objective function at `x = r - o`
   *
   * @param real the expected __real output__, `r`
   * @param output the computed __output of the network__, `o`
   * @return differentiation value at `f(x)=fx`, which is __a column vector__
   */
  override def derivative(real: ScalarMatrix, output: ScalarMatrix): ScalarMatrix = output - real

  /**
   * Compute error (loss)
   *
   * @param real the expected __real output__
   * @param output the computed __output of the network__
   * @return the error
   */
  override def apply(real: ScalarMatrix, output: ScalarMatrix): Scalar = {
    val diff = real - output
    val sqdiff = diff :^ 2.0
    sum(sqdiff)
  }
}
