package kr.ac.kaist.ir.deep.fn

import breeze.linalg.{DenseMatrix, sum}

/**
 * __Objective Function__: Sum of Cross-Entropy (Logistic)
 *
 * @note This objective function prefer 0/1 output
 * @example
 * {{{val output = net(input)
 *        val err = CrossEntropyErr(real, output)
 *        val diff = CrossEntropyErr.derivative(real, output)
 * }}}
 */
object CrossEntropyErr extends Objective {
  /**
   * Entropy function
   */
  val entropy = (r: Scalar, o: Scalar) ⇒
    (if (r != 0.0f) -r * Math.log(o).toFloat else 0.0f) + (if (r != 1.0f) -(1.0f - r) * Math.log(1.0f - o).toFloat else 0.0f)

  /**
   * Derivative of Entropy function
   */
  val entropyDiff = (r: Scalar, o: Scalar) ⇒ (r - o) / (o * (o - 1.0f))

  /**
   * Compute differentiation value of this objective function at `x = r - o`
   *
   * @param real the expected __real output__, `r`
   * @param output the computed __output of the network__, `o`
   * @return differentiation value at `f(x)=fx`, which is __a column vector__
   */
  override def derivative(real: ScalarMatrix, output: ScalarMatrix): ScalarMatrix =
    DenseMatrix.tabulate(real.rows, real.cols)((r, c) ⇒ entropyDiff(real(r, c), output(r, c)))

  /**
   * Compute error (loss)
   *
   * @param real the expected __real output__
   * @param output the computed __output of the network__
   * @return the error
   */
  override def apply(real: ScalarMatrix, output: ScalarMatrix): Scalar =
    sum(DenseMatrix.tabulate(real.rows, real.cols)((r, c) ⇒ entropy(real(r, c), output(r, c))))
}
