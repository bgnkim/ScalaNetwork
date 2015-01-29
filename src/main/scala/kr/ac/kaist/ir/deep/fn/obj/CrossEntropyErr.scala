package kr.ac.kaist.ir.deep.fn.obj

import breeze.linalg.{DenseMatrix, sum}
import kr.ac.kaist.ir.deep.fn._

/**
 * __Objective Function__: Sum of Cross-Entropy (Logistic)
 *
 * @note This objective function prefer 0/1 output
 * @example
 * {{{val output = net(input)
 *   val err = CrossEntropyErr(real, output)
 *   val diff = CrossEntropyErr.derivative(real, output)
 * }}}
 */
object CrossEntropyErr extends Objective {
  /**
   * Entropy function
   */
  val entropy = (r: Scalar, o: Scalar) ⇒ (if (r != 0.0) -r * Math.log(o) else 0.0) + (if (r != 1.0) -(1.0 - r) * Math.log(1.0 - o) else 0.0)

  /**
   * Derivative of Entropy function
   */
  val entropyDiff = (r: Scalar, o: Scalar) ⇒ (r - o) / (o * (o - 1.0))

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
