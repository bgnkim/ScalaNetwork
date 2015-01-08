package kr.ac.kaist.ir.deep.function

import breeze.linalg.{DenseMatrix, sum}

/**
 * Objective Function: Sum of Cross-Entropy (Logistic)
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
   * Compute derivative of this objective function
   * @param real is expected real output
   * @param output is computational output of the network
   * @return is differentiation(Gradient) vector at f(X)=output, i.e. error of each output neuron.
   */
  override def derivative(real: ScalarMatrix, output: ScalarMatrix): ScalarMatrix =
    DenseMatrix.tabulate(real.rows, real.cols)((r, c) ⇒ entropyDiff(real(r, c), output(r, c)))

  /**
   * Compute error
   * @param real is expected real output
   * @param output is computational output of the network
   * @return is the error
   */
  override def apply(real: ScalarMatrix, output: ScalarMatrix): Scalar =
    sum(DenseMatrix.tabulate(real.rows, real.cols)((r, c) ⇒ entropy(real(r, c), output(r, c))))
}
