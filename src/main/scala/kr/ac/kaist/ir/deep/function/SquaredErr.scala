package kr.ac.kaist.ir.deep.function

import breeze.linalg.sum

/**
 * Objective Function: Sum of Square Error
 */
object SquaredErr extends Objective {
  /**
   * Compute derivative of this objective function
   * @param real is expected real output
   * @param output is computational output of the network
   * @return is differentiation(Gradient) vector at f(X)=output, i.e. error of each output neuron.
   */
  override def derivative(real: ScalarMatrix, output: ScalarMatrix): ScalarMatrix = output - real

  /**
   * Compute error
   * @param real is expected real output
   * @param output is computational output of the network
   * @return is the error
   */
  override def apply(real: ScalarMatrix, output: ScalarMatrix): Scalar = {
    val diff: ScalarMatrix = real - output
    val sqdiff: ScalarMatrix = diff :^ 2.0
    0.5 * sum(sqdiff)
  }
}
