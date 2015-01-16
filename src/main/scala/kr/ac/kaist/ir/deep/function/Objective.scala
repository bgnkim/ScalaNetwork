package kr.ac.kaist.ir.deep.function

/**
 * Define Objective Function trait.
 */
trait Objective extends ((ScalarMatrix, ScalarMatrix) ⇒ Scalar) with Serializable {
  /**
   * Compute derivative of this objective function
   * @param real is expected real output
   * @param output is computational output of the network
   * @return differentiation value at f(x)=fx, which is Square, diagonal matrix
   */
  def derivative(real: ScalarMatrix, output: ScalarMatrix): ScalarMatrix

  /**
   * Compute error
   * @param real is expected real output
   * @param output is computational output of the network
   * @return is the error
   */
  override def apply(real: ScalarMatrix, output: ScalarMatrix): Scalar
}

