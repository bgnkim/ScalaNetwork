package kr.ac.kaist.ir.deep.function

/**
 * Define Objective Function trait.
 */
trait Objective extends ((ScalarMatrix, ScalarMatrix) ⇒ Scalar) with Serializable {
  /**
   * Compute derivative of this objective function
   * @param real is expected real output
   * @param output is computational output of the network
   * @return is differentiation(Gradient) vector at f(X)=output, i.e. error of each output neuron.
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

