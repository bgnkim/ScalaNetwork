package kr.ac.kaist.ir.deep.function

/**
 * Define Activation Function trait.
 */
trait Activation extends (ScalarMatrix ⇒ ScalarMatrix) with Serializable {
  /**
   * Compute derivative of this function
   * @param fx is output of this function
   * @return differentiation value at f(x)=fx, which is Square, diagonal matrix
   */
  def derivative(fx: ScalarMatrix): ScalarMatrix

  /**
   * Compute mapping at x
   * @param x is input scalar.
   * @return f(x)
   */
  def apply(x: ScalarMatrix): ScalarMatrix

  /**
   * Initialize Weight matrix
   * @param fanIn is a weight vector indicates fan-in
   * @param fanOut is a count of fan-out
   * @return weight matrix
   */
  def initialize(fanIn: Int, fanOut: Int, rows: Int = 0, cols: Int = 0): ScalarMatrix = {
    val pmMatx: ScalarMatrix = ScalarMatrix.of(if (rows > 0) rows else fanOut, if (cols > 0) cols else fanIn) :* 1e-2
    pmMatx :+ 1e-2
  }
}