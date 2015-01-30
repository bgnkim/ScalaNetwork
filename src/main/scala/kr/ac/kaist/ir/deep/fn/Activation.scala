package kr.ac.kaist.ir.deep.fn

/**
 * __Trait__ that describes an activation function for '''each layer'''
 *
 * Because these activation functions can be shared, we recommend to make inherited one as an object.
 */
trait Activation extends (ScalarMatrix ⇒ ScalarMatrix) with Serializable {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, diagonal matrix__
   */
  def derivative(fx: ScalarMatrix): ScalarMatrix

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  def apply(x: ScalarMatrix): ScalarMatrix

  /**
   * Initialize the weight matrix
   *
   * @param fanIn the number of __fan-in__ ''i.e. the number of neurons in previous layer''
   * @param fanOut the number of __fan-out__ ''i.e. the number of neurons in next layer''
   * @param rows the number of __rows of resulting matrix__ `(Default 0)`
   * @param cols the number of __cols of resulting matrix__ `(Default 0)`
   * @return the initialized weight matrix
   */
  def initialize(fanIn: Int, fanOut: Int, rows: Int = 0, cols: Int = 0): ScalarMatrix = {
    val pmMatx: ScalarMatrix = ScalarMatrix.of(if (rows > 0) rows else fanOut, if (cols > 0) cols else fanIn) :* 1e-2
    pmMatx :+ 1e-2
  }
}