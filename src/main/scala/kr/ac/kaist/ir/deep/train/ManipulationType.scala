package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn.{Objective, Scalar, ScalarMatrix}
import kr.ac.kaist.ir.deep.network.Network

/**
 * __Trait__ that describes how to convert input into corrupted matrix
 *
 * Input operation corrupts the given input, and apply network propagations onto matrix representation of input 
 *
 * @tparam IN the type of input
 * @tparam OUT the type of output
 */
trait ManipulationType[IN, OUT] extends Serializable {
  /** Corruption function */
  val corrupt: Corruption
  /** Objective function */
  val error: Objective

  // We didn't assign a "network" value, because of dist-belief training style.

  /**
   * Corrupt input
   *
   * @param x input to be corrupted 
   * @return corrupted input
   */
  def corrupted(x: IN): IN

  /**
   * Apply & Back-prop given single input
   *
   * @param net A network that gets input
   * @param in Input for error computation.
   * @param real Real output for error computation.
   * @param isPositive Boolean that indicates whether this example is positive or not.             
   */
  def roundTrip(net: Network, in: IN, real: OUT, isPositive: Boolean = true): Unit

  /**
   * Apply given single input as one-way forward trip.
   *
   * @param net A network that gets input
   * @param x input to be computed
   * @return output of the network.
   */
  def onewayTrip(net: Network, x: IN): ScalarMatrix

  /**
   * Make validation output
   *
   * @param net A network that gets input
   * @param in (Input, Real output) pair for computation
   * @return input as string
   */
  def stringOf(net: Network, in: (IN, OUT)): String

  /**
   * Apply given input and compute the error
   *
   * @param net A network that gets input  
   * @param pair (Input, Real output) for error computation.
   * @return error of this network
   */
  def lossOf(net: Network)(pair: (IN, OUT)): Scalar

  /**
   * Check whether given two are same or not.
   * @param x Out-type object
   * @param y Out-type object
   * @return True if they are different.
   */
  def different(x: OUT, y: OUT): Boolean = true
}
