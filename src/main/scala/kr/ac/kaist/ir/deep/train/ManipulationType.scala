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
  protected[train] val corrupt: Corruption
  /** Objective function */
  protected[train] val error: Objective

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
   * @param seq Sequence of (Input, Real output) for error computation.
   */
  def roundTrip(net: Network, seq: Seq[(IN, OUT)]): Unit

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
   * @return input as string
   */
  def stringOf(net: Network, in: (IN, OUT)): String

  /**
   * Apply given input and compute the error
   *
   * @param net A network that gets input  
   * @param validation Sequence of (Input, Real output) for error computation.
   * @return error of this network
   */
  def lossOf(net: Network, validation: Seq[(IN, OUT)]): Scalar
}
