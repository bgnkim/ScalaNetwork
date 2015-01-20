package kr.ac.kaist.ir.deep.train.op

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.fn.obj.Objective
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.train.Corruption

/**
 * __Trait__ that describes how to convert input into corrupted matrix
 *
 * Input operation corrupts the given input, and apply network propagations onto matrix representation of input 
 *
 * @tparam IN the type of input
 */
trait InputOp[IN] extends Serializable {
  /** Corruption function */
  protected[train] val corrupt: Corruption
  /** Objective function */
  protected[train] val error: Objective

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
   * @param in __corrupted__ input
   * @param real __Real label__ for comparing
   */
  def roundTrip(net: Network, in: IN, real: ScalarMatrix): Unit

  /**
   * Apply given single input as one-way forward trip.
   *
   * @param net A network that gets input
   * @param x input to be computed            
   * @return output of the network.
   */
  def onewayTrip(net: Network, x: IN): ScalarMatrix

  /**
   * Make input to string
   *
   * @return input as string
   */
  def stringOf(in: (IN, ScalarMatrix)): String
}
