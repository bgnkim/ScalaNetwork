package kr.ac.kaist.ir.deep.train.op

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.fn.obj.Objective
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.train.Corruption

/**
 * Corrupt and apply given single input
 */
trait InputOp[IN] extends Serializable {
  /** Corruption function */
  protected[train] val corrupt: Corruption
  /** Objective function */
  protected[train] val error: Objective

  /**
   * Corrupt input
   * @return corrupted input
   */
  def corrupted(x: IN): IN

  /**
   * Apply & Back-prop given single input
   * @param net that gets input
   */
  def roundTrip(net: Network, in: IN, real: ScalarMatrix): Unit

  /**
   * Apply given single input
   * @param net that gets input
   * @return output of the network.
   */
  def onewayTrip(net: Network, x: IN): ScalarMatrix

  /**
   * Make input to string
   * @return input as string
   */
  def stringOf(in: (IN, ScalarMatrix)): String
}
