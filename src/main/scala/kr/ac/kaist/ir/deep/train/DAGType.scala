package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.rec._

/**
 * __Trait of Input Operation__ : VectorTree as Input. This is an '''Abstract Implementation'''
 */
trait DAGType extends ManipulationType[DAG, Null] {

  /**
   * Corrupt input
   *
   * @param x input to be corrupted
   * @return corrupted input
   */
  override def corrupted(x: DAG): DAG = x through corrupt

  /**
   * Apply given single input as one-way forward trip.
   *
   * @param net A network that gets input
   * @param x input to be computed
   * @return output of the network.
   */
  override def onewayTrip(net: Network, x: DAG): ScalarMatrix =
    x forward net.of

  /**
   * Make input to string
   *
   * @return input as string
   */
  override def stringOf(in: (DAG, Null)): String = "IN: DAG"

}
