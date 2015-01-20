package kr.ac.kaist.ir.deep.train.op

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.fn.obj.{Objective, SquaredErr}
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.rec._
import kr.ac.kaist.ir.deep.train.{Corruption, NoCorruption}

/**
 * __Input Operation__ : VectorTree as Input. This is an '''Abstract Implementation'''
 *
 * @param corrupt Corruption that supervises how to corrupt the input matrix. `(Default : [[kr.ac.kaist.ir.deep.train.NoCorruption]])`
 * @param error An objective function `(Default: [[kr.ac.kaist.ir.deep.fn.obj.SquaredErr]])`
 */
abstract class ScalarTree(override protected[train] val corrupt: Corruption = NoCorruption,
                          override protected[train] val error: Objective = SquaredErr)
  extends InputOp[VectorTree] {

  /**
   * Corrupt input
   *
   * @param x input to be corrupted
   * @return corrupted input
   */
  override def corrupted(x: VectorTree): VectorTree = x ? corrupt

  /**
   * Apply given single input as one-way forward trip.
   *
   * @param net A network that gets input
   * @param x input to be computed
   * @return output of the network.
   */
  override def onewayTrip(net: Network, x: VectorTree): ScalarMatrix =
    x postOrder {
      (v1, v2) ⇒
        val in = v1 row_+ v2
        net on in
    }

  /**
   * Make input to string
   *
   * @return input as string
   */
  override def stringOf(in: (VectorTree, ScalarMatrix)): String =
    "IN: Tree, Out: " + in._2.mkString

}
