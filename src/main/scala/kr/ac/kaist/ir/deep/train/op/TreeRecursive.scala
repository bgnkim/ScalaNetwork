package kr.ac.kaist.ir.deep.train.op

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.rec.VectorTree
import kr.ac.kaist.ir.deep.train._

/**
 * Input Operation : For VectorTree, & General Recursive Network Training
 *
 * @param corrupt supervises how to corrupt the input matrix. (Default : [[NoCorruption]])
 * @param error is an objective function (Default: [[SquaredErr]])
 */
class TreeRecursive(override protected[train] val corrupt: Corruption = NoCorruption,
                    override protected[train] val error: Objective = SquaredErr)
  extends InputOp[VectorTree] {

  /**
   * Corrupt input
   * @return corrupted input
   */
  override def corrupted(x: VectorTree): VectorTree = x ? corrupt

  /**
   * Apply & Back-prop given single input
   * @param net that gets input
   */
  override def roundTrip(net: Network, in: VectorTree, real: ScalarMatrix): Unit = {
    val out = in postOrder {
      (v1, v2) ⇒
        val x = v1 row_+ v2
        x >>: net
    }
    val err = error.derivative(real, out)
    in preOrder(err, { e ⇒ net ! e})
  }

  /**
   * Apply given single input
   * @param net that gets input
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
   * @return input as string
   */
  override def stringOf(in: (VectorTree, ScalarMatrix)): String =
    "IN: Tree, Out: " + in._2.mkString

}
