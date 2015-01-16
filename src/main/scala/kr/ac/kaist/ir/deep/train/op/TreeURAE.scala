package kr.ac.kaist.ir.deep.train.op

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.network.{AutoEncoder, Network}
import kr.ac.kaist.ir.deep.rec.VectorTree
import kr.ac.kaist.ir.deep.train._

/**
 * Input Operation : For VectorTree, & Unfolding Recursive Auto Encoder Training
 *
 * <p>
 * '''Note''' : This cannot be applied into non-AutoEncoder tasks
 * </p> 
 *
 * @param corrupt supervises how to corrupt the input matrix. (Default : [[NoCorruption]])
 * @param error is an objective function (Default: [[kr.ac.kaist.ir.deep.function.SquaredErr]])
 */
class TreeURAE(override protected[train] val corrupt: Corruption = NoCorruption,
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
  override def roundTrip(net: Network, in: VectorTree, real: ScalarMatrix): Unit =
    net match {
      case net: AutoEncoder ⇒
        // Encode phrase of Reconstruction
        in postOrder {
          (v1, v2) ⇒
            val in = v1 row_+ v2
            net encode in
        }
        // Decode phrase of reconstruciton
        val leafErr = in preOrderCopy(in.value, {
          out ⇒ net decode out
        }, error.derivative)

        // Error propagation for decoder
        val err = leafErr postOrder {
          (e1, e2) ⇒
            val rows = e1.rows + e2.rows
            val err = ScalarMatrix $0(rows, rows)
            err(0 until e1.rows, 0 until e1.rows) := e1
            err(e1.rows to -1, e1.rows to -1) := e2
            net decode_! err
        }
        // Error propagation for encoder
        in preOrder(err, {
          err ⇒ net encode_! err
        })
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
