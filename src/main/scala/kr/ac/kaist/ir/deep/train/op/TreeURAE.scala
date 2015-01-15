package kr.ac.kaist.ir.deep.train.op

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.rec.{Tree, VectorTree}
import kr.ac.kaist.ir.deep.train._

import scala.annotation.tailrec

/**
 * Input Operation : For VectorTree, & Unfolding Recursive Auto Encoder Training
 *
 * <p>
 * '''Note''' : This cannot be applied into non-AutoEncoder tasks
 * </p> 
 *
 * @param corrupt supervises how to corrupt the input matrix. (Default : [[NoCorruption]])
 * @param error is an objective function (Default: [[SquaredErr]])
 */
class TreeURAE(override protected[train] val corrupt: Corruption = NoCorruption,
               override protected[train] val error: Objective = SquaredErr)
  extends InputOp[VectorTree] {

  @tailrec
  final def breathFirstOrdering(seq: Seq[VectorTree], out: Seq[VectorTree] = Seq()): Seq[VectorTree] =
    if (seq.isEmpty)
      out
    else
      breathFirstOrdering(seq.tail ++ seq.head.children, out ++ seq.head.children)

  @tailrec
  final def ordering[A, B](seq: Seq[Tree[A]], out: Seq[Tree[B]], fn: Tree[A] ⇒ Tree[B]): Seq[Tree[B]] =
    if (seq.nonEmpty) {
      val tree = fn(seq.head)
      ordering(seq.tail, tree +: out, fn)
    } else
      out

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
    //TODO
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
