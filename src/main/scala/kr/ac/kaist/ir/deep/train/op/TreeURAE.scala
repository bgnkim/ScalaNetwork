package kr.ac.kaist.ir.deep.train.op

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.fn.obj.{Objective, SquaredErr}
import kr.ac.kaist.ir.deep.network.{AutoEncoder, Network}
import kr.ac.kaist.ir.deep.rec.VectorTree
import kr.ac.kaist.ir.deep.train._
import org.apache.spark.annotation.AlphaComponent

/**
 * __Input Operation__ : VectorTree as Input & Unfolding Recursive Auto Encoder Training
 *
 * @note This cannot be applied into non-AutoEncoder tasks
 * @note This is designed for Unfolding RAE, in
 *       [[http://ai.stanford.edu/~ang/papers/nips11-DynamicPoolingUnfoldingRecursiveAutoencoders.pdf this paper]]
 *
 * @param corrupt Corruption that supervises how to corrupt the input matrix. `(Default : [[kr.ac.kaist.ir.deep.train.NoCorruption]])`
 * @param error An objective function `(Default: [[kr.ac.kaist.ir.deep.fn.obj.SquaredErr]])`
 *
 * @example
 * {{{var make = new TreeURAE(error = CrossEntropyErr)
 *  var corruptedIn = make corrupted in
 *  var out = make onewayTrip (net, corruptedIn)}}}
 */
@AlphaComponent
class TreeURAE(override protected[train] val corrupt: Corruption = NoCorruption,
               override protected[train] val error: Objective = SquaredErr)
  extends ScalarTree(corrupt, error) {

  /**
   * Apply & Back-prop given single input
   *
   * @param net A network that gets input
   * @param in __corrupted__ input
   * @param real __Real label__ for comparing
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
}
