package kr.ac.kaist.ir.deep.train.op

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.fn.obj.{Objective, SquaredErr}
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.rec.VectorTree
import kr.ac.kaist.ir.deep.train._
import org.apache.spark.annotation.AlphaComponent

/**
 * __Input Operation__ : VectorTree as Input & General Recursive Network Training
 *
 * @note This is designed for Recursive Neural Tensor Network, in 
 *       [[http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf this paper]]
 *
 * @param corrupt Corruption that supervises how to corrupt the input matrix. `(Default : [[kr.ac.kaist.ir.deep.train.NoCorruption]])`
 * @param error An objective function `(Default: [[kr.ac.kaist.ir.deep.fn.obj.SquaredErr]])`
 *
 * @example
 * {{{var make = new TreeRecursive(error = CrossEntropyErr)
 *  var corruptedIn = make corrupted in
 *  var out = make onewayTrip (net, corruptedIn)}}}
 */
@AlphaComponent
class TreeRecursive(override protected[train] val corrupt: Corruption = NoCorruption,
                    override protected[train] val error: Objective = SquaredErr)
  extends ScalarTree(corrupt, error) {

  /**
   * Apply & Back-prop given single input
   *
   * @param net A network that gets input
   * @param in __corrupted__ input
   * @param real __Real label__ for comparing
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
}
