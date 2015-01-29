package kr.ac.kaist.ir.deep.train.op

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.fn.obj.{Objective, SquaredErr}
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.rec.DAG
import kr.ac.kaist.ir.deep.train._
import org.apache.spark.annotation.AlphaComponent

/**
 * __Input Operation__ : VectorTree as Input & Recursive Auto-Encoder Training
 *
 * @note We recommend that you should not apply this method to non-AutoEncoder tasks
 * @note This implementation designed as a replica of the traditional RAE in
 *       [[http://ai.stanford.edu/~ang/papers/emnlp11-RecursiveAutoencodersSentimentDistributions.pdf this paper]]
 *
 * @param corrupt Corruption that supervises how to corrupt the input matrix. `(Default : [[kr.ac.kaist.ir.deep.train.NoCorruption]])`
 * @param error An objective function `(Default: [[kr.ac.kaist.ir.deep.fn.obj.SquaredErr]])`
 *
 * @example
 * {{{var make = new TreeRAE(error = CrossEntropyErr)
 *  var corruptedIn = make corrupted in
 *  var out = make onewayTrip (net, corruptedIn)}}}
 */
@AlphaComponent
class TreeRAE(override protected[train] val corrupt: Corruption = NoCorruption,
              override protected[train] val error: Objective = SquaredErr)
  extends ScalarTree(corrupt, error) {

  /**
   * Apply & Back-prop given single input
   *
   * @param net A network that gets input
   * @param in __corrupted__ input
   * @param real __Real label__ for comparing
   */
  override def roundTrip(net: Network, in: DAG, real: ScalarMatrix): Unit =
    in forward {
      x ⇒
        val out = x >>: net
        val err = error.derivative(x, out)
        net ! err
        out
    }
}
