package kr.ac.kaist.ir.deep.train.op

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.fn.obj.{Objective, SquaredErr}
import kr.ac.kaist.ir.deep.network.{AutoEncoder, Network}
import kr.ac.kaist.ir.deep.rec.DAG
import kr.ac.kaist.ir.deep.train._
import org.apache.spark.annotation.AlphaComponent

/**
 * __Input Operation__ : VectorTree as Input & Unfolding Recursive Auto Encoder Training
 *
 * ::Experimental::
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
  override def roundTrip(net: Network, in: DAG, real: ScalarMatrix): Unit =
    net match {
      case net: AutoEncoder ⇒
        // Encode phrase of Reconstruction
        val out = in forward net.encode

        // Decode phrase of reconstruction
        in backward(out, net.decode) map {
          leaf ⇒
            leaf.out = error.derivative(leaf.out, leaf.x)
        }

        // Error propagation for decoder
        val err = in forward net.decode_!
        
        // Error propagation for encoder
        in backward(err, net.encode_!)
    }
}
