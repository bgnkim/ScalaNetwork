package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network.{AutoEncoder, Network}
import kr.ac.kaist.ir.deep.rec.DAG
import org.apache.spark.annotation.AlphaComponent

/**
 * __Input Operation__ : VectorTree as Input & Unfolding Recursive Auto Encoder Training (no output type)
 *
 * ::Experimental::
 * @note This cannot be applied into non-AutoEncoder tasks
 * @note This is designed for Unfolding RAE, in
 *       [[http://ai.stanford.edu/~ang/papers/nips11-DynamicPoolingUnfoldingRecursiveAutoencoders.pdf this paper]]
 *
 * @param corrupt Corruption that supervises how to corrupt the input matrix. `(Default : [[kr.ac.kaist.ir.deep.train.NoCorruption]])`
 * @param error An objective function `(Default: [[kr.ac.kaist.ir.deep.fn.SquaredErr]])`
 *
 * @example
 * {{{var make = new URAEType(error = CrossEntropyErr)
 *            var corruptedIn = make corrupted in
 *            var out = make onewayTrip (net, corruptedIn)}}}
 */
@AlphaComponent
class URAEType(override protected[train] val corrupt: Corruption = NoCorruption,
               override protected[train] val error: Objective = SquaredErr)
  extends DAGType {

  /**
   * Apply & Back-prop given single input
   *
   * @param net A network that gets input
   * @param seq Sequence of (Input, Real output) for error computation.
   */
  def roundTrip(net: Network, seq: Seq[(DAG, Null)]): Unit =
    net match {
      case net: AutoEncoder ⇒
        seq foreach {
          pair ⇒
            val in = pair._1
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


  /**
   * Apply given input and compute the error
   *
   * @param net A network that gets input  
   * @param validation Sequence of (Input, Real output) for error computation.
   * @return error of this network
   */
  override def lossOf(net: Network, validation: Seq[(DAG, Null)]): Scalar =
    net match {
      case net: AutoEncoder ⇒
        validation.map {
          pair ⇒
            val in = pair._1
            // Encode phrase of Reconstruction
            val out = in forward net.apply

            // Decode phrase of reconstruction
            in.backward(out, net.reconstruct).map {
              leaf ⇒ error(leaf.out, leaf.x)
            }.sum
        }.sum
      case _ ⇒ 0.0
    }


  /**
   * Make validation output
   *
   * @return input as string
   */
  def stringOf(net: Network, pair: (DAG, Null)): String =
    net match {
      case net: AutoEncoder ⇒
        val string = StringBuilder.newBuilder
        val in = pair._1
        // Encode phrase of Reconstruction
        val out = in forward net.apply

        // Decode phrase of reconstruction
        in.backward(out, net.reconstruct).foreach {
          leaf ⇒
            string append s"IN: ${leaf.x.mkString} URAE → OUT: ${leaf.out.mkString};"
        }
        string.mkString
      case _ ⇒ "NOT AN AUTOENCODER"
    }
}
