package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network.{AutoEncoder, Network}
import kr.ac.kaist.ir.deep.rec.BinaryTree
import org.apache.spark.annotation.Experimental

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
@Experimental
class URAEType(override protected[train] val corrupt: Corruption = NoCorruption,
               override protected[train] val error: Objective = SquaredErr)
  extends TreeType {

  /**
   * Apply & Back-prop given single input
   *
   * @param net A network that gets input
   * @param in Input for error computation.
   * @param real Real Output for error computation.
   */
  def roundTrip(net: Network, in: BinaryTree, real: Null): Unit =
    net match {
      case net: AutoEncoder ⇒
        val out = in forward net.encode

        // Decode phrase of reconstruction
        var terminals = in.backward(out, net.decode)
        while (terminals.nonEmpty) {
          val leaf = terminals.head
          terminals = terminals.tail

          leaf.out = error.derivative(leaf.out, leaf.x)
        }

        // Error propagation for decoder
        val err = in forward net.decode_!

        // Error propagation for encoder
        in backward(err, net.encode_!)
    }


  /**
   * Apply given input and compute the error
   *
   * @param net A network that gets input
   * @param pair (Input, Real output) for error computation.
   * @return error of this network
   */
  def lossOf(net: Network)(pair: (BinaryTree, Null)): Scalar =
    net match {
      case net: AutoEncoder ⇒
        var sum = 0.0f
        val in = pair._1
        // Encode phrase of Reconstruction
        val out = in forward net.apply

        // Decode phrase of reconstruction
        var terminals = in.backward(out, net.reconstruct)
        val size = terminals.size
        while (terminals.nonEmpty) {
          val leaf = terminals.head
          terminals = terminals.tail
          sum += error(leaf.out, leaf.x)
        }
        sum
      case _ ⇒ 0.0f
    }


  /**
   * Make validation output
   *
   * @return input as string
   */
  def stringOf(net: Network, pair: (BinaryTree, Null)): String =
    net match {
      case net: AutoEncoder ⇒
        val string = StringBuilder.newBuilder
        val in = pair._1
        // Encode phrase of Reconstruction
        val out = in forward net.apply

        // Decode phrase of reconstruction
        var terminals = in.backward(out, net.reconstruct)
        while (terminals.nonEmpty) {
          val leaf = terminals.head
          terminals = terminals.tail

          string append s"IN: ${leaf.x.mkString} URAE → OUT: ${leaf.out.mkString};"
        }
        string.mkString
      case _ ⇒ "NOT AN AUTOENCODER"
    }
}
