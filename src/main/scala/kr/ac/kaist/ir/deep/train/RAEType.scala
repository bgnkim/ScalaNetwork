package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.rec.BinaryTree

/**
 * __Input Operation__ : VectorTree as Input & Recursive Auto-Encoder Training (no output type)
 *
 * @note We recommend that you should not apply this method to non-AutoEncoder tasks
 * @note This implementation designed as a replica of the traditional RAE in
 *       [[http://ai.stanford.edu/~ang/papers/emnlp11-RecursiveAutoencodersSentimentDistributions.pdf this paper]]
 *
 * @param corrupt Corruption that supervises how to corrupt the input matrix. `(Default : [[kr.ac.kaist.ir.deep.train.NoCorruption]])`
 * @param error An objective function `(Default: [[kr.ac.kaist.ir.deep.fn.SquaredErr]])`
 *
 * @example
 * {{{var make = new RAEType(error = CrossEntropyErr)
 *            var corruptedIn = make corrupted in
 *            var out = make onewayTrip (net, corruptedIn)}}}
 */
class RAEType(override val corrupt: Corruption = NoCorruption,
              override val error: Objective = SquaredErr)
  extends TreeType {

  /**
   * Apply & Back-prop given single input
   *
   * @param net A network that gets input
   * @param in Input for error computation.
   * @param real Real Output for error computation.
   */
  def roundTrip(net: Network, in: BinaryTree, real: Null): Unit = {
    in forward {
      x ⇒
        val err = error.derivative(x, net passedBy x)
        net updateBy err
        // propagate hidden-layer value
        net(x)
    }
  }

  /**
   * Apply given input and compute the error
   *
   * @param net A network that gets input  
   * @param pair (Input, Real output) for error computation.
   * @return error of this network
   */
  def lossOf(net: Network)(pair: (BinaryTree, Null)): Scalar = {
    var sum = 0.0f
    val in = pair._1
    in forward {
      x ⇒
        sum += error(x, net of x)
        //propagate hidden-layer value
        net(x)
    }
    sum
  }

  /**
   * Make validation output
   *
   * @return input as string
   */
  def stringOf(net: Network, pair: (BinaryTree, Null)): String = {
    val string = StringBuilder.newBuilder
    pair._1 forward {
      x ⇒
        val out = net of x
        val hid = net(x)
        string append s"IN: ${x.mkString} RAE → OUT: ${out.mkString}, HDN: ${hid.mkString}; "
        // propagate hidden-layer value
        hid
    }
    string.mkString
  }
}
