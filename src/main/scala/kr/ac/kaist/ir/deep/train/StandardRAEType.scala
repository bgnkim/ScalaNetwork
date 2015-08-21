package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.layer.NormalizeOperation
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.rec.BinaryTree

/**
 * __Input Operation__ : VectorTree as Input & Recursive Auto-Encoder Training (no output type)
 *
 * @note We recommend that you should not apply this method to non-AutoEncoder tasks
 * @note This implementation designed as a replica of the standard RAE (RAE + normalization) in
 *       [[http://ai.stanford.edu/~ang/papers/emnlp11-RecursiveAutoencodersSentimentDistributions.pdf this paper]]
 *
 * @param corrupt Corruption that supervises how to corrupt the input matrix. `(Default : [[kr.ac.kaist.ir.deep.train.NoCorruption]])`
 * @param error An objective function `(Default: [[kr.ac.kaist.ir.deep.fn.SquaredErr]])`
 *
 * @example
 * {{{var make = new RAEType(error = CrossEntropyErr)
 *             var corruptedIn = make corrupted in
 *             var out = make onewayTrip (net, corruptedIn)}}}
 */
class StandardRAEType(override val corrupt: Corruption = NoCorruption,
                      override val error: Objective = SquaredErr)
  extends TreeType {
  /** Normalization layer */
  val normalizeLayer = new NormalizeOperation()

  /**
   * Apply & Back-prop given single input
   *
   * @param net A network that gets input
   * @param delta Sequence of delta updates
   */
  def roundTrip(net: Network, delta: Seq[ScalarMatrix]) = (in: BinaryTree, real: Null) ⇒ {
    in forward {
      x ⇒
        val out = net passedBy x
        val zOut = normalizeLayer passedBy out
        val dit = delta.toIterator

        // un-normalize the error
        val normalErr = error.derivative(x, zOut)
        val err = normalizeLayer updateBy(dit, normalErr)

        net updateBy(dit, err)

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
    var total = 0.0f
    val in = pair._1
    in forward {
      x ⇒
        val out = net of x
        val normalized = normalizeLayer(out)
        total += error(x, normalized)
        //propagate hidden-layer value
        net(x)
    }
    total
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
        val normalized = normalizeLayer(out)
        val hid = net(x)
        string append s"IN: ${x.mkString} RAE → OUT: ${normalized.mkString}, HDN: ${hid.mkString}; "
        // propagate hidden-layer value
        hid
    }
    string.mkString
  }
}
