package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.rec.DAG

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
class RAEType(override protected[train] val corrupt: Corruption = NoCorruption,
              override protected[train] val error: Objective = SquaredErr)
  extends DAGType {

  /**
   * Apply & Back-prop given single input
   *
   * @param net A network that gets input
   * @param seq Sequence of (Input, Real output) for error computation.
   */
  def roundTrip(net: Network, seq: Iterator[(DAG, Null)]): Unit = {
    while (seq.hasNext) {
      val pair = seq.next()
      pair._1 forward {
        x ⇒
          val err = error.derivative(x, x into_: net)
          net updateBy err
          // propagate hidden-layer value
          net(x)
      }
    }
  }

  /**
   * Apply given input and compute the error
   *
   * @param net A network that gets input  
   * @param validation Sequence of (Input, Real output) for error computation.
   * @return error of this network
   */
  override def lossOf(net: Network, validation: Iterator[(DAG, Null)]): Scalar = {
    var sum = 0.0
    while (validation.hasNext) {
      val pair = validation.next()
      val in = pair._1
      in forward {
        x ⇒
          sum += error(x, net of x)
          //propagate hidden-layer value
          net(x)
      }
    }
    sum
  }

  /**
   * Make validation output
   *
   * @return input as string
   */
  def stringOf(net: Network, pair: (DAG, Null)): String = {
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
