package kr.ac.kaist.ir.deep.train

import breeze.linalg.sum
import breeze.numerics.{pow, sqrt}
import kr.ac.kaist.ir.deep.fn._
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

  /**
   * Apply & Back-prop given single input
   *
   * @param net A network that gets input
   * @param in Input for error computation.
   * @param real Real Output for error computation.
   * @param isPositive *(Unused)* Boolean that indicates whether this example is positive or not.
   *                   We don't need this because StandardRAE does not get negative input.
   */
  def roundTrip(net: Network, in: BinaryTree, real: Null, isPositive: Boolean = true): Unit = {
    in forward {
      x ⇒
        val out = x into_: net

        // normalize the output
        val xSq = pow(out, 2.0f)
        val lenSq = sum(xSq)
        val len: Scalar = sqrt(lenSq)
        val normalized = out :/ len

        // Note that length is the function of x_i.
        // Let z_i := x_i / len(x_i).
        // Then d z_i / d x_i = (len^2 - x_i^2) / len^3,
        //      d z_j / d x_i = - x_i * x_j / len^3
        val rows = xSq.rows
        val dZdX = ScalarMatrix $0(rows, rows)
        var r = 0
        while (r < rows) {
          //dZ_r
          var c = 0
          while (c < rows) {
            if (r == c) {
              //dX_c
              dZdX.update(r, c, (lenSq - xSq(r, 0)) / (len * lenSq))
            } else {
              dZdX.update(r, c, (-out(r, 0) * out(c, 0)) / (len * lenSq))
            }
            c += 1
          }
          r += 1
        }

        // un-normalize the error
        val normalErr = error.derivative(x, normalized)
        val err = dZdX * normalErr

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
    var total = 0.0f
    val in = pair._1
    in forward {
      x ⇒
        val out = net of x
        val len = sqrt(sum(pow(out, 2.0f)))
        val normalized = out :/ len
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
        val len = sqrt(sum(pow(out, 2.0f)))
        val normalized: ScalarMatrix = out :/ len
        val hid = net(x)
        string append s"IN: ${x.mkString} RAE → OUT: ${normalized.mkString}, HDN: ${hid.mkString}; "
        // propagate hidden-layer value
        hid
    }
    string.mkString
  }
}
