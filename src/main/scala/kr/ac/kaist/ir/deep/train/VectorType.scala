package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network.Network

/**
 * __Input Operation__ : Vector as Input and output
 *
 * @param corrupt Corruption that supervises how to corrupt the input matrix. (Default : [[kr.ac.kaist.ir.deep.train.NoCorruption]])
 * @param error An objective function (Default: [[SquaredErr]])
 *
 * @example
 * {{{var make = new VectorType(error = CrossEntropyErr)
 *       var corruptedIn = make corrupted in
 *       var out = make onewayTrip (net, corruptedIn)}}}
 */
class VectorType(override protected[train] val corrupt: Corruption = NoCorruption,
                 override protected[train] val error: Objective = SquaredErr)
  extends ManipulationType[ScalarMatrix, ScalarMatrix] {

  /**
   * Corrupt input
   *
   * @param x input to be corrupted 
   * @return corrupted input
   */
  override def corrupted(x: ScalarMatrix): ScalarMatrix = corrupt(x)

  /**
   * Apply & Back-prop given single input
   *
   * @param net A network that gets input
   * @param seq Sequence of (Input, Real output) for error computation.
   */
  def roundTrip(net: Network, seq: Seq[(ScalarMatrix, ScalarMatrix)]): Unit =
    seq foreach {
      pair ⇒
        val in = pair._1
        val real = pair._2
        val out = in into_: net
        val err: ScalarMatrix = error.derivative(real, out)
        net updateBy err
    }

  /**
   * Apply given input and compute the error
   *
   * @param net A network that gets input  
   * @param validation Sequence of (Input, Real output) for error computation.
   * @return error of this network
   */
  override def lossOf(net: Network, validation: Seq[(ScalarMatrix, ScalarMatrix)]): Scalar =
    validation.map {
      pair ⇒
        val in = pair._1
        val real = pair._2
        val out = net of in
        error(real, out)
    }.sum

  /**
   * Apply given single input as one-way forward trip.
   *
   * @param net A network that gets input
   * @param x input to be computed
   * @return output of the network.
   */
  override def onewayTrip(net: Network, x: ScalarMatrix): ScalarMatrix = net of x

  /**
   * Make input to string
   *
   * @return input as string
   */
  override def stringOf(in: (ScalarMatrix, ScalarMatrix)): String =
    "IN:" + in._1.mkString + " EXP:" + in._2.mkString
}
