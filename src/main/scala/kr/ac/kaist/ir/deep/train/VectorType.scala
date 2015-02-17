package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network.Network

/**
 * __Input Operation__ : Vector as Input and output
 *
 * @param corrupt Corruption that supervises how to corrupt the input matrix. (Default : [[NoCorruption]])
 * @param error An objective function (Default: [[SquaredErr]])
 *
 * @example
 * {{{var make = new VectorType(error = CrossEntropyErr)
 *            var corruptedIn = make corrupted in
 *            var out = make onewayTrip (net, corruptedIn)}}}
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
   * @param in Input for error computation.
   * @param real Real Output for error computation.
   */
  def roundTrip(net: Network, in: ScalarMatrix, real: ScalarMatrix): Unit = {
    val out = in into_: net
    val err: ScalarMatrix = error.derivative(real, out)
    net updateBy err
  }

  /**
   * Apply given input and compute the error
   *
   * @param net A network that gets input
   * @param pair (Input, Real output) for error computation.
   * @return error of this network
   */
  override def lossOf(net: Network)(pair: (ScalarMatrix, ScalarMatrix)): Scalar = {
    val in = pair._1
    val real = pair._2
    val out = net of in
    error(real, out)
  }

  /**
   * Apply given single input as one-way forward trip.
   *
   * @param net A network that gets input
   * @param x input to be computed
   * @return output of the network.
   */
  override def onewayTrip(net: Network, x: ScalarMatrix): ScalarMatrix = net of x

  /**
   * Make validation output
   *
   * @return input as string
   */
  def stringOf(net: Network, pair: (ScalarMatrix, ScalarMatrix)): String = {
    val in = pair._1
    val real = pair._2
    val out = net of in
    s"IN: ${in.mkString} EXP: ${real.mkString} → OUT: ${out.mkString}"
  }
}
