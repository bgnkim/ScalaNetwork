package kr.ac.kaist.ir.deep.train.op

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.fn.obj.{Objective, SquaredErr}
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.train.{Corruption, NoCorruption}

/**
 * __Input Operation__ : Vector as Input
 *
 * @param corrupt Corruption that supervises how to corrupt the input matrix. (Default : [[kr.ac.kaist.ir.deep.train.NoCorruption]])
 * @param error An objective function (Default: [[kr.ac.kaist.ir.deep.fn.obj.SquaredErr]])
 *
 * @example
 * {{{var make = new ScalarVector(error = CrossEntropyErr)
 *  var corruptedIn = make corrupted in
 *  var out = make onewayTrip (net, corruptedIn)}}}
 */
class ScalarVector(override protected[train] val corrupt: Corruption = NoCorruption,
                   override protected[train] val error: Objective = SquaredErr)
  extends InputOp[ScalarMatrix] {

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
   * @param in __corrupted__ input
   * @param real __Real label__ for comparing
   */
  override def roundTrip(net: Network, in: ScalarMatrix, real: ScalarMatrix): Unit = {
    val out = in >>: net
    val err: ScalarMatrix = error.derivative(real, out)
    net ! err
  }

  /**
   * Apply given single input as one-way forward trip.
   *
   * @param net A network that gets input
   * @param x input to be computed
   * @return output of the network.
   */
  override def onewayTrip(net: Network, x: ScalarMatrix): ScalarMatrix = net on x

  /**
   * Make input to string
   *
   * @return input as string
   */
  override def stringOf(in: (ScalarMatrix, ScalarMatrix)): String =
    "IN:" + in._1.mkString + " EXP:" + in._2.mkString
}
