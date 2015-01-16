package kr.ac.kaist.ir.deep.train.op

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.train.{Corruption, NoCorruption}

/**
 * Input Operation : For Vector
 *
 * @param corrupt supervises how to corrupt the input matrix. (Default : [[NoCorruption]])
 * @param error is an objective function (Default: [[kr.ac.kaist.ir.deep.function.SquaredErr]])
 */
class ScalarVector(override protected[train] val corrupt: Corruption = NoCorruption,
                   override protected[train] val error: Objective = SquaredErr)
  extends InputOp[ScalarMatrix] {

  /**
   * Corrupt input
   * @return corrupted input
   */
  override def corrupted(x: ScalarMatrix): ScalarMatrix = corrupt(x)

  /**
   * Apply & Back-prop given single input
   * @param net that gets input
   */
  override def roundTrip(net: Network, in: ScalarMatrix, real: ScalarMatrix): Unit = {
    val out = in >>: net
    val err: ScalarMatrix = error.derivative(real, out)
    net ! err
  }

  /**
   * Apply given single input
   * @param net that gets input
   * @return output of the network.
   */
  override def onewayTrip(net: Network, x: ScalarMatrix): ScalarMatrix = net on x

  /**
   * Make input to string
   * @return input as string
   */
  override def stringOf(in: (ScalarMatrix, ScalarMatrix)): String =
    "IN:" + in._1.mkString + " EXP:" + in._2.mkString
}
