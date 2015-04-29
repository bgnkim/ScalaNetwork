package kr.ac.kaist.ir.deep.wordvec

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.train.{Corruption, ManipulationType}
import org.apache.spark.broadcast.Broadcast

/**
 * __Trait of Input Operation__ : String as Input. This is an '''Abstract Implementation'''
 *
 * @tparam OUT Output type.
 */
trait StringType[OUT] extends ManipulationType[String, OUT] {
  override val corrupt: Corruption = null
  protected val model: Broadcast[WordModel]

  /**
   * Corrupt input : No corruption for string.
   *
   * @param x input to be corrupted
   * @return corrupted input
   */
  override def corrupted(x: String): String = x

  /**
   * Apply given single input as one-way forward trip.
   *
   * @param net A network that gets input
   * @param x input to be computed
   * @return output of the network.
   */
  override def onewayTrip(net: Network, x: String): ScalarMatrix =
    net of model.value(x)
}
