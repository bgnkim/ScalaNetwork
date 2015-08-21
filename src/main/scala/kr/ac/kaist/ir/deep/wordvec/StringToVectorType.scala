package kr.ac.kaist.ir.deep.wordvec

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network.Network
import org.apache.spark.broadcast.Broadcast

/**
 * __Input Operation__ : String as Input & ScalarMatrix as Otput __(Spark ONLY)__
 *
 * @param model Broadcast of WordEmbedding model that contains all meaningful words.
 * @param error An objective function `(Default: [[kr.ac.kaist.ir.deep.fn.SquaredErr]])`
 *
 * @example
 * {{{var make = new StringToVectorType(model = wordModel, error = CrossEntropyErr)
 *     var out = make onewayTrip (net, in)}}}
 */
class StringToVectorType(protected override val model: Broadcast[WordModel],
                         override val error: Objective) extends StringType[ScalarMatrix] {
  /**
   * Apply & Back-prop given single input
   *
   * @param net A network that gets input
   * @param delta Sequence of delta updates
   */
  def roundTrip(net: Network, delta: Seq[ScalarMatrix]) = (in: String, real: ScalarMatrix) ⇒ {
    val out = net.passedBy(model.value(in))
    val err: ScalarMatrix = error.derivative(real, out)
    net updateBy(delta.toIterator, err)
  }

  /**
   * Make validation output
   *
   * @param net A network that gets input
   * @param pair (Input, Real output) pair for computation
   * @return input as string
   */
  override def stringOf(net: Network, pair: (String, ScalarMatrix)): String = {
    val in = pair._1
    val real = pair._2
    val out = net of model.value(in)
    s"IN: $in EXP: ${real.mkString} → OUT: ${out.mkString}"
  }

  /**
   * Apply given input and compute the error
   *
   * @param net A network that gets input
   * @param pair (Input, Real output) for error computation.
   * @return error of this network
   */
  override def lossOf(net: Network)(pair: (String, ScalarMatrix)): Scalar = {
    val in = pair._1
    val real = pair._2
    val out = net of model.value(in)
    error(real, out)
  }
}
