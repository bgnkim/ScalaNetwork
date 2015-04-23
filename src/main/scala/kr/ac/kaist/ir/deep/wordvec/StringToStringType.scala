package kr.ac.kaist.ir.deep.wordvec

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.train.StringType
import org.apache.commons.lang.NotImplementedException
import org.apache.spark.Accumulator
import org.apache.spark.broadcast.Broadcast

import scala.collection.mutable

/**
 * __Input Operation__ : String as Input & Expected Otput. __(Spark ONLY)__
 *
 * :: WORKING ::
 * @note This is not implemented yet.
 * @note This is for WordEmbedding Training.
 *       Be warned that the `onewayTrip` function generates ScalarMatrix rather than String.
 *
 * @param model Broadcast of WordEmbedding model that contains all meaningful words.
 * @param error An objective function `(Default: [[kr.ac.kaist.ir.deep.fn.SquaredErr]])`
 *
 * @example
 * {{{var make = new StringToStringType(model = wordModel, error = CrossEntropyErr)
 *     var out = make onewayTrip (net, in)}}}
 */
class StringToStringType(protected override val model: Broadcast[WordModel],
                         override val error: Objective) extends StringType[String] {
  var accWords: Accumulator[mutable.HashMap[String, ScalarMatrix]] = _

  /**
   * Set Accumulator of Word Embedding.
   * @param acc New Accumulator
   */
  def setAccumulator(acc: Accumulator[mutable.HashMap[String, ScalarMatrix]]) = {
    accWords = acc
  }

  /**
   * Apply & Back-prop given single input
   *
   * @param net A network that gets input
   * @param in Input for error computation.
   * @param real Real output for error computation.
   * @param isPositive Boolean that indicates whether this example is positive or not.
   */
  override def roundTrip(net: Network, in: String, real: String, isPositive: Boolean): Unit = {
    val out = model.value(in) into_: net
    val err: ScalarMatrix = error.derivative(model.value(real), out)

    val errToVec =
      if (isPositive) {
        net updateBy err
      } else {
        net updateBy (-err)
      }

    // TODO implement update
    throw new NotImplementedException()
  }

  /**
   * Make validation output
   *
   * @param net A network that gets input
   * @param pair (Input, Real output) pair for computation
   * @return input as string
   */
  override def stringOf(net: Network, pair: (String, String)): String = {
    val in = pair._1
    val real = pair._2
    val diff = net.of(model.value(in)) - model.value(real)
    s"IN: $in EXP: $real → DIFF: ${diff.mkString}"
  }

  /**
   * Apply given input and compute the error
   *
   * @param net A network that gets input
   * @param pair (Input, Real output) for error computation.
   * @return error of this network
   */
  override def lossOf(net: Network)(pair: (String, String)): Scalar = {
    val in = model.value(pair._1)
    val real = model.value(pair._2)
    val out = net of in
    error(real, out)
  }
}
