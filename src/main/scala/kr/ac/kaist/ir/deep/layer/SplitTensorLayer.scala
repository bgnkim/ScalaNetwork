package kr.ac.kaist.ir.deep.layer

import kr.ac.kaist.ir.deep.fn._
import play.api.libs.json.{JsArray, JsObject, Json}

/**
 * __Layer__: Basic, Fully-connected Rank 3 Tensor Layer.
 *
 * @note <pre>
 * v0 = a column vector concatenate v2 after v1 (v11, v12, ... v1in1, v21, ...)
 * Q = Rank 3 Tensor with size out, in1 × in2 is its entry.
 * L = Rank 3 Tensor with size out, 1 × (in1 + in2) is its entry.
 * b = out × 1 matrix.
 *
 * output = f( v1'.Q.v2 + L.v0 + b )
 * </pre>
 *
 * @param IO is a tuple of the number of input and output, i.e. ((2, 3) → 4)
 * @param act is an activation function to be applied
 * @param quad is initial quadratic-level weight matrix Q for the case that it is restored from JSON (default: Seq())
 * @param lin is initial linear-level weight matrix L for the case that it is restored from JSON (default: Seq())
 * @param const is initial bias weight matrix b for the case that it is restored from JSON (default: null)
 */
class SplitTensorLayer(IO: ((Int, Int), Int),
                       protected override val act: Activation,
                       quad: Seq[ScalarMatrix] = Seq(),
                       lin: Seq[ScalarMatrix] = Seq(),
                       const: ScalarMatrix = null)
  extends Rank3TensorLayer((IO._1._1, IO._1._2, IO._1._1 + IO._1._2), IO._2, act, quad, lin, const) {
  /**
   * Translate this layer into JSON object (in Play! framework)
   *
   * @return JSON object describes this layer
   */
  override def toJSON: JsObject = Json.obj(
    "type" → "SplitTensorLayer",
    "in" → Json.arr(fanInA, fanInB),
    "out" → fanOut,
    "act" → act.getClass.getSimpleName,
    "quadratic" → JsArray.apply(quadratic.map(_.to2DSeq)),
    "linear" → JsArray.apply(linear.map(_.to2DSeq)),
    "bias" → bias.to2DSeq
  )

  /**
   * Retrieve first input
   *
   * @param x input to be separated
   * @return first input
   */
  protected override def in1(x: ScalarMatrix): ScalarMatrix = x(0 until fanInA, ::)

  /**
   * Retrive second input
   * @param x input to be separated
   * @return second input
   */
  protected override def in2(x: ScalarMatrix): ScalarMatrix = x(fanInA to -1, ::)
}
