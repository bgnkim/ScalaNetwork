package kr.ac.kaist.ir.deep.layer

import kr.ac.kaist.ir.deep.fn.{Activation, ScalarMatrix, ScalarMatrixOp}
import play.api.libs.json.{JsArray, JsObject, Json}

/**
 * __Layer__: Basic, Fully-connected Rank 3 Tensor Layer.
 *
 * @note <pre>
 *       v0 = a column vector
 *       Q = Rank 3 Tensor with size out, in × in is its entry.
 *       L = Rank 3 Tensor with size out, 1 × in is its entry.
 *       b = out × 1 matrix.
 *
 *       output = f( v0'.Q.v0 + L.v0 + b )
 *       </pre>
 *
 * @param IO is a tuple of the number of input and output, i.e. (2 → 4)
 * @param act is an activation function to be applied
 * @param quad is initial quadratic-level weight matrix Q for the case that it is restored from JSON (default: Seq())
 * @param lin is initial linear-level weight matrix L for the case that it is restored from JSON (default: null)
 * @param const is initial bias weight matrix b for the case that it is restored from JSON (default: null)
 */
class FullTensorLayer(IO: (Int, Int),
                      protected override val act: Activation,
                      quad: Seq[ScalarMatrix] = Seq(),
                      lin: ScalarMatrix = null,
                      const: ScalarMatrix = null)
  extends Rank3TensorLayer((IO._1, IO._1, IO._1), IO._2, act, quad, lin, const) {

  /**
   * Constructor, to support legacy versions.
   * @param IO is a tuple of the number of input and output, i.e. (2 → 4)
   * @param act is an activation function to be applied
   * @param quad is initial quadratic-level weight matrix Q for the case that it is restored from JSON (default: Seq())
   * @param lin is initial linear-level weight matrix L for the case that it is restored from JSON (default: Seq())
   * @param const is initial bias weight matrix b for the case that it is restored from JSON (default: null)
   * @return Initialized object (this)
   */
  def this(IO: (Int, Int), act: Activation,
           quad: Seq[ScalarMatrix] = Seq(), lin: Seq[ScalarMatrix] = Seq(), const: ScalarMatrix = null) =
    this(IO, quad, lin.zipWithIndex.foldLeft(ScalarMatrix $0(IO._2, IO._1)) {
      case (matx, (line, id)) ⇒ matx(id to id, ::) := line
    }, const)

  /**
   * Translate this layer into JSON object (in Play! framework)
   *
   * @return JSON object describes this layer
   */
  override def toJSON: JsObject = Json.obj(
    "type" → "FullTensorLayer",
    "in" → fanIn,
    "out" → fanOut,
    "act" → act.toJSON,
    "quadratic" → JsArray.apply(quadratic.map(_.to2DSeq)),
    "linear" → linear.to2DSeq,
    "bias" → bias.to2DSeq
  )

  /**
   * Retrieve first input
   *
   * @param x input to be separated
   * @return first input
   */
  protected override def in1(x: ScalarMatrix): ScalarMatrix = x

  /**
   * Retrive second input
   *
   * @param x input to be separated
   * @return second input
   */
  protected override def in2(x: ScalarMatrix): ScalarMatrix = x

  /**
   * Reconstruct error from fragments
   * @param in1 error of input1
   * @param in2 error of input2
   * @return restored error
   */
  override protected def restoreError(in1: ScalarMatrix, in2: ScalarMatrix): ScalarMatrix = in1 + in2
}
