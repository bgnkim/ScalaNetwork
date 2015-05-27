package kr.ac.kaist.ir.deep

import kr.ac.kaist.ir.deep.fn._
import play.api.libs.json.{JsObject, JsValue}

/**
 * Package for layer implementation
 */
package object layer {

  /**
   * __Trait__ that describes layer-level computation
   *
   * Layer is an instance of ScalaMatrix => ScalaMatrix function.
   * Therefore "layers" can be composed together.
   */
  trait Layer extends (ScalarMatrix ⇒ ScalarMatrix) with Serializable {
    /** Activation Function */
    protected val act: Activation

    /**
     * Forward computation
     *
     * @param x input matrix
     * @return output matrix
     */
    override def apply(x: ScalarMatrix): ScalarMatrix

    /**
     * <p>Backward computation.</p>
     *
     * @note <p>
     *       Let this layer have function F composed with function <code> X(x) = W.x + b </code>
     *       and higher layer have function G.
     *       </p>
     *
     *       <p>
     *       Weight is updated with: <code>dG/dW</code>
     *       and propagate <code>dG/dx</code>
     *       </p>
     *
     *       <p>
     *       For the computation, we only used denominator layout. (cf. Wikipedia Page of Matrix Computation)
     *       For the computation rules, see "Matrix Cookbook" from MIT.
     *       </p>
     *
     * @param error to be propagated ( <code>dG / dF</code> is propagated from higher layer )
     * @param input of this layer (in this case, <code>x = entry of dX / dw</code>)
     * @param output of this layer (in this case, <code>y</code>)
     * @return propagated error (in this case, <code>dG/dx</code> )
     */
    protected[deep] def updateBy(error: ScalarMatrix, input: ScalarMatrix, output: ScalarMatrix): ScalarMatrix

    /**
     * Sugar: Forward computation. Calls apply(x)
     *
     * @param x input matrix
     * @return output matrix
     */
    protected[deep] def into_:(x: ScalarMatrix) = apply(x)

    /**
     * Translate this layer into JSON object (in Play! framework)
     *
     * @return JSON object describes this layer
     */
    def toJSON: JsObject

    /**
     * weights for update
     *
     * @return weights
     */
    val W: IndexedSeq[ScalarMatrix]

    /**
     * accumulated delta values
     *
     * @return delta-weight
     */
    val dW: IndexedSeq[ScalarMatrix]
  }

  /**
   * Companion object of Layer
   */
  object Layer {
    /** Sequence of supported activation functions */
    private val acts = Seq(Sigmoid, HyperbolicTangent, Rectifier, Softplus, Linear, HardSigmoid, HardTanh, Softmax)

    /**
     * Load layer from JsObject
     *
     * @param obj JsObject to be parsed
     * @return New layer reconstructed from this object
     */
    def apply(obj: JsValue) = {
      val in = obj \ "in"
      val out = obj \ "out"
      val typeStr = (obj \ "type").as[String]

      val act = if (typeStr.endsWith("Layer")) {
        val actStr = (obj \ "act").as[String]
        (acts find {
          x ⇒ x.getClass.getSimpleName == actStr
        }).getOrElse(HyperbolicTangent)
      } else null

      typeStr match {
        case "NormOp" ⇒
          val factor = (obj \ "factor").as[Scalar]
          new NormalizeOperation(factor)
        case "DropoutOp" ⇒
          val presence = (obj \ "presence").as[Probability]
          new DropoutOperation(presence)
        case "BasicLayer" ⇒
          val b = ScalarMatrix restore (obj \ "bias").as[IndexedSeq[IndexedSeq[String]]]
          val w = ScalarMatrix restore (obj \ "weight").as[IndexedSeq[IndexedSeq[String]]]
          (obj \ "reconst_bias").asOpt[IndexedSeq[IndexedSeq[String]]] match {
            case Some(rb) ⇒
              new ReconBasicLayer(in.as[Int] → out.as[Int], act, w, b, ScalarMatrix restore rb)
            case None ⇒
              new BasicLayer(in.as[Int] → out.as[Int], act, w, b)
          }
        case "SplitTensorLayer" ⇒
          val b = ScalarMatrix restore (obj \ "bias").as[IndexedSeq[IndexedSeq[String]]]
          val tuple = in.as[Seq[Int]]
          val quad = (obj \ "quadratic").as[Seq[IndexedSeq[IndexedSeq[String]]]] map { x ⇒ ScalarMatrix restore x }
          val linear = (obj \ "linear").as[Seq[IndexedSeq[IndexedSeq[String]]]] map { x ⇒ ScalarMatrix restore x }
          new SplitTensorLayer((tuple.head, tuple(1)) → out.as[Int], act, quad, linear, b)
        case "FullTensorLayer" ⇒
          val b = ScalarMatrix restore (obj \ "bias").as[IndexedSeq[IndexedSeq[String]]]
          val quad = (obj \ "quadratic").as[Seq[IndexedSeq[IndexedSeq[String]]]] map { x ⇒ ScalarMatrix restore x }
          val linear = (obj \ "linear").as[Seq[IndexedSeq[IndexedSeq[String]]]] map { x ⇒ ScalarMatrix restore x }
          new FullTensorLayer(in.as[Int] → out.as[Int], act, quad, linear, b)
      }
    }
  }

}
