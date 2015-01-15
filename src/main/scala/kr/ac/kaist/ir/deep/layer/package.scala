package kr.ac.kaist.ir.deep

import kr.ac.kaist.ir.deep.function._
import play.api.libs.json.JsValue

/**
 * Package for layer implementation
 *
 * Created by bydelta on 2015-01-06.
 */
package object layer {

  /**
   * Companion object of Layer
   */
  object Layer {
    /** Sequence of supported activation functions */
    private val acts = Seq(Sigmoid, HyperbolicTangent, Rectifier, Softplus)

    /**
     * Load layer from JsObject
     * @param obj to be parsed
     * @return New layer reconstructed from this object
     */
    def apply(obj: JsValue) = {
      val in = obj \ "in"
      val out = (obj \ "out").as[Int]

      val actStr = (obj \ "act").as[String]
      val act = (acts find {
        x ⇒ x.getClass.getSimpleName == actStr
      }).getOrElse(HyperbolicTangent)

      val b = ScalarMatrix restore (obj \ "bias").as[Seq[Seq[Scalar]]]

      (obj \ "type").as[String] match {
        case "BasicLayer" ⇒
          val w = ScalarMatrix restore (obj \ "weight").as[Seq[Seq[Scalar]]]
          (obj \ "reconst_bias").asOpt[Seq[Seq[Scalar]]] match {
            case Some(rb) ⇒
              new ReconBasicLayer(in.as[Int] → out, act, w, b, ScalarMatrix restore rb)
            case None ⇒
              new BasicLayer(in.as[Int] → out, act, w, b)
          }
        case "Rank3TensorLayer" ⇒
          val tuple = in.as[Seq[Int]]
          val quad = (obj \ "quadratic").as[Seq[Seq[Seq[Scalar]]]] map { x ⇒ ScalarMatrix restore x}
          val linear = (obj \ "linear").as[Seq[Seq[Seq[Scalar]]]] map { x ⇒ ScalarMatrix restore x}
          new SplitVecTensorLayer((tuple(0), tuple(1)) → out, act, quad, linear, b)
      }
    }
  }

}
