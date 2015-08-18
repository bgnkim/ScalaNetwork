package kr.ac.kaist.ir.deep

import kr.ac.kaist.ir.deep.fn._
import play.api.libs.json.{JsObject, JsValue}

import scala.reflect.runtime._

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
    protected var X: ScalarMatrix = _
    protected var dFdX: ScalarMatrix = _

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
     * @return propagated error (in this case, <code>dG/dx</code> )
     */
    protected[deep] def updateBy(error: ScalarMatrix): ScalarMatrix

    /**
     * Sugar: Forward computation. Calls apply(x)
     *
     * @param x input matrix
     * @return output matrix
     */
    protected[deep] def passedBy(x: ScalarMatrix) = {
      this.X = x
      val out = apply(x)
      dFdX =
        if (act != null)
          act.derivative(out)
        else
          out
      out
    }

    /**
     * Translate this layer into JSON object (in Play! framework)
     * @note Please make an LayerReviver object if you're using custom layer.
     *       In that case, please specify LayerReviver object's full class name as "__reviver__,"
     *       and fill up LayerReviver.revive method.
     * @return JSON object describes this layer
     */
    def toJSON: JsObject

    /**
     * Sugar: Forward computation. Calls apply(x)
     *
     * @param x input matrix
     * @return output matrix
     */
    @deprecated
    protected[deep] def into_:(x: ScalarMatrix) = passedBy(x)

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
   * __Trait__ that revives layer from JSON value
   */
  trait LayerReviver extends Serializable {
    /**
     * Revive layer using given JSON value
     * @param obj JSON value to be revived
     * @return Revived layer.
     */
    def revive(obj: JsValue): Layer
  }

  /**
   * Companion object of Layer
   */
  object Layer extends LayerReviver {
    @transient val runtimeMirror = universe.synchronized(universe.runtimeMirror(getClass.getClassLoader))

    /**
     * Load layer from JsObject
     *
     * @param obj JsObject to be parsed
     * @return New layer reconstructed from this object
     */
    def apply(obj: JsValue) = {
      val companion =
        universe.synchronized {
          (obj \ "reviver").asOpt[String] match {
            case Some(clsName) ⇒
              val module = runtimeMirror.staticModule(clsName)
              runtimeMirror.reflectModule(module).instance.asInstanceOf[LayerReviver]
            case None ⇒
              this
          }
        }
      companion.revive(obj)
    }

    /**
     * Load layer from JsObject
     *
     * @param obj JsObject to be parsed
     * @return New layer reconstructed from this object
     */
    def revive(obj: JsValue) = {
      val in = obj \ "in"
      val out = obj \ "out"
      val typeStr = (obj \ "type").as[String]

      val act = if (typeStr.endsWith("Layer")) {
        Activation.apply(obj \ "act")
      } else null

      val dropout = (obj \ "Dropout").asOpt[Probability]
      val normalize = (obj \ "Normalize").asOpt[String]

      typeStr match {
        case "NormOp" ⇒
          val factor = (obj \ "factor").as[Scalar]
          new NormalizeOperation(factor)
        case "DropoutOp" ⇒
          val presence = (obj \ "presence").as[Probability]
          new DropoutOperation(presence)
        case "GaussianRBF" ⇒
          val w = ScalarMatrix restore (obj \ "weight").as[IndexedSeq[IndexedSeq[String]]]
          val c = ScalarMatrix restore (obj \ "center").as[IndexedSeq[IndexedSeq[String]]]
          val modifiable = (obj \ "canModifyCenter").as[Boolean]
          (dropout, normalize) match {
            case (Some(p), Some(_)) ⇒
              new GaussianRBFLayer(in.as[Int], c, modifiable, w) with Dropout with Normalize withProbability p
            case (Some(p), None) ⇒
              new GaussianRBFLayer(in.as[Int], c, modifiable, w) with Dropout withProbability p
            case (None, Some(_)) ⇒
              new GaussianRBFLayer(in.as[Int], c, modifiable, w) with Normalize
            case _ ⇒
              new GaussianRBFLayer(in.as[Int], c, modifiable, w)
          }

        case "BasicLayer" ⇒
          val i = in.as[Int]
          val o = out.as[Int]
          val b = ScalarMatrix restore (obj \ "bias").as[IndexedSeq[IndexedSeq[String]]]
          val w = ScalarMatrix restore (obj \ "weight").as[IndexedSeq[IndexedSeq[String]]]
          (obj \ "reconst_bias").asOpt[IndexedSeq[IndexedSeq[String]]] match {
            case Some(rbraw) ⇒
              val rb = ScalarMatrix restore rbraw
              (dropout, normalize) match {
                case (Some(p), Some(_)) ⇒
                  new ReconBasicLayer(i → o, act, w, b, rb) with Dropout with Normalize withProbability p
                case (Some(p), None) ⇒
                  new ReconBasicLayer(i → o, act, w, b, rb) with Dropout withProbability p
                case (None, Some(_)) ⇒
                  new ReconBasicLayer(i → o, act, w, b, rb) with Normalize
                case _ ⇒
                  new ReconBasicLayer(i → o, act, w, b, rb)
              }
            case None ⇒
              (dropout, normalize) match {
                case (Some(p), Some(_)) ⇒
                  new BasicLayer(i → o, act, w, b) with Dropout with Normalize withProbability p
                case (Some(p), None) ⇒
                  new BasicLayer(i → o, act, w, b) with Dropout withProbability p
                case (None, Some(_)) ⇒
                  new BasicLayer(i → o, act, w, b) with Normalize
                case _ ⇒
                  new BasicLayer(i → o, act, w, b)
              }
          }

        case "LowerTriangularLayer" ⇒
          val i = in.as[Int]
          val o = out.as[Int]
          val b = ScalarMatrix restore (obj \ "bias").as[IndexedSeq[IndexedSeq[String]]]
          val w = ScalarMatrix restore (obj \ "weight").as[IndexedSeq[IndexedSeq[String]]]

          (dropout, normalize) match {
            case (Some(p), Some(_)) ⇒
              new LowerTriangularLayer(i → o, act, w, b) with Dropout with Normalize withProbability p
            case (Some(p), None) ⇒
              new LowerTriangularLayer(i → o, act, w, b) with Dropout withProbability p
            case (None, Some(_)) ⇒
              new LowerTriangularLayer(i → o, act, w, b) with Normalize
            case _ ⇒
              new LowerTriangularLayer(i → o, act, w, b)
          }

        case "SplitTensorLayer" ⇒
          val tuple = in.as[Seq[Int]]
          val i = (tuple.head, tuple(1))
          val o = out.as[Int]
          val b = ScalarMatrix restore (obj \ "bias").as[IndexedSeq[IndexedSeq[String]]]
          val quad = (obj \ "quadratic").as[Seq[IndexedSeq[IndexedSeq[String]]]] map ScalarMatrix.restore
          val linear =
            try {
              ScalarMatrix restore (obj \ "linear").as[IndexedSeq[IndexedSeq[String]]]
            } catch {
              case _: Throwable ⇒
                (obj \ "linear").as[Seq[IndexedSeq[IndexedSeq[String]]]].map(ScalarMatrix.restore)
                  .zipWithIndex.foldLeft(ScalarMatrix.$0(out.as[Int], tuple.sum)) {
                  case (matx, (row, id)) ⇒
                    matx(id to id, ::) := row
                    matx
                }
            }

          (dropout, normalize) match {
            case (Some(p), Some(_)) ⇒
              new SplitTensorLayer(i → o, act, quad, linear, b) with Dropout with Normalize withProbability p
            case (Some(p), None) ⇒
              new SplitTensorLayer(i → o, act, quad, linear, b) with Dropout withProbability p
            case (None, Some(_)) ⇒
              new SplitTensorLayer(i → o, act, quad, linear, b) with Normalize
            case _ ⇒
              new SplitTensorLayer(i → o, act, quad, linear, b)
          }
        case "FullTensorLayer" ⇒
          val i = in.as[Int]
          val o = out.as[Int]
          val b = ScalarMatrix restore (obj \ "bias").as[IndexedSeq[IndexedSeq[String]]]
          val quad = (obj \ "quadratic").as[Seq[IndexedSeq[IndexedSeq[String]]]] map ScalarMatrix.restore
          val linear =
            try {
              ScalarMatrix restore (obj \ "linear").as[IndexedSeq[IndexedSeq[String]]]
            } catch {
              case _: Throwable ⇒
                (obj \ "linear").as[Seq[IndexedSeq[IndexedSeq[String]]]].map(ScalarMatrix.restore)
                  .zipWithIndex.foldLeft(ScalarMatrix.$0(out.as[Int], in.as[Int])) {
                  case (matx, (row, id)) ⇒
                    matx(id to id, ::) := row
                    matx
                }
            }

          (dropout, normalize) match {
            case (Some(p), Some(_)) ⇒
              new FullTensorLayer(i → o, act, quad, linear, b) with Dropout with Normalize withProbability p
            case (Some(p), None) ⇒
              new FullTensorLayer(i → o, act, quad, linear, b) with Dropout withProbability p
            case (None, Some(_)) ⇒
              new FullTensorLayer(i → o, act, quad, linear, b) with Normalize
            case _ ⇒
              new FullTensorLayer(i → o, act, quad, linear, b)
          }
      }
    }
  }

}
