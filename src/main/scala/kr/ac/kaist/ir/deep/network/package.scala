package kr.ac.kaist.ir.deep

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.layer.{BasicLayer, Layer, Reconstructable}
import play.api.libs.json._

/**
 * Package for network structure
 *
 * Created by bydelta on 2014-12-30.
 */
package object network {

  /**
   * Operation : Network
   * @param net to be applied
   */
  implicit class NetworkOp(net: Network) extends Serializable {

    /**
     * Copy given network by given amount
     * @param n is the nuber of copies (default: 1)
     * @return Sequence of copied network (Not linked)
     */
    def copy(n: Int = 1) = {
      val json = net.toJSON
      (1 to n) map { _ ⇒ Network(json)}
    }
  }

  /**
   * Companion object of BasicNetwork
   */
  object Network {
    /**
     * Load network from JsObject
     * @param obj to be parsed
     * @return New Network reconstructed from this object
     */
    def apply(obj: JsObject): Network = {
      (obj \ "type").as[String] match {
        case "AutoEncoder" ⇒
          AutoEncoder(obj)
        case "BasicNetwork" ⇒
          BasicNetwork(obj)
        case "StackedAutoEncoder" ⇒
          val layers = (obj \ "stack").as[Seq[JsObject]] map Network.AutoEncoder
          new StackedAutoEncoder(layers)
      }
    }

    /**
     * Load network from JsObject
     * @param obj to be parsed
     * @return New AutoEncoder reconstructed from this object
     */
    def AutoEncoder(obj: JsObject): AutoEncoder = {
      val layers = (obj \ "layers").as[JsArray].value map Layer.apply
      val presence = (obj \ "presence").as[Probability]
      new AutoEncoder(layers(0).asInstanceOf[Reconstructable], presence)
    }

    /**
     * Load network from JsObject
     * @param obj to be parsed
     * @return New Basic Network reconstructed from this object
     */
    def BasicNetwork(obj: JsObject): BasicNetwork = {
      val layers = (obj \ "layers").as[JsArray].value map Layer.apply
      val presence = (obj \ "presence").as[Probability]
      new BasicNetwork(layers, presence)
    }

    /**
     * Construct network from given layer size information
     * @param act for activation function
     * @param layerSizes for construct layers
     */
    def apply(act: Activation, layerSizes: Int*): Network =
      new BasicNetwork(layerSizes.indices.tail map {
        i ⇒ {
          new BasicLayer(layerSizes(i - 1) → layerSizes(i), act)
        }
      })
  }
}
