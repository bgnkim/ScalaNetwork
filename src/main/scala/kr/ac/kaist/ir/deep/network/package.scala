package kr.ac.kaist.ir.deep

import kr.ac.kaist.ir.deep.fn.{Activation, Probability, ScalarMatrix}
import kr.ac.kaist.ir.deep.layer.{BasicLayer, Layer, Reconstructable}
import play.api.libs.json.{JsArray, JsObject}

import scala.collection.mutable.ArrayBuffer

/**
 * Package for network structure
 */
package object network {

  /**
   * __Trait__: Network interface
   */
  trait Network extends (ScalarMatrix ⇒ ScalarMatrix) with Serializable {
    /**
     * All weights of layers
     *
     * @return all weights of layers
     */
    val W: IndexedSeq[ScalarMatrix]

    /**
     * All accumulated delta weights of layers
     *
     * @return all accumulated delta weights
     */
    val dW: IndexedSeq[ScalarMatrix]

    /**
     * Serialize network to JSON
     *
     * @return JsObject of this network
     */
    def toJSON: JsObject

    /**
     * Backpropagation algorithm
     *
     * @param err backpropagated error from error function
     */
    protected[deep] def updateBy(err: ScalarMatrix): ScalarMatrix

    /**
     * Forward computation for training
     *
     * @param x input matrix
     * @return output matrix
     */
    protected[deep] def into_:(x: ScalarMatrix): ScalarMatrix

    /**
     * Sugar: Forward computation for validation. Calls apply(x)
     *
     * @param x input matrix
     * @return output matrix
     */
    protected[deep] def of(x: ScalarMatrix): ScalarMatrix = apply(x)
  }

  /**
   * Operation for Network
   *
   * @param net the network that operation will be applied
   */
  implicit class NetworkOp(net: Network) extends Serializable {

    /**
     * Copy given network by given amount
     *
     * @param n the number of copies (default: 1)
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
     *
     * @param obj JsObject to be parsed
     * @return New Network reconstructed from this object
     */
    def apply(obj: JsObject): Network =
      (obj \ "type").as[String] match {
        case "AutoEncoder" ⇒
          AutoEncoder(obj)
        case "BasicNetwork" ⇒
          BasicNetwork(obj)
        case "StackedAutoEncoder" ⇒
          val layers = (obj \ "stack").as[Seq[JsObject]] map Network.AutoEncoder
          new StackedAutoEncoder(layers)
      }

    /**
     * Load network from JsObject
     *
     * @param obj JsObject to be parsed
     * @return New AutoEncoder reconstructed from this object
     */
    def AutoEncoder(obj: JsObject): AutoEncoder = {
      val layers = (obj \ "layers").as[JsArray].value map Layer.apply
      val presence = (obj \ "presence").as[Probability]
      new AutoEncoder(layers(0).asInstanceOf[Reconstructable], presence)
    }

    /**
     * Load network from JsObject
     *
     * @param obj JsObject to be parsed
     * @return New Basic Network reconstructed from this object
     */
    def BasicNetwork(obj: JsObject): BasicNetwork = {
      val layers = ArrayBuffer[Layer]()
      (obj \ "layers").as[JsArray].value.foreach {
        obj ⇒
          layers.append(Layer(obj))
      }
      new BasicNetwork(layers)
    }

    /**
     * Construct network from given layer size information
     *
     * @param act Activation function for activation function
     * @param layerSizes Sizes for construct layers
     */
    def apply(act: Activation, layerSizes: Int*): Network = {
      val layers = ArrayBuffer[Layer]()
      layerSizes.indices.tail.foreach {
        i ⇒ layers.append(new BasicLayer(layerSizes(i - 1) → layerSizes(i), act))
      }
      new BasicNetwork(layers)
    }
  }

}
