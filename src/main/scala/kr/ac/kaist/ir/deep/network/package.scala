package kr.ac.kaist.ir.deep

import java.io.Serializable

import kr.ac.kaist.ir.deep.fn.{Activation, Probability, ScalarMatrix}
import kr.ac.kaist.ir.deep.layer.{BasicLayer, Layer, Reconstructable}
import play.api.libs.json.{JsArray, JsObject, JsValue, Json}

import scala.collection.mutable.ArrayBuffer
import scala.io.Codec
import scala.reflect.io.{File, Path}
import scala.reflect.runtime.universe

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
     * @note Please make an NetReviver object if you're using custom network.
     *       In that case, please specify NetReviver object's full class name as "__reviver__,"
     *       and fill up NetReviver.revive method.
     *
     * @return JsObject of this network
     */
    def toJSON: JsObject

    /**
     * Backpropagation algorithm
     *
     * @param err backpropagated error from error function
     */
    def updateBy(err: ScalarMatrix): ScalarMatrix

    /**
     * Forward computation for training
     *
     * @param x input matrix
     * @return output matrix
     */
    def into_:(x: ScalarMatrix): ScalarMatrix

    /**
     * Sugar: Forward computation for validation. Calls apply(x)
     *
     * @param x input matrix
     * @return output matrix
     */
    def of(x: ScalarMatrix): ScalarMatrix = apply(x)

    /**
     * Copy given network by given amount
     *
     * @return Sequence of copied network (Not linked)
     */
    def copy = {
      val json = this.toJSON
      Network(json)
    }

    /**
     * Save given network into given file.
     * @param path Path to save this network.
     * @param codec Codec used for writer. `(Default: Codec.UTF8)`
     */
    def saveAsJsonFile(path: Path, codec: Codec = Codec.UTF8): Unit = {
      val writer = File(path).bufferedWriter(append = false, codec = codec)
      writer.write(Json.prettyPrint(this.toJSON))
      writer.close()
    }
  }

  /**
   * __Trait__ of Network Reviver (Companion) objects
   */
  trait NetReviver extends Serializable {
    /**
     * Revive network using given JSON value
     * @param obj JSON value to be revived
     * @return Revived network.
     */
    def revive(obj: JsValue): Network
  }

  /**
   * Companion object of BasicNetwork
   */
  object Network extends NetReviver {
    @transient lazy val runtimeMirror = universe.synchronized(universe.runtimeMirror(getClass.getClassLoader))
    /**
     * Construct network from given layer size information
     *
     * @param act Activation function for activation function
     * @param layerSizes Sizes for construct layers
     */
    def apply(act: Activation, layerSizes: Int*): Network = {
      val layers = ArrayBuffer[Layer]()
      layers ++= layerSizes.indices.tail.map {
        i ⇒ new BasicLayer(layerSizes(i - 1) → layerSizes(i), act)
      }
      new BasicNetwork(layers)
    }

    /**
     * Load network from given file.
     * @param path Path to save this network.
     * @param codec Codec used for writer. `(Default: Codec.UTF8)`
     *
     * @tparam T Type of network casted into.
     */
    def jsonFile[T >: Network](path: Path, codec: Codec = Codec.UTF8): T = {
      val line = File(path).lines(codec).mkString("")
      val json = Json.parse(line)
      apply(json).asInstanceOf[T]
    }

    /**
     * Load network from JsObject
     *
     * @param obj JsObject to be parsed
     * @return New Network reconstructed from this object
     */
    def apply(obj: JsValue): Network = {
      val companion =
        universe.synchronized {
          (obj \ "reviver").asOpt[String] match {
            case Some(clsName) ⇒
              val module = runtimeMirror.staticModule(clsName)
              runtimeMirror.reflectModule(module).instance.asInstanceOf[NetReviver]
            case None ⇒
              this
          }
        }
      companion.revive(obj)
    }

    /**
     * Revive network using given JSON value
     * @param obj JSON value to be revived
     * @return Revived network.
     */
    override def revive(obj: JsValue): Network = {
      (obj \ "type").as[String] match {
        case "AutoEncoder" ⇒ AutoEncoder(obj)
        case "BasicNetwork" ⇒ BasicNetwork(obj)
        case "StackedAutoEncoder" ⇒ StackedAutoEncoder(obj)
      }
    }

    /**
     * Load network from JsObject
     *
     * @param obj JsObject to be parsed
     * @return New AutoEncoder reconstructed from this object
     */
    def AutoEncoder(obj: JsValue): AutoEncoder = {
      val layers = (obj \ "layers").as[JsArray].value map Layer.apply
      val presence = (obj \ "presence").as[Probability]
      new AutoEncoder(layers.head.asInstanceOf[Reconstructable], presence)
    }

    /**
     * Load network from JsObject
     *
     * @param obj JsObject to be parsed
     * @return New Basic Network reconstructed from this object
     */
    def BasicNetwork(obj: JsValue): BasicNetwork = {
      val layers = ArrayBuffer[Layer]()
      layers ++= (obj \ "layers").as[JsArray].value.map(Layer.apply)
      new BasicNetwork(layers)
    }

    /**
     * Load network from JsObject
     *
     * @param obj JsObject to be parsed
     * @return New Stacked AutoEncoder reconstructed from this object
     */
    def StackedAutoEncoder(obj: JsValue): StackedAutoEncoder = {
      val layers = (obj \ "stack").as[Seq[JsObject]] map Network.AutoEncoder
      new StackedAutoEncoder(layers)
    }

  }

}
