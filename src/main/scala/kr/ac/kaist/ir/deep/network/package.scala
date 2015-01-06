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
   * Trait: Network interface
   */
  trait Network extends (ScalarMatrix ⇒ ScalarMatrix) with Serializable {
    /**
     * Probability that each input neuron of each layer is not dropped (For neuron drop-out)
     */
    val presence: Probability

    /**
     * All weights of layers
     * @return all weights of layers
     */
    def W: Seq[ScalarMatrix]

    /**
     * All accumulated delta weights of layers
     * @return all accumulated delta weights
     */
    def dW: Seq[ScalarMatrix]

    /**
     * Backpropagation algorithm
     * @param err backpropagated error from error function
     */
    def !(err: ScalarMatrix): ScalarMatrix

    /**
     * Sugar: Forward computation for training. Calls apply(x)
     *
     * @param x of input matrix
     * @return output matrix
     */
    def >>:(x: ScalarMatrix): ScalarMatrix

    /**
     * Serialize network to JSON
     * @return JsObject
     */
    def toJSON: JsObject
  }

  /**
   * Network: A basic network implementation
   * @param layers of this network
   * @param presence is the probability of non-dropped neurons (for drop-out training). Default value = 1.0
   */
  class BasicNetwork(val layers: Seq[Layer], override val presence: Probability = 1.0) extends Network {
    /** Collected input & output of each layer */
    var input: Seq[ScalarMatrix] = Seq()

    /**
     * All weights of layers
     * @return all weights of layers
     */
    override def W = layers flatMap {
      _.W
    }

    /**
     * All accumulated delta weights of layers
     * @return all accumulated delta weights
     */
    override def dW = layers flatMap {
      _.dW
    }


    /**
     * Backpropagation algorithm
     * @param err backpropagated error from error function
     */
    override def !(err: ScalarMatrix) =
      layers.indices.foldRight(err) {
        (id, e) ⇒ {
          val l = layers(id)
          val out = input(id + 1)
          val in = input(id)
          l !(e, in, out)
        }
      }

    /**
     * Compute output of neural network with given input
     * If drop-out is used, to average drop-out effect, we need to multiply output by presence probability.
     *
     * @param in is an input vector
     * @return output of the vector
     */
    override def apply(in: ScalarMatrix): ScalarMatrix = {
      // We don't have to store this value
      val localInput = layers.indices.foldLeft(Seq(in)) {
        (seq, id) ⇒ {
          val layerOut = seq.head >>: layers(id)
          val adjusted: ScalarMatrix = layerOut :* presence.safe
          adjusted +: seq
        }
      }
      localInput.head
    }

    /**
     * Forward computation for training.
     * If drop-out is used, we need to drop-out entry of input vector.
     *
     * @param x of input matrix
     * @return output matrix
     */
    override def >>:(x: ScalarMatrix): ScalarMatrix = {
      // We have to store this value
      input = layers.indices.foldLeft(Seq(x)) {
        (seq, id) ⇒ {
          val in = seq.head
          if (presence < 1.0)
            in :*= ScalarMatrix $01(in.rows, in.cols, presence.safe)
          (in >>: layers(id)) +: seq
        }
      }
      input.head
    }

    /**
     * Serialize network to JSON
     * @return JsObject
     */
    override def toJSON = Json.obj(
      "type" → "BasicNetwork",
      "presence" → presence.safe,
      "layers" → JsArray(layers map (_.toJSON))
    )
  }

  class AutoEncoder(val layer: Reconstructable, override val presence: Probability = 1.0) extends Network {
    /** Collected input & output of each layer */
    var input: ScalarMatrix = null
    var hidden: ScalarMatrix = null
    var output: ScalarMatrix = null

    /**
     * All weights of layers
     * @return all weights of layers
     */
    override def W = layer.W

    /**
     * All accumulated delta weights of layers
     * @return all accumulated delta weights
     */
    override def dW = layer.dW


    /**
     * Backpropagation algorithm
     * @param err backpropagated error from error function
     */
    override def !(err: ScalarMatrix) = {
      val reconErr = layer rec_!(err, hidden, output)
      layer !(reconErr, input, output)
    }

    /**
     * Compute output of neural network with given input (without reconstruction)
     * If drop-out is used, to average drop-out effect, we need to multiply output by presence probability.
     *
     * @param in is an input vector
     * @return output of the vector
     */
    override def apply(in: ScalarMatrix): ScalarMatrix = layer(in) :* presence.safe

    /**
     * Forward computation for training.
     * If drop-out is used, we need to drop-out entry of input vector.
     *
     * @param x of input matrix
     * @return output matrix
     */
    override def >>:(x: ScalarMatrix): ScalarMatrix = {
      input = x
      if (presence < 1.0)
        input :*= ScalarMatrix $01(x.rows, x.cols, presence.safe)
      hidden = input >>: layer
      if (presence < 1.0)
        hidden :*= ScalarMatrix $01(hidden.rows, hidden.cols, presence.safe)
      output = hidden rec_>>: layer
      output
    }

    /**
     * Serialize network to JSON
     * @return JsObject
     */
    override def toJSON = Json.obj(
      "type" → "AutoEncoder",
      "presence" → presence.safe,
      "layers" → Json.arr(layer.toJSON)
    )
  }

  class StackedAutoEncoder(val encoders: Seq[AutoEncoder]) extends Network {
    override val presence: Probability = 0.0

    /**
     * All accumulated delta weights of layers
     * @return all accumulated delta weights
     */
    override def dW: Seq[ScalarMatrix] = encoders flatMap (_.dW)

    /**
     * All weights of layers
     * @return all weights of layers
     */
    override def W: Seq[ScalarMatrix] = encoders flatMap (_.W)

    /**
     * Sugar: Forward computation for training. Calls apply(x)
     *
     * @param x of input matrix
     * @return output matrix
     */
    override def >>:(x: ScalarMatrix): ScalarMatrix =
      encoders.foldLeft(x) {
        (in, enc) ⇒ in >>: enc
      }

    /**
     * Serialize network to JSON
     * @return JsObject
     */
    override def toJSON: JsObject =
      Json.obj(
        "type" → "StackedAutoEncoder",
        "stack" → Json.arr(encoders map (_.toJSON))
      )

    /**
     * Backpropagation algorithm
     * @param err backpropagated error from error function
     */
    override def !(err: ScalarMatrix): ScalarMatrix =
      encoders.foldRight(err) {
        (enc, err) ⇒ enc ! err
      }

    /**
     * Compute output of neural network with given input (without reconstruction)
     * If drop-out is used, to average drop-out effect, we need to multiply output by presence probability.
     *
     * @param in is an input vector
     * @return output of the vector
     */
    override def apply(in: ScalarMatrix): ScalarMatrix =
      encoders.foldLeft(in) {
        (x, enc) ⇒ enc(x)
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
    def fromJSON(obj: JsObject): Network = {
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
      val layers = (obj \ "layers").as[JsArray].value map Layer.fromJSON
      val presence = (obj \ "presence").as[Probability]
      new AutoEncoder(layers(0).asInstanceOf[Reconstructable], presence)
    }

    /**
     * Load network from JsObject
     * @param obj to be parsed
     * @return New Basic Network reconstructed from this object
     */
    def BasicNetwork(obj: JsObject): BasicNetwork = {
      val layers = (obj \ "layers").as[JsArray].value map Layer.fromJSON
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
