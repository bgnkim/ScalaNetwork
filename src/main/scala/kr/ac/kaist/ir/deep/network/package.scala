package kr.ac.kaist.ir.deep

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.trainer.WeightUpdater
import kr.ac.kaist.ir.deep.weight.Weight
import play.api.libs.json._

import scala.annotation.tailrec

/**
 * Package for network structure
 *
 * Created by bydelta on 2014-12-30.
 */
package object network {
  private final val WEIGHT = "weights"
  private final val WEIGHTID = "w"
  private final val NEURONS = "neurons"
  private final val SYNAPSES = "synapses"
  private final val SOURCE = "src"
  private final val DESTINATION = "dst"
  private final val ACTIVATION = "activation"
  private final val ID = "id"
  private final val OUTPUT = "outputs"
  private final val SIGMOID = "Sigmoid"
  private final val TANH = "Tanh"
  private final val SOFTPLUS = "Softplus"
  private final val RELU = "ReLU"

  /**
   * Trait: Neural Object (extended by Neuron & Synapse)
   */
  trait NeuralObject extends Serializable {
    /**
     * Weight for this object
     */
    protected[deep] val weight: Weight

    /**
     * Error accumulate function.
     * @param dy is a back propagaged error = \frac{df}{dy}
     * @return Error for back propagation = \frac{df}{dx}
     */
    def error(dy: Scalar): Seq[(NeuronId, Scalar)]

    /**
     * Serialize this object as JSON
     * @return JSON object
     */
    def toJSONObject: JsObject
  }

  /**
   * Trait: Neuron
   */
  trait Neuron extends NeuralObject with (Scalar ⇒ Scalar)

  /**
   * Trait: Synapse
   */
  trait Synapse extends NeuralObject {
    /** Input of this synapse */
    protected[network] var input = 0.0
    /** Input Seq */
    protected var inputSeq: Seq[(NeuronId, Scalar)] = null
    /** Indicate whether this is dropped or not */
    protected var dropped = false

    /**
     * Set whether this synapse dropped or not.
     * @param clause for checking
     */
    def dropIf(clause: Boolean) = {
      dropped = clause
    }

    /**
     * Process input of sequence and give output of this synapse.
     * @param in is a sequence of input
     * @return output of this synapse (Weighted multiplication of the input)
     */
    def apply(in: (NeuronId, Scalar)*) = if (!dropped) {
      inputSeq = in
      input = in.foldLeft(1.0) { (one, pair) ⇒ one * pair._2}
      weight * input
    } else 0.0

    /**
     * Error accumulate function.
     * @param df is a back propagaged error = \frac{df}{dy}
     * @return Error for back propagation = \frac{df}{dx}
     */
    override def error(df: Scalar) = if (!dropped) {
      weight delta_+= (input * df)
      (0 until inputSeq.size) map {
        i ⇒ inputSeq(i)._1 → (weight * df * (0 until inputSeq.size).foldLeft(1.0) {
          (x, j) ⇒ if (j != i) x * inputSeq(j)._2 else x
        })
      }
    } else inputSeq map { pair ⇒ pair._1 → 0.0}
  }

  /**
   * Neuron: Basic computational unit.
   * @param id is an identifier of this neuron.
   * @param weight is an object for bias
   * @param act is an activation function
   */
  case class ComputeNeuron(id: NeuronId,
                           override val weight: Weight,
                           act: Activation)
    extends Neuron {

    /**
     * String Representation
     * @return Representative String
     */
    override def toString = s"{$id: $weight}"

    /**
     * Stored previous output
     */
    protected[network] var output = 0.0

    /**
     * Process input of neuron and give output of this neuron
     * @param x is an input of this neuron.
     * @return output, applied by activation function
     */
    def apply(x: Scalar) = {
      output = act(weight + x)
      output
    }

    /**
     * Error accumulate function.
     * @param dy is a back propagaged error = \frac{df}{dy}
     * @return Error for back propagation = \frac{df}{dx}
     */
    override def error(dy: Scalar) = {
      val df = dy * act.derivative(output)
      weight delta_+= df
      Seq(-id → df)
    }

    /**
     * Serialize this object as JSON
     * @return JSON object
     */
    override def toJSONObject =
      Json.obj(
        ID → id,
        ACTIVATION → (act match {
          case Sigmoid ⇒ SIGMOID
          case Rectifier ⇒ RELU
          case Softplus ⇒ SOFTPLUS
          case _ ⇒ TANH
        }),
        WEIGHTID → weight.id.toString
      )
  }

  /**
   * Synapse: with 1 input and 1 output
   * @param src1 is an input source ID
   * @param dst is an output destination ID
   * @param weight is a weight
   */
  case class Synapse1(src1: NeuronId,
                      dst: NeuronId,
                      override val weight: Weight)
    extends Synapse {
    /**
     * String Representation
     * @return Representative String
     */
    override def toString = s"($src1→$dst: $weight)"

    /**
     * Serialize this object as JSON
     * @return JSON object
     */
    override def toJSONObject =
      Json.obj(
        WEIGHTID → weight.id.toString,
        SOURCE → Json.arr(src1),
        DESTINATION → dst
      )
  }

  /**
   * Synapse: with 2 input and 1 output
   * @param src1 is an input source ID
   * @param src2 is an input source ID
   * @param dst is an output destination ID
   * @param weight is a weight
   */
  case class Synapse2(src1: NeuronId,
                      src2: NeuronId,
                      dst: NeuronId,
                      override val weight: Weight)
    extends Synapse {
    /**
     * String Representation
     * @return Representative String
     */
    override def toString = s"($src1.$src2→$dst: $weight)"

    /**
     * Serialize this object as JSON
     * @return JSON object
     */
    override def toJSONObject =
      Json.obj(
        WEIGHTID → weight.id.toString,
        SOURCE → Json.arr(src1, src2),
        DESTINATION → dst
      )
  }

  /**
   * Synapse: with 2 input and 1 output
   * @param src1 is an input source ID
   * @param src2 is an input source ID
   * @param src3 is an input source ID
   * @param dst is an output destination ID
   * @param weight is a weight
   */
  case class Synapse3(src1: NeuronId,
                      src2: NeuronId,
                      src3: NeuronId,
                      dst: NeuronId,
                      override val weight: Weight)
    extends Synapse {
    /**
     * String Representation
     * @return Representative String
     */
    override def toString = s"($src1.$src2.$src3→$dst: $weight)"

    /**
     * Serialize this object as JSON
     * @return JSON object
     */
    override def toJSONObject =
      Json.obj(
        WEIGHTID → weight.id.toString,
        SOURCE → Json.arr(src1, src2, src3),
        DESTINATION → dst
      )
  }

  /**
   * Synapse: with 2 input and 1 output
   * @param src1 is an input source ID
   * @param src2 is an input source ID
   * @param src3 is an input source ID
   * @param src4 is an input source ID
   * @param dst is an output destination ID
   * @param weight is a weight
   */
  case class Synapse4(src1: NeuronId,
                      src2: NeuronId,
                      src3: NeuronId,
                      src4: NeuronId,
                      dst: NeuronId,
                      override val weight: Weight)
    extends Synapse {
    /**
     * String Representation
     * @return Representative String
     */
    override def toString = s"($src1.$src2.$src3.$src4→$dst: $weight)"

    /**
     * Serialize this object as JSON
     * @return JSON object
     */
    override def toJSONObject =
      Json.obj(
        WEIGHTID → weight.id.toString,
        SOURCE → Json.arr(src1, src2, src3, src4),
        DESTINATION → dst
      )
  }

  /**
   * Operator Def: Input Sequence
   * @param input sequence which is grouped by destination id
   */
  implicit class InputSeqOp(val input: Map[NeuronId, Seq[Scalar]]) extends AnyVal {
    /**
     * Size of current input set for given id
     * @param id of neuron that use this input set
     * @return size of input set
     */
    def sizeOf(id: NeuronId) = if (input contains id) input(id).size else 0

    /**
     * Concatenate given sequence into map
     * @param seq to be concatenated
     * @return element-wise concatenated map
     */
    def ++?(seq: Seq[(NeuronId, Scalar)]) =
      seq.foldLeft(input) {
        (map, pair) ⇒ {
          val set = map.getOrElse(pair._1, Seq[Scalar]())
          map + (pair._1 → (pair._2 +: set))
        }
      }
  }

  /**
   * Network: Abstract implementation without initialization
   */
  abstract class Network extends (NeuronVector ⇒ NeuronVector) with Serializable {
    /** Set of output neurons */
    protected[deep] val outNeurons: Seq[Neuron]
    /** Set of hidden neurons (Input is not included) */
    protected[network] val neurons: Seq[Neuron]
    /** Set of synapses */
    protected[deep] val synapses: Seq[Synapse]
    /** Map of id -> fan-in/fan-out for each neuron */
    protected[network] val idMap: Map[NeuronId, Int]

    /**
     * id Mapping setting method.
     * @return idMap instance
     */
    protected[network] def afterSynapseSetup() = synapses.foldLeft(Map[NeuronId, Int]()) {
      case (map, s@Synapse1(src1, dst, _)) ⇒
        map.+(src1 → (map.getOrElse(src1, 0) + 1), -dst → (map.getOrElse(-dst, 0) + 1))
      case (map, s@Synapse2(src1, src2, dst, _)) ⇒
        map.+(src1 → (map.getOrElse(src1, 0) + 1),
          src2 → (map.getOrElse(src2, 0) + 1),
          -dst → (map.getOrElse(-dst, 0) + 1))
      case (map, s@Synapse3(src1, src2, src3, dst, _)) ⇒
        map.+(src1 → (map.getOrElse(src1, 0) + 1),
          src2 → (map.getOrElse(src2, 0) + 1),
          src3 → (map.getOrElse(src3, 0) + 1),
          -dst → (map.getOrElse(-dst, 0) + 1))
      case (map, s@Synapse4(src1, src2, src3, src4, dst, _)) ⇒
        map.+(src1 → (map.getOrElse(src1, 0) + 1),
          src2 → (map.getOrElse(src2, 0) + 1),
          src3 → (map.getOrElse(src3, 0) + 1),
          src4 → (map.getOrElse(src4, 0) + 1),
          -dst → (map.getOrElse(-dst, 0) + 1))
    }

    /**
     * Sugar: All computational neurons
     * @return output neurons and hidden neurons.
     */
    def allNeurons = neurons union outNeurons

    /**
     * Sugar: All neural objects
     * @return output neurons, hidden neurons and synapses
     */
    def all = neurons union outNeurons union synapses

    /**
     * Online Training Procedure
     * @param error function for calculation
     * @param in is input vector for a iteration
     * @param real is output vector for a iteration
     * @return training functionality
     */
    def trainerOf(error: Objective)(in: NeuronVector, real: NeuronVector) = this backprop error.derivative(real, this apply in)

    /**
     * Backpropagation algorithm
     * @param err backpropagated error from error function
     */
    def backprop(err: NeuronVector) = {
      backpropRec(all, err mapValues { x ⇒ Seq(x)})
    }

    /**
     * Tail Recursion of Backpropagation
     * @param set to be propagated
     * @param delta is a mapping of errors
     */
    @tailrec
    private def backpropRec(set: Seq[NeuralObject], delta: Map[NeuronId, Seq[Scalar]]): Unit = {
      var inc = Seq[NeuralObject]()

      // update backpropagated errors
      val nDelta = delta ++? (set flatMap {
        case n@ComputeNeuron(id, _, _) if (delta sizeOf id) == idMap.getOrElse(id, 1) ⇒
          n.error(delta(id).sum)

        case s@Synapse1(src, dst, _) if delta contains -dst ⇒
          s.error(delta(-dst).sum)

        case s@Synapse2(src1, src2, dst, _) if delta contains -dst ⇒
          s.error(delta(-dst).sum)

        case s@Synapse3(src1, src2, src3, dst, _) if delta contains -dst ⇒
          s.error(delta(-dst).sum)

        case s@Synapse4(src1, src2, src3, src4, dst, _) if delta contains -dst ⇒
          s.error(delta(-dst).sum)

        case o if o.isInstanceOf[NeuralObject] ⇒
          inc = o +: inc
          Seq()
      })

      // exclude updated ones
      if (inc.nonEmpty)
        backpropRec(inc, nDelta)
    }

    /**
     * Apply updated delta values
     * @param method for update these values
     */
    def !(method: WeightUpdater): Unit = method(all)

    /**
     * Compute output of neural network with given input
     * @param in is an input vector
     * @return output of the vector
     */
    def apply(in: NeuronVector) = {
      val input = applyRec(neurons union synapses, in mapValues { x ⇒ Seq(x)})

      // collect all output values
      (outNeurons collect {
        case n@ComputeNeuron(id, _, _) if (input sizeOf -id) == idMap(-id) ⇒ id → n(input(-id).sum)
      }).toMap
    }

    /**
     * Tail Recursion of forward propagation
     * @param set to be propagated
     * @param input is computed input vector
     * @return
     */
    @tailrec
    private def applyRec(set: Seq[NeuralObject], input: Map[NeuronId, Seq[Scalar]]): Map[NeuronId, Seq[Scalar]] = {
      var exclude = Seq[NeuralObject]()

      // update intermediate inputs
      val nInput = input ++? (set collect {
        case n@ComputeNeuron(id, _, _) if (input sizeOf -id) == idMap(-id) ⇒
          exclude = n +: exclude
          id → n(input(-id).sum)
        // Negative sign(-) indicates intermediate input for neurons.

        case s@Synapse1(src, dst, _) if input contains src ⇒
          exclude = s +: exclude
          -dst → s(src → input(src).sum)

        case s@Synapse2(src1, src2, dst, _) if (input contains src1) && (input contains src2) ⇒
          exclude = s +: exclude
          -dst → s(src1 → input(src1).sum,
            src2 → input(src2).sum)

        case s@Synapse3(src1, src2, src3, dst, _)
          if (input contains src1) && (input contains src2) && (input contains src3) ⇒
          exclude = s +: exclude
          -dst → s(src1 → input(src1).sum,
            src2 → input(src2).sum,
            src3 → input(src3).sum)

        case s@Synapse4(src1, src2, src3, src4, dst, _)
          if (input contains src1) && (input contains src2) && (input contains src3) && (input contains src4) ⇒
          exclude = s +: exclude
          -dst → s(src1 → input(src1).sum,
            src2 → input(src2).sum,
            src3 → input(src3).sum,
            src4 → input(src4).sum)
      })

      // exclude updated ones
      val include = set diff exclude
      if (include.nonEmpty)
        applyRec(include, nInput)
      else
        nInput
    }

    /**
     * Generate subnet with given vertices
     * @param neurons is a set of all neurons including out Neurons
     * @param outNeurons is a set of only out neurons.
     * @return Sub network of this network
     */
    def subnetOf(neurons: Seq[NeuronId], outNeurons: Seq[NeuronId]) = new SubNetwork(this, neurons, outNeurons)

    /**
     * Serialize network to JSON
     * @return JsObject
     */
    def toJSON = {
      var weights = Json.obj()

      val neurons = this.neurons map { n ⇒
        weights +=(n.weight.id.toString, JsNumber(n.weight.value))
        n.toJSONObject
      }
      val output = this.outNeurons map { n ⇒
        weights +=(n.weight.id.toString, JsNumber(n.weight.value))
        n.toJSONObject
      }
      val synapses = this.synapses map { s ⇒
        weights +=(s.weight.id.toString, JsNumber(s.weight.value))
        s.toJSONObject
      }

      Json.obj(
        NEURONS → neurons,
        OUTPUT → output,
        SYNAPSES → synapses,
        WEIGHT → weights
      )
    }
  }

  /**
   * Basic Network implementation
   * @param neurons is a set of hidden neurons
   * @param synapses is a set of synapses
   * @param outNeurons is a set of output neurons
   */
  class BasicNetwork(override val neurons: Seq[Neuron],
                     override val synapses: Seq[Synapse],
                     override val outNeurons: Seq[Neuron])
    extends Network {
    /** Generate idMap */
    val idMap = afterSynapseSetup()
  }

  /**
   * Subnet implementation
   * @param net is a baseline network
   * @param subset is a set of all neurons
   * @param output is a set of only output neurons
   */
  class SubNetwork(net: Network, subset: Seq[NeuronId], output: Seq[NeuronId]) extends Network {
    /** Generate hidden neurons     */
    val neurons = net.neurons filter {
      case ComputeNeuron(id, _, _) ⇒ (subset contains id) && !(output contains id)
      case _ ⇒ false
    }

    /** Generate output neurons     */
    val outNeurons = net.allNeurons filter {
      case ComputeNeuron(id, _, _) ⇒ output contains id
      case _ ⇒ false
    }

    /** Generate synapses     */
    val synapses = net.synapses filter {
      case s@Synapse1(src, dst, _) ⇒
        (subset contains src) && (subset contains dst)
      case s@Synapse2(src1, src2, dst, _) ⇒
        (subset contains src1) && (subset contains src2) && (subset contains dst)
      case s@Synapse3(src1, src2, src3, dst, _) ⇒
        (subset contains src1) && (subset contains src2) && (subset contains src3) && (subset contains dst)
      case s@Synapse4(src1, src2, src3, src4, dst, _) ⇒
        (subset contains src1) && (subset contains src2) && (subset contains src3) &&
          (subset contains src4) && (subset contains dst)
    }

    /** Generate idMap */
    val idMap = afterSynapseSetup()
  }

  /**
   * Abstract class of encoder network
   */
  abstract class EncodedNetwork extends Network {
    /**
     * Generate Encoder subnet
     * @return a subnet exclude decoder layer
     */
    def encoderSubnet(): SubNetwork
  }

  /**
   * 1-layer Auto Encoder
   * @param input the number of input neurons
   * @param hidden the number of hidden neurons
   * @param encoder the encoder layer activation function (decoder generated by TanH)
   * @param start starting index of neurons
   */
  class AutoEncoder(val input: NeuronId,
                    val hidden: NeuronId,
                    encoder: Activation,
                    val start: NeuronId = 1)
    extends EncodedNetwork {
    /** Weight initializer     */
    private val weightGen = Weight.initialize
    private val weightScalar = encoder.initialize(input, hidden)

    /** Generate Neurons     */
    val neurons = (input + start) until (input + hidden + start) map
      { id ⇒ ComputeNeuron(id, weightGen(weightScalar), encoder)}

    /** Generate output neurons     */
    val outNeurons = (input + hidden + start) until (input * 2 + hidden + start) map
      { id ⇒ ComputeNeuron(id, weightGen(weightScalar), encoder)}

    /** Generate synapses  */
    val synapses = 0 until input flatMap {
      in ⇒ 0 until hidden flatMap {
        out ⇒ {
          val w = weightGen(weightScalar)
          val src = in + start
          val hid = input + out + start
          val dec = input + hidden + src
          Seq(Synapse1(src, hid, w), Synapse1(hid, dec, w))
        }
      }
    }

    /** Generate idMap */
    val idMap = afterSynapseSetup()

    /**
     * Generate Encoder subnet
     * @return a subnet exclude decoder layer
     */
    def encoderSubnet() =
      this.subnetOf(start until (input + hidden + start), (input + start) until (input + hidden + start))
  }

  /**
   * Stack two network (numbering is already agreed)
   * @param input is input-layer network
   * @param output is output-layer network
   */
  class StackedNetwork(input: Network, output: Network) extends EncodedNetwork {
    /** Generate neurons */
    val neurons = input.allNeurons union output.neurons
    /** Generate output neurons */
    val outNeurons = output.outNeurons
    /** Generate synapses */
    val synapses = input.synapses union output.synapses
    /** Generate idMap */
    val idMap = afterSynapseSetup()
    /** Generate encoder output */
    val encoderOut = output.neurons collect { case ComputeNeuron(id, _, _) ⇒ id}

    /**
     * Generate Encoder subnet
     * @return a subnet exclude decoder layer
     */
    def encoderSubnet() = {
      val neuronSet = neurons collect { case ComputeNeuron(id, _, _) ⇒ id}
      val outSet = encoderOut
      this.subnetOf(neuronSet, outSet)
    }
  }

  /**
   * Companion object of Network
   */
  object Network {
    /**
     * Load network from JsObject
     * @param obj to be parsed
     * @return New Basic Network reconstructed from this object
     */
    def fromJSON(obj: JsObject) = {
      // Parse Weights
      val weights = ((obj \ WEIGHT).as[Map[String, Double]] map
        { pair ⇒ pair._1 → new Weight(pair._1.toLong, pair._2)}).toMap

      // Neuron parsers
      val neuronConverter = (obj: JsObject) ⇒
        new ComputeNeuron(
          (obj \ ID).as[NeuronId],
          weights((obj \ WEIGHTID).as[String]),
          (obj \ ACTIVATION).as[String] match {
            case SIGMOID ⇒ Sigmoid
            case RELU ⇒ Rectifier
            case SOFTPLUS ⇒ Softplus
            case _ ⇒ HyperbolicTangent
          })

      // Synapse parsers
      val synapseConverter = (obj: JsObject) ⇒ {
        val source = (obj \ SOURCE).as[List[NeuronId]]
        val dst = (obj \ DESTINATION).as[NeuronId]
        val weight = weights((obj \ WEIGHTID).as[String])
        source.size match {
          case 1 ⇒ Synapse1(source(0), dst, weight)
          case 2 ⇒ Synapse2(source(0), source(1), dst, weight)
          case 3 ⇒ Synapse3(source(0), source(1), source(2), dst, weight)
          case 4 ⇒ Synapse4(source(0), source(1), source(2), source(3), dst, weight)
        }
      }

      val neurons = (obj \ NEURONS).as[Seq[JsObject]] map neuronConverter
      val outputs = (obj \ OUTPUT).as[Seq[JsObject]] map neuronConverter
      val synapse = (obj \ SYNAPSES).as[Seq[JsObject]] map synapseConverter
      new BasicNetwork(neurons, synapse, outputs)
    }

    /**
     * Construct network from given synapse information
     * @param act for activation function
     * @param connections for synapse specifiaction
     * @return a new BasicNetwork
     */
    def fromConnection(act: Activation, connections: Seq[(NeuronId, NeuronId)]):Network = {
      val fanIO = connections flatMap {
        pair ⇒ Seq(-pair._1 → 1, pair._2 → 1)
      } groupBy (_._1) mapValues (seq ⇒ seq.foldLeft(0)((a, p) ⇒ a + p._2))

      val scalar = (id: NeuronId) ⇒ act.initialize(fanIO.getOrElse(id, 0), fanIO.getOrElse(-id, 0))
      val init = Weight.initialize

      val synapses = connections map {
        pair ⇒ {
          val in = pair._1
          val out = pair._2
          Synapse1(in, out, init(scalar(out)))
        }
      }

      val neurons = fanIO.keySet collect {
        case key if key > 0 ⇒ (fanIO contains -key) → ComputeNeuron(key, init(scalar(key)), act)
      } groupBy (_._1) mapValues { seq ⇒ seq map (_._2)}

      new BasicNetwork(neurons.getOrElse(true, Seq()).toSeq, synapses, neurons.getOrElse(false, Seq()).toSeq)
    }

    /**
     * Construct network from given layer size information
     * @param act for activation function
     * @param layerSizes for construct layers
     */
    def apply(act: Activation, layerSizes: NeuronId*):Network = {
      var synapses = Seq[(NeuronId, NeuronId)]()
      (1 until layerSizes.size).foldLeft(1) {
        (start, i) ⇒ {
          val inSize = start + layerSizes(i-1)
          val outSize = inSize + layerSizes(i)
          for (in ← start until inSize; out ← inSize until outSize){
            synapses = (in → out) +: synapses
          }
          inSize
        }
      }

      fromConnection(act, synapses)
    }
  }

}
